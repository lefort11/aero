"""Проверяет откат в задаче Сода и глобальные счётчики первого порядка в Report."""
import math
import re
import shutil
from pathlib import Path

from mpi_common import report_rows, ROOT, mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run


def check_auto_retry_steps(log):
    """Сопоставляет каждый повтор отклонённого шага с фактически принятым минимумом dt."""
    attempts = {}
    pattern = (r"^RETRY step (\d+) \| CFL (\S+) -> (\S+) \| failed nodes = [1-9]\d*"
               r" \| dt_min (\S+) -> (\S+)$")
    for step, old_cfl, new_cfl, old_dt, new_dt in re.findall(pattern, log, re.M):
        # Вся попытка должна уменьшиться в той же пропорции, в том числе если
        # нижний шаг auto до отказа уже был меньше шага, вычисленного из CFL.
        step = int(step)
        old_dt, new_dt = float(old_dt), float(new_dt)
        assert math.isclose(new_dt / old_dt, float(new_cfl) / float(old_cfl), rel_tol=2e-14), log
        if step in attempts:
            assert old_dt == attempts[step], log
        attempts[step] = new_dt
    accepted = {int(step): float(dt) for step, dt in report_rows(log, r"^dt_min = (\S+)")}
    assert 2 in attempts and attempts.keys() <= accepted.keys(), log
    for step, final_dt in attempts.items():
        # В report печатаются шесть знаков после десятичной точки.
        assert math.isclose(accepted[step], final_dt, rel_tol=1e-6), log
    assert "next CFL is below cfl_min" in log, log


def check_boundary_retry(args, case, mesh, archive):
    # Возмущение давления подходит к правой границе. Узловые плотность и давление
    # ещё положительны, но дозвуковое условие уже отклоняет поправку Ньютона.
    (case / "init.txt").write_text("initial_type formula\nrho=1\nux=0.02\np=if(x < 0.99,0.5,1)\n")
    (case / "bc.txt").write_text(
        "[xmin]\nbc_type free\nsuperin initial\ninflow base_p\noutflow base_u\n"
        "[xmax]\nbc_type free\nsuperin initial\ninflow base_u initial\noutflow base_p initial\n"
    )
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")
    for parallel in (False, True):
        cmd = [args.mpiexec, "-n", "2", args.aero] if parallel else [args.aero]
        if parallel and args.devices:
            cmd += ["main:backend=cuda", f"cuda:devices={args.devices}"]
        cmd += [f"--input={case}", f"main:mesh={archive if parallel else mesh}",
                "main:stabilization=low", "cfl:cfl_init=.5", "cfl:cfl_max=.5",
                "cfl:cfl_min=.05", "main:report_every=1"]
        log = run([*cmd, f"--output={case / f'boundary-{parallel}'}", "main:time_scheme=bdf1"], success=False)
        assert "RETRY step 2 | CFL 0.5 -> 0.25 | failed nodes = 1" in log, log
        assert re.search(r"STEP 2 +\|.*?failed since last report = 1 +first-order now = 1", log, re.S), log
        assert "next CFL is below cfl_min" in log, log
        steady = run([*cmd, f"--output={case / f'auto-retry-{parallel}'}", "main:time_scheme=steady",
                      "implicit:local_time_step=auto", "implicit:local_time_step_neighbor_ratio=2",
                      "implicit:local_time_step_max_ratio=off", "implicit:steady_max_steps=12"], success=False)
        check_auto_retry_steps(steady)
        print(f"OK steady auto retries after boundary rejection, MPI={parallel}", flush=True)


def check(args, case):
    source = ROOT / "qa/cases/sod_1d"
    for path in source.glob("*.txt"):
        shutil.copyfile(path, case / path.name)
    shutil.copyfile(ROOT / "templates/cfl.txt", case / "cfl.txt")
    (case / "output.txt").write_text(
        "[all]\nfile=final.fields\nevery_time=.04\nlayout=global\nmulti=0\nformat=binary\nvariables=rho,ux,p,E\n"
    )
    (case / "cuda.txt").write_text("devices=0\n")
    archive = case / "sod.mdec"
    mesh = source / "grid.mesh"
    run([args.meshdec, mesh, archive, "--target-ranks=2", "--compress"])
    # Уменьшаем долю сетки master: узлы у разрыва обслуживает второй процесс.
    common = [f"--input={case}", "main:time_scheme=euler", "main:final_time=.04", "main:cfl=4",
              "cfl:cfl_init=4", "cfl:cfl_max=4", "cfl:cfl_num_steps=3", "main:report_every=2"]
    mpi = [args.mpiexec, "-n", "2", args.aero]
    for mode in ("off", "cfl", "low", "on"):
        reference = None
        for parallel in (False, True):
            dest = case / f"{mode}-{parallel}"
            cmd = mpi if parallel else [args.aero]
            opts = ["main:mpi_rank_weights=1,8"] if parallel else []
            if parallel and args.devices:
                opts += ["main:backend=cuda", f"cuda:devices={args.devices}"]
            cmd = [*cmd, *common, *opts, f"main:mesh={archive if parallel else mesh}", f"--output={dest}",
                   f"main:stabilization={mode}"]
            log = run(cmd, success=mode != "off")
            if mode == "off":
                assert "stage" in log and "RETRY step" not in log
                continue
            reductions = re.findall(r"^RETRY step (\d+) \| CFL (\S+) -> (\S+) \| failed nodes = [1-9]\d*$", log, re.M)
            assert reductions and len(reductions) == log.count("RETRY step"), log
            assert all(float(new) == float(old) / 2 for _, old, new in reductions), reductions
            # Строка после нормы энергии относится к отчёту за принятый шаг.
            counts = list(report_rows(log, r"^Stabilization .*failed since last report = (\d+)\s+first-order now = (\d+)"))
            assert counts and any(int(failed) for _, failed, _ in counts)
            assert any(int(low) for _, _, low in counts) == (mode in ("low", "on"))
            _, fields = binary_snapshot(dest / "final.fields")
            if reference is None:
                reference = counts, fields, reductions
            else:
                assert reductions == reference[2]  # Каждый отказ печатает только процесс 0.
                assert counts == reference[0]
                for key in fields:
                    assert max(abs(a-b) for a,b in zip(fields[key], reference[1][key])) < 1e-11
        if mode == "cfl":
            # Уменьшение CFL сообщается сразу, даже когда обычный отчёт отключён.
            log = run([args.aero, *common, f"main:mesh={mesh}", f"--output={case / 'quiet'}",
                       "main:stabilization=cfl", "main:report_every=0"])
            assert "CFL 4 -> 2 |" in log, log
        print("OK stabilization", mode)
    check_boundary_retry(args, case, mesh, archive)


if __name__ == "__main__":
    parser = mpi_parser(__doc__)
    parser.add_argument("--devices", help="two comma-separated GPU IDs for the MPI run")
    with mpi_workspace("stabilization-") as tmp:
        check(parser.parse_args(), Path(tmp))
