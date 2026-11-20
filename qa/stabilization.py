"""Проверяет откат в задаче Сода и глобальные счётчики первого порядка в Report."""
import re
import shutil
from pathlib import Path

from mpi_common import report_rows, ROOT, mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run


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


if __name__ == "__main__":
    parser = mpi_parser(__doc__)
    parser.add_argument("--devices", help="two comma-separated GPU IDs for the MPI run")
    with mpi_workspace("stabilization-") as tmp:
        check(parser.parse_args(), Path(tmp))
