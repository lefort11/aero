"""Проверяет ограничение отношения локальных псевдошагов на соседних узлах."""

import math
import shutil
from pathlib import Path

from mpi_common import ROOT, mpi_parser, mpi_workspace, report_rows
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from viscosity import max_difference


def prepare(case):
    """Первый отрезок линии в 256 раз короче остальных. Скорость звука всюду одинакова."""
    short = 1 / 2048
    positions = [0.] + [short + i / 8 for i in range(8)]
    sparse_line(case / "grid.mesh", len(positions), positions)
    (case / "main.txt").write_text(
        "mesh=grid.mesh\nbackend=cpu\nomp_threads=2\ntime_scheme=steady\n"
        "cfl_method=height\ncfl=.05\nreconstruction=first_order\nreconstruction_variables=characteristic\n"
        "euler_solver=lf\nreport_every=1\n"
    )
    (case / "gas.txt").write_text("gamma=1.4\n")
    # Градиент давления запускает движение. При постоянном p/rho первый шаг
    # определяется только длиной меньшего из отрезков, прилегающих к узлу.
    (case / "init.txt").write_text("initial_type=formula\nrho=1+.2*x\np=1+.2*x\nux=0\n")
    (case / "bc.txt").write_text("[xmin, xmax]\nbc_type novisc_wall\n")
    (case / "output.txt").write_text(
        "[final]\nfile=final.fields\nvariables=rho,ux,p\nevery_steps=3\nformat=binary\nlayout=global\n"
    )
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")


def solve(case, args, label, *, limit=None, global_limit=None, mpi=False, gpu=False, residual_limit=False):
    steps = 68 if residual_limit else 3
    command = [args.mpiexec, "-n", "2"] if mpi else []
    command += [args.aero, f"--input={case}", f"--output={case / label}",
                "implicit:local_time_step=auto",
                f"implicit:steady_max_steps={steps}", "implicit:steady_tol=1e-14"]
    if residual_limit:
        command += ["implicit:local_time_step_residual_limit=on", "implicit:local_time_step_probe_node=7"]
    if global_limit is not None:
        command.append(f"implicit:local_time_step_max_ratio={global_limit}")
    if limit is not None:
        command.append(f"implicit:local_time_step_neighbor_ratio={limit}")
    if mpi:
        command.append("main:mesh=grid.mdec")
    if gpu:
        command.append("main:backend=cuda")
    log = run(command, success=False)
    if "Steady: pseudo-step limit reached before convergence" not in log:
        raise AssertionError(log)

    # На первом шаге допустимые множители по пути равны 1,1,2,4,...,128.
    # При одном общем потолке 2 вместо 128 получилось бы 2, а при одном
    # обходе соседей осталось бы 256. Передача через границу MPI тоже обязательна.
    ranges = list(report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)"))
    if [int(step) for step, _, _ in ranges] != list(range(1, steps+1)):
        raise AssertionError(log)
    _, smallest, largest = ranges[0]
    measured = float(largest) / float(smallest)
    before = min(256, global_limit or 256)
    # К самому дальнему узлу от ближайшего малого шага ведут семь рёбер.
    expected = min(before, limit ** 7) if limit else before
    if not math.isclose(measured, expected, rel_tol=2.e-6):
        raise AssertionError(f"{label}: неверный разброс шагов {measured:g}")
    if limit:
        rows = list(report_rows(log, r"^LTS edge ratio\s+(\S+)\s+->\s+(\S+)\s+limit = (\S+)"))
        if ([int(step) for step, *_ in rows] != list(range(1, steps+1))
                or not math.isclose(float(rows[0][1]), before, rel_tol=2.e-6)
                or not math.isclose(float(rows[0][2]), min(before, limit), rel_tol=2.e-6)
                or any(not 1 <= float(after) <= limit * (1 + 1.e-6) or float(setting) != limit
                       for _, _, after, setting in rows)):
            raise AssertionError(log)
    if residual_limit:
        # До первой полной серии из 32 принятых шагов история только накапливается.
        # На этой неоднородной линии затем растёт невязка лишь в части узлов.
        counts = [int(n) for _, n in report_rows(log, r"^LTS residual limit  nodes = (\d+)")]
        if len(counts) != steps or any(counts[:32]) or not 0 < max(counts[32:]) < 9:
            raise AssertionError(f"{label}: неверная активация узлового ограничения {counts}")
        probe = list(report_rows(log, r"^LTS probe +node = 7 +dt = (\S+) +gas rate = (\S+) +SA rate = (\S+) +cap/CFL step = (\S+)"))
        if len(probe) != steps:
            raise AssertionError(f"{label}: неполная история выбранного узла")
        for (_, dt, gas, sa, cap), (_, minimum, maximum) in zip(probe, ranges):
            if not (float(minimum) <= float(dt) <= float(maximum) and float(gas) > 0
                    and float(sa) == 0 and 0 < float(cap) <= 1):
                raise AssertionError(f"{label}: неверный шаг или невязка узла 7")
    metadata, fields = binary_snapshot(case / label / "final.fields")
    if metadata[3] != steps or set(fields) != {"rho", "ux", "p"}:
        raise AssertionError(f"{label}: неполный снимок последнего шага")
    return fields


def main():
    parser = mpi_parser(__doc__)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    with mpi_workspace("lts-neighbors-") as work:
        case = Path(work)
        prepare(case)
        original = solve(case, args, "default")
        expected = solve(case, args, "limited", limit=2)
        if max_difference(original, expected) < 1.e-7:
            raise AssertionError("Ограничение не изменило решение на неоднородной сетке")
        # Предел 1 делает шаги на всей линии одинаковыми. Независимый эталон
        # даёт уже существующее глобальное ограничение при том же значении.
        uniform = solve(case, args, "global-1", global_limit=1)
        if max_difference(uniform, solve(case, args, "neighbor-1", limit=1)) > 2.e-11:
            raise AssertionError("При отношении 1 не воспроизведён единый шаг")
        combined = solve(case, args, "combined", limit=2, global_limit=5)
        feedback = solve(case, args, "feedback", limit=2, residual_limit=True)
        if args.mpi:
            run([args.meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2"])
        if args.gpu:
            (case / "cuda.txt").write_text(f"devices={args.device}\n")
        for label, use_mpi, use_gpu in (("MPI", args.mpi, False), ("GPU", False, args.gpu)):
            if use_mpi or use_gpu:
                fields = solve(case, args, label, limit=2, mpi=use_mpi, gpu=use_gpu)
                delta = max_difference(expected, fields)
                if delta > 2.e-11:
                    raise AssertionError(f"{label}: максимальное расхождение полей {delta:g}")
                controlled = solve(case, args, label + "-feedback", limit=2, mpi=use_mpi, gpu=use_gpu, residual_limit=True)
                feedback_delta = max_difference(feedback, controlled)
                if feedback_delta > 2.e-10:
                    raise AssertionError(f"{label}: различаются поля после двух проверок местной невязки")
                other = solve(case, args, label + "-combined", limit=2, global_limit=5, mpi=use_mpi, gpu=use_gpu)
                combined_delta = max_difference(combined, other)
                if combined_delta > 2.e-11:
                    raise AssertionError(f"{label}: расхождение при двух ограничениях {combined_delta:g}")
                print(f"PASS LTS neighbor ratio {label}, maximum field difference {delta:.3g}, combined {combined_delta:.3g}, residual limit {feedback_delta:.3g}")
        print("PASS LTS neighbor ratio on a graded line")


if __name__ == "__main__":
    main()
