"""Двухуровневая поправка steady в последовательном расчёте и при разбиении MPI."""
import measurements
import math
import re
from pathlib import Path

from local_step_neighbors import prepare
from mpi_common import ROOT, backend_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from viscosity import max_difference
from turbulence import configure as prepare_rans


def solve(case, args, label, distributed=False, gpu=False, enabled=True, end=6, resume=False,
          local_steps="on"):
    """Проверяет каждое принятие по полной невязке, затем читает итоговое поле."""
    directory = case / label
    command = [args.mpiexec, "-n", "2"] if distributed else []
    command += [args.aero, f"--input={case}", f"--output={directory}",
                "main:time_scheme=steady", f"implicit:local_time_step={local_steps}",
                "main:backend=" + ("cuda" if gpu else "cpu"),
                "main:mesh=" + ("grid.mdec" if distributed else "grid.mesh"),
                "implicit:steady_multigrid=" + ("on" if enabled else "off"),
                f"implicit:steady_max_steps={end}", "implicit:steady_tol=1e-14"]
    if resume:
        command.append("--recover")
    log = run(command, success=False)
    if (f"local_time_step = {local_steps}\n" not in log or
            "pseudo-step limit reached before convergence" not in log):
        raise AssertionError(log)
    trials = re.findall(r"Steady multigrid: (accepted|skipped), scale = (\S+), spatial residual = (\S+) -> ([\d.eE+-]+)", log)
    if not resume and measurements.compare(enabled, "!=", bool(trials)):
        raise AssertionError(log)
    for status, fraction, before, after in trials:
        previous, current = float(before), float(after)
        if status == "accepted":
            if not (measurements.compare(0, "<", float(fraction)) and measurements.compare(current, "<", previous)):
                raise AssertionError((status, before, after))
        elif measurements.compare(previous, "!=", current) or measurements.compare(float(fraction), "!=", 0):
            raise AssertionError((status, before, after))
    output = next(directory.glob("resumed-*")) if resume else directory
    header, fields = binary_snapshot(output / "final.fields")
    if measurements.compare(header[3], "!=", end) or not all(math.isfinite(x) for column in fields.values() for x in column):
        raise AssertionError(f"Invalid final state in {directory}")
    return fields, sum(row[0] == "accepted" for row in trials)


def check_rans(case, args):
    """При локальных шагах сравнивает газ и рабочую вязкость SA после одних и тех же циклов."""
    case.mkdir()
    prepare_rans(case,args)
    (case / "cuda.txt").write_text(f"devices={args.device}\n")
    (case / "output.txt").write_text(
        "[state]\nfile=final.fields\nformat=binary\nlayout=global\nevery_steps=6\n"
        "variables=rho,ux,uy,p,nu_tilde,nut\n")
    reference, _ = solve(case,args,"cpu",local_steps="auto")
    if measurements.compare(min(reference["nu_tilde"]), "<", 0.):
        raise AssertionError("Negative working viscosity")
    if args.gpu:
        fields, _ = solve(case,args,"gpu",gpu=True,local_steps="auto")
        if measurements.compare(max_difference(reference,fields), ">", 1.e-9):
            raise AssertionError("RANS multigrid CPU/GPU mismatch")
    if args.mpi:
        fields, _ = solve(case,args,"mpi",distributed=True,local_steps="auto")
        if measurements.compare(min(fields["nu_tilde"]), "<", 0.):
            raise AssertionError("Negative working viscosity in MPI")


# Разбирает параметры запуска, выполняет набор QA-проверок и возвращает статус результата.
def main():
    parser = backend_parser(__doc__, single_device=True)
    args = parser.parse_args()
    with mpi_workspace("steady-amg-") as work:
        case = Path(work)
        prepare(case)
        if args.gpu:
            (case / "cuda.txt").write_text(f"devices={args.device}\n")
        # Гладкая начальная неоднородность. Несколько агрегатов пересекают
        # область градиента, а между разделами MPI остаётся ненулевой поток.
        sparse_line(case / "grid.mesh", 33, [i/32 for i in range(33)])
        if args.mpi:
            run([args.meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2"])
        reference, count = solve(case,args,"cpu")
        ordinary, _ = solve(case,args,"ordinary",enabled=False)
        if not count or measurements.compare(max_difference(reference,ordinary), "==", 0.):
            raise AssertionError("Coarse correction was never applied on the smooth line")
        # Сохраняем после второго обычного шага внутри цикла. Сразу после
        # перезапуска счётчик должен продолжиться из recovery.
        solve(case,args,"restart",end=3)
        continued, _ = solve(case,args,"restart",resume=True)
        if measurements.compare(max_difference(reference,continued), ">", 1.e-13):
            raise AssertionError("Restart changed the multigrid cycle")
        if args.gpu:
            fields, accepted = solve(case,args,"gpu",gpu=True)
            if measurements.compare(accepted, "!=", count) or measurements.compare(max_difference(reference,fields), ">", 1.e-9):
                raise AssertionError("CPU/GPU multigrid mismatch")
        if args.mpi:
            _, accepted = solve(case,args,"mpi",distributed=True)
            if not accepted:
                raise AssertionError("MPI coarse correction was never applied")
        if (ROOT / "source/turbulence/sa.h").is_file() and (ROOT / "source/viscosity/transport.h").is_file():
            check_rans(case / "rans",args)
    print("Steady multigrid QA passed")


if __name__ == "__main__":
    main()
