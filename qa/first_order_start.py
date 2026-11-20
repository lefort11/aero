"""Сверяет автоматическую смену порядка с двумя отдельными расчётами через контрольную точку."""
import re
import shutil
from pathlib import Path

from mpi_common import report_rows, ROOT, mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run
from run import settings


def check(args, work, name, equation, time_scheme, parallel):
    case = work / f"{name}-{time_scheme}-{parallel}"
    case.mkdir()
    sample = ROOT / "qa/cases" / name
    for path in sample.glob("*.txt"):
        shutil.copyfile(path, case / path.name)
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")
    mesh = sample / settings(sample / "main.txt")["mesh"]
    variables = re.search(r"(?m)^variables\s*=\s*(.+)$", (sample / "output.txt").read_text())[1]
    solver = [args.aero]
    options = []
    if parallel:
        archive = case / "mesh.mdec"
        run([args.meshdec, mesh, archive, "--target-ranks=2", "--compress"])
        mesh = archive
        solver = [args.mpiexec, "-n", "2", args.aero]
        if args.devices:
            (case / "cuda.txt").write_text(f"devices={args.devices}\n")
            options.append("main:backend=cuda")
    (case / "output.txt").write_text("[all]\nfile=state.fields\nevery_steps=1\n"
                                     f"layout=global\nformat=binary\nvariables={variables}\n")
    # Порог специально не совпадает ни с шагом CFL, ни со сроком записи по времени.
    switch, end = .00713, .016
    common = [*solver, f"--input={case}", f"main:mesh={mesh.resolve()}", f"main:time_scheme={time_scheme}",
              f"main:Equations={equation}", "main:report_every=1", *options]
    selected = "main:reconstruction=weno5"
    split, auto = case / "split", case / "auto"
    run([*common, f"--output={split}", "main:reconstruction=first_order", f"main:final_time={switch}",
         "main:recovery_every_steps=1000"])
    checkpoint, first_fields = binary_snapshot(split / "state.fields")
    assert checkpoint[-1] == switch
    run([*common, f"--output={split}", "--recover", selected, f"main:final_time={end}"])
    expected, expected_fields = binary_snapshot(next(split.glob("resumed-*/state.fields")))
    log = run([*common, f"--output={auto}", selected, f"main:t_fo={switch}", f"main:final_time={end}"])
    actual, actual_fields = binary_snapshot(auto / "state.fields")
    assert actual == expected and actual_fields != first_fields
    for field in actual_fields:
        assert max(abs(a-b) for a,b in zip(actual_fields[field], expected_fields[field])) < 1e-12, field
    reports = [(int(n), float(t)) for n,t in report_rows(log, r"^(?:t|Pseudo-time) = (\S+)")]
    assert (checkpoint[-2], switch) in reports
    if equation == "Linear":
        assert all(re.search(r"(?m)^" + name + r"'\s+", log) for name in ("rho", "ux", "p", "E"))
        assert not re.search(r"(?m)^Stabilization\s+", log)
    if equation == "Euler":
        counts = {int(n): int(c) for n,c in report_rows(log, r"^Stabilization .*failed since last report = 0\s+first-order now = (\d+)")}
        for step, time in reports:
            assert (counts[step] > 0) == (time <= switch)
    print(f"OK {equation} {time_scheme}, MPI={parallel}: transition at t={switch}")


if __name__ == "__main__":
    parser = mpi_parser(__doc__)
    parser.add_argument("--devices", help="two GPU IDs for the MPI run")
    args = parser.parse_args()
    with mpi_workspace("first-order-start-") as tmp:
        for mpi in (False, True):
            for name, model, scheme in (("pulse_1d", "Euler", "ssprk3"), ("wave_3d", "Euler", "euler"),
                                        ("linear_acoustic_1d", "Linear", "bdf2")):
                check(args, Path(tmp), name, model, scheme, mpi)
