"""Проверяет диагностику входных файлов при запуске и в каждом отчёте."""
import os
import re
import shutil
from pathlib import Path

from mpi_common import ROOT, mpi_workspace
from mpi_geometry import run


def check(case):
    source = ROOT / "qa/cases/pulse_1d"
    for path in source.glob("*.txt"):
        shutil.copy(path, case)
    (case / "exact.txt").write_text("rho=1\nux=1\np=1\n")
    for name in ("main", "gas", "scheme", "output", "exact", "init", "bc"):
        with (case / (name + ".txt")).open("a") as file:
            file.write(f"\ntypo_{name} = 1\n")
    cmd = [ROOT / "bin" / ("aero.exe" if os.name == "nt" else "aero"), f"--input={case}", f"--output={case / 'out'}",
           f"main:mesh={source / 'grid.mesh'}", "main:exact_solution=on",
           "main:report_every=1", "main:final_time=.01"]
    log = run(cmd)
    reports = len(re.findall(r"^Report\s+Step =", log, re.M))
    assert reports >= 2, log
    for name in ("main", "gas", "scheme", "output", "exact", "init", "bc"):
        assert len(re.findall(rf"^WARNING: .*\btypo_{name}\b", log, re.M)) == reports + 1, log
    for name, key in (("main", "backend"), ("gas", "gamma"), ("exact", "ux"), ("init", "rho")):
        path = case / (name + ".txt")
        original = path.read_text()
        path.write_text(re.sub(rf"(?m)^\s*{key}\s*=.*$", "", original))
        log = run(cmd, success=False)
        assert key in log and "Starting time integration" not in log, log
        path.write_text(original)


if __name__ == "__main__":
    with mpi_workspace("settings-warnings-") as directory:
        check(Path(directory))
    print("OK required parameters and report warnings")
