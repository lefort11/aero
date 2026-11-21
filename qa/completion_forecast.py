"""Проверка периодичности и вида прогноза, в том числе при продолжении расчёта."""

import re
import shutil
from pathlib import Path

from mpi_common import ROOT, mpi_parser, mpi_workspace
from mpi_geometry import run


def forecasts(log):
    return re.findall(r"^ETA  step (\d+)  \|  (.+)$", log, re.M)


def main():
    args = mpi_parser(__doc__).parse_args()
    with mpi_workspace("eta-") as work:
        work = Path(work)
        case = work/"input"
        shutil.copytree(ROOT/"qa/cases/pulse_1d",case)
        shutil.copyfile(ROOT/"templates/implicit.txt",case/"implicit.txt")
        solver = [args.aero,f"--input={case}","main:report_every=0","main:reconstruction=first_order"]
        for scheme in ("euler","ssprk3","BDF1","BDF2"):
            cmd = [*solver,"main:time_scheme="+scheme,f"--output={work/scheme}"]
            if forecasts(run([*solver, "main:time_scheme="+scheme, f"--output={work/(scheme+'off')}", "main:final_time=.024"])):
                raise AssertionError("Default interval should disable ETA")
            log = run([*cmd,"main:final_time=.024","main:prophecy_every_step=2"])
            steps = int(re.search(r"Completed (\d+) steps",log)[1])
            rows = forecasts(log)
            if [int(n) for n,_ in rows] != list(range(2,steps+1,2)) or len(rows)<2:
                raise AssertionError(f"Incorrect ETA interval for {scheme}: {rows}")
            if not re.search(r"final time: \d{4}-\d\d-\d\d \d\d:\d\d:\d\d", rows[0][1]):
                raise AssertionError(f"Missing finish date for {scheme}")
            resumed = run([*cmd,"main:prophecy_every_step=2","--recover","main:final_time=.036"])
            restart_rows = forecasts(resumed)
            if not restart_rows or int(restart_rows[0][0]) != steps//2*2+2:
                raise AssertionError(f"Wrong first forecast after recovery: {restart_rows}")
            print(f"PASS ETA {scheme} and recovery",flush=True)
        for criterion in ("relative","absolute"):
            for local in ("on","off"):
                log = run([*solver,f"--output={work/(criterion+local)}","main:time_scheme=steady",
                           "implicit:local_time_step="+local,"implicit:steady_criterion="+criterion,
                           "implicit:steady_tol=1e-14","implicit:steady_max_steps=5",
                           "main:prophecy_every_step=2"],success=False)
                rows = forecasts(log)
                if [n for n,_ in rows]!=["2","4"] or not all("residual target:" in s and "step limit:" in s for _,s in rows):
                    raise AssertionError(f"Wrong steady forecasts: {rows}\n{log[-900:]}")
                if "limit reached before convergence" not in log:
                    raise AssertionError("Unexpected steady termination")
                print(f"PASS steady ETA {criterion}, local={local}",flush=True)


if __name__ == "__main__":
    main()
