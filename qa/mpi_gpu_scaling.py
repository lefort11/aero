"""Измеряет время одной и той же задачи на одной и двух видеокартах при разных весах разбиения."""

from mpi_common import ROOT, mpi_parser
import csv
import math
from pathlib import Path
import re
import shutil
import statistics
import time

from mpi_euler import launch
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from run import settings


def benchmark(work, aero, meshdec, launcher, devices):
    """Чередует запуски на одной и двух картах и сравнивает времена для одной фиксированной сетки."""
    work.mkdir(parents=True, exist_ok=False)
    case = work / "case"
    shutil.copytree(ROOT / "qa/cases/sod_1d", case)
    sparse_line(case / "grid.mesh", 65536)
    run([meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=32"])
    (case / "init.txt").write_text(
        "initial_type formula\nrho = 1+.05*sin(2*pi*x)\nux = .2\np = 1+.02*cos(2*pi*x)\n"
    )
    final = 0.0005
    (case / "output.txt").write_text(
        f"[final]\nfile = final.fields\nevery_time = {final}\nlayout = global\n"
        "format = binary\nvariables = rho, ux, p, E\n"
    )
    options = settings(case / "main.txt")
    options.update(
        mesh="grid.mdec",
        backend="cuda",
        reconstruction="weno5",
        reconstruction_variables="characteristic",
        euler_solver="roe",
        time_scheme="ssprk3",
        final_time=final,
        cfl=0.35,
        mpi_timing=1,
        report_every=0,
        recovery_every_steps=1000000,
        omp_threads=1,
    )
    modes = [("single", devices[:1], "1"), ("equal", devices, "1,1"), ("auto", devices, "auto")]
    reference = None
    rows = []
    for repeat in range(3):
        for mode, cards, weights in modes[repeat:] + modes[:repeat]:
            options["mpi_rank_weights"] = weights
            (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))
            (case / "cuda.txt").write_text("devices = " + ",".join(cards) + "\n")
            out = work / f"{mode}-{repeat+1}"
            start = time.perf_counter()
            log = launch(aero, case, out, launcher, len(cards))
            elapsed = time.perf_counter() - start
            out.with_suffix(".log").write_text(log, encoding="utf-8")
            header = re.search(r"GPU timings: ranks=(\d+) start_step=0 end_step=(\d+) steps=(\d+)", log)
            if not header or int(header[1]) != len(cards) or header[2] != header[3]:
                raise AssertionError(log)
            if mode == "single":
                if f"Active backend = CUDA device {cards[0]}" not in log:
                    raise AssertionError(log)
            else:
                assigned = re.findall(r"MPI CUDA placement: rank \d+ local_rank \d+ device (\d+)", log)
                if assigned != cards or ("MPI GPU calibration: rank" in log) != (mode == "auto"):
                    raise AssertionError(log)
            table = log.split("max_s_per_step\n")[1].split("Completed")[0]
            phases = {line.split()[0]: float(line.split()[4]) for line in table.splitlines()}
            snapshot = binary_snapshot(out / "final.fields")
            if reference is None:
                reference = snapshot
            if (
                snapshot[0] != reference[0]
                or snapshot[0][-2:] != (int(header[3]), final)
                or snapshot[1].keys() != reference[1].keys()
            ):
                raise AssertionError("Snapshot mesh, step, time or field list changed")
            if any(not math.isfinite(value) for field in snapshot[1].values() for value in field):
                raise AssertionError(f"{out}: nonfinite field")
            error = max(
                abs(a - b) for name in snapshot[1] for a, b in zip(reference[1][name], snapshot[1][name])
            )
            if not error <= 1e-10:
                raise AssertionError(f"{out}: maximum field difference {error}")
            rows.append(
                dict(
                    mode=mode,
                    repeat=repeat + 1,
                    steps=int(header[3]),
                    **phases,
                    job_s=elapsed,
                    field_error=error,
                )
            )
            print(
                f"{out.name}: {header[3]} steps, integration {phases['integration']:.4f} s, job {elapsed:.2f} s",
                flush=True,
            )
    with (work / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    median = {
        m: statistics.median(r["integration"] for r in rows if r["mode"] == m) for m, _, _ in modes
    }
    print(
        f"Equal/auto {median['equal']/median['auto']:.3f}x, single/auto {median['single']/median['auto']:.3f}x"
    )


if __name__ == "__main__":
    parser = mpi_parser(__doc__, require_programs=True)
    parser.add_argument(
        "--work", type=Path, required=True, help="new directory for inputs, results and timings"
    )
    parser.add_argument(
        "--devices", default="0,1", help="two GPU IDs, faster card first for the single-GPU baseline"
    )
    args = parser.parse_args()
    cards = args.devices.split(",")
    if len(cards) != 2 or len(set(cards)) != 2:
        parser.error("--devices requires two distinct GPU IDs")
    benchmark(args.work.resolve(), args.aero.resolve(), args.meshdec.resolve(), args.mpiexec, cards)
