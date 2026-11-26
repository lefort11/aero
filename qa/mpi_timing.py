"""Проверяет счётчики стадий, обменов и записи по таблице времени MPI-расчёта."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import math
from pathlib import Path
import re
import shutil

from mpi_euler import launch
from mpi_geometry import run
from run import settings, write_settings


def check(log, ranks, start, stages, backend):
    """Проверяет состав фаз, числа вызовов и диапазоны накопленного времени."""
    end = int(re.search(r"Completed (\d+) steps", log)[1])
    steps = end - start
    header = f"{backend} timings: ranks={ranks} start_step={start} end_step={end} steps={steps}"
    assert log.count(" timings:") == 1 and header in log, log
    lines = log.split(header)[1].strip().splitlines()
    assert lines[0].split() == ["phase", "calls_per_rank", "min_s", "mean_s", "max_s", "max_s_per_step"]
    exchange_calls = stages * steps if ranks > 1 else 0
    # При завершении все зоны и recovery пишутся один раз, в том числе
    # если после восстановления не понадобился ни один новый шаг.
    field_writes = end // 2 - start // 2 + int(not steps or end % 2 != 0)
    recovery_writes = end // 3 - start // 3 + int(not steps or end % 3 != 0)
    readbacks = (
        int(not steps) + sum(i % 2 == 0 or i % 3 == 0 or i == end for i in range(start + 1, end + 1))
        if backend == "GPU"
        else 0
    )
    expected = dict(
        flux_compute=stages * steps,
        stage_update=stages * steps,
        halo_exchange=exchange_calls,
        flux_exchange=exchange_calls,
        cfl=steps,
        output_readback=readbacks,
        fields_write=field_writes,
        recovery_write=recovery_writes,
        sa_transport=0,
        les_closure=0,
        integration=1,
    )
    maxima = {}
    for line in lines[1:]:
        if line.startswith("Completed"):
            break
        phase, calls, low, mean, high, per_step = line.split()
        assert int(calls) == expected.pop(phase), line
        low, mean, high = map(float, (low, mean, high))
        assert 0 <= low <= mean <= high and math.isfinite(high), line
        assert (high > 0) == (int(calls) > 0), line
        if steps:
            assert math.isclose(float(per_step), high / steps, rel_tol=2e-9), line
        else:
            assert per_step == "n/a", line
        maxima[phase] = high
    assert not expected and maxima["integration"] >= max(maxima.values()), log
    if steps:
        assert (
            maxima["integration"] > 0 and maxima["flux_compute"] > 0 and maxima["recovery_write"] > 0
        ), log
    if field_writes:
        assert maxima["fields_write"] > 0, log
    return end


def verify(case, aero, meshdec, launcher, devices):
    """Проверяет счётчики стадий, обменов и записи по таблице времени MPI-расчёта."""
    shutil.copytree(ROOT / "qa/cases/sod_1d", case)
    run([meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=4"])
    config = settings(case / "main.txt")
    config.update(
        mesh="grid.mdec",
        reconstruction="first_order",
        final_time=0.004,
        omp_threads=1,
        recovery_every_steps=3,
    )
    backend = "GPU" if devices else "CPU"
    if devices:
        config.update(backend="cuda", mpi_rank_weights="1")
    (case / "output.txt").write_text(
        "[state]\nfile = state.fields\nevery_steps = 2\nmulti = 0\n"
        "layout = global\nformat = binary\nvariables = rho, ux, p\n"
    )

    # Создаёт файлы и параметры тестовой постановки для выбранного сценария.
    def configure(**changes):
        config.update(changes)
        write_settings(case / "main.txt", config)

    configure(time_scheme="euler")
    if devices:
        (case / "cuda.txt").write_text(f"devices = {devices}\n")
    assert " timings:" not in launch(aero, case, case / "disabled", launcher, 2)
    for ranks, scheme, stages in ((1, "euler", 1), (2, "euler", 1), (2 if devices else 4, "ssprk3", 3)):
        configure(time_scheme=scheme, mpi_timing=1)
        if devices:
            (case / "cuda.txt").write_text("devices = " + ",".join(devices.split(",")[:ranks]) + "\n")
        directory = case / f"run-{ranks}-{scheme}"
        last = check(launch(aero, case, directory, launcher, ranks), ranks, 0, stages, backend)
        assert last >= 2
        if ranks == 2 and scheme == "euler":
            assert (directory / "state.fields").read_bytes() == (
                case / "disabled/state.fields"
            ).read_bytes()
    configure(final_time=0.008)
    resumed = check(launch(aero, case, directory, launcher, 2, "--recover"), 2, last, 3, backend)
    assert resumed > last
    assert (
        check(launch(aero, case, directory, launcher, 2, "--recover"), 2, resumed, 3, backend) == resumed
    )
    print(f"OK {backend} timings, Euler/SSPRK3, recovery and zero additional steps")


# Разбирает параметры запуска, выполняет набор QA-проверок и возвращает статус результата.
def main():
    parser = mpi_parser(__doc__, require_programs=True)
    parser.add_argument("--devices", help="run CUDA timings on these two GPUs")
    args = parser.parse_args()
    with mpi_workspace("mpi-timing-") as directory:
        verify(
            Path(directory) / "case",
            args.aero.resolve(),
            args.meshdec.resolve(),
            args.mpiexec,
            args.devices,
        )


if __name__ == "__main__":
    main()
