"""Сравнивает результаты распределённого расчёта на CUDA, включая выбор GPU и переразбиение по памяти."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import math
from pathlib import Path
import re
import shutil
import struct

from mpi_euler import check_output, launch, stamp
from mpi_geometry import run
from mpi_periodic import report_rows
from run import nodes, settings


def memory_rows(log):
    """Извлекает из журнала сведения о доступной и зарезервированной памяти GPU."""
    pattern = (
        r"MPI GPU fit: rank (\d+) leaves \[(\d+),(\d+)\) required_bytes (\d+) "
        r"budget_bytes (\d+) (reserved|rejected)"
    )
    rows = [
        (int(r), int(first), int(end), int(need), int(budget), status)
        for r, first, end, need, budget, status in re.findall(pattern, log)
    ]
    if len(rows) < 2 or any(row[-1] != "reserved" or not 0 < row[3] <= row[4] for row in rows[-2:]):
        raise AssertionError(log)
    return rows


def check_gpu_weights(log, archive, devices):
    """Сверяет границы листьев с весами GPU и стоимостью из каталога архива."""
    scores = re.findall(r"MPI GPU calibration: rank (\d+) faces_per_s (\S+) weight (\S+)", log)
    if [int(row[0]) for row in scores] != [0, 1]:
        raise AssertionError(scores)
    rates = [float(row[1]) for row in scores]
    weights = [float(row[2]) for row in scores]
    plan = report_rows(log, "mpi_rank\tweight")
    if (
        any(not math.isfinite(rate) or rate <= 0 for rate in rates)
        or any(not math.isclose(w, r / max(rates), rel_tol=2e-5) for w, r in zip(weights, rates))
        or [float(row["weight"]) for row in plan] != weights
    ):
        raise AssertionError((scores, plan))
    # Сверяем разрез с независимой суммой работы по записям каталога листьев.
    data = archive.read_bytes()
    (leaves,) = struct.unpack_from("<Q", data, 72)
    (directory,) = struct.unpack_from("<Q", data, 104)
    work = [0]
    for leaf in range(leaves):
        raw, nodes = struct.unpack_from("<QQ", data, directory + 48 * leaf + 16)
        work.append(work[-1] + raw + 32 * nodes)
    cut = int(plan[0]["end_leaf"])
    if leaves == 1:
        if rates[0] != rates[1] and cut != int(rates[0] > rates[1]):
            raise AssertionError((scores, plan))
    else:
        target = work[-1] * weights[0] / sum(weights)
        nearest = min(abs(value - target) for value in work[1:-1])
        if not 0 < cut < leaves or abs(work[cut] - target) > nearest + work[-1] * 2e-5:
            raise AssertionError((scores, plan, target))
    print(f"OK GPU {devices} measured weights {weights}, leaves {cut}+{leaves-cut}")


def verify(case, aero, meshdec, launcher, devices, fixture, scheme, recon, empty=False):
    """Сравнивает результаты распределённого расчёта на CUDA, включая выбор GPU и переразбиение по памяти."""
    shutil.copytree(ROOT / "qa/cases" / fixture, case)
    dim = 2 if fixture == "shear_jump_2d" else 1
    names = ["rho", "ux"] + (["uy"] if dim == 2 else []) + ["p", "E"]
    count = len(nodes(case / "grid.mesh", dim))
    run([meshdec, case / "grid.mesh", case / "grid.mdec", *([] if empty else ["--target-ranks=4"])])
    (case / "cuda.txt").write_text(f"devices = {devices}\n")
    options = settings(case / "main.txt")
    options.update(
        reconstruction=recon,
        time_scheme=scheme,
        final_time=0.004,
        cfl=0.025,
        report_every=2,
        omp_threads=1,
    )
    if dim == 2:
        (case / "init.txt").write_text(
            "initial_type formula\nrho = 1+.05*cos(2*pi*x)*cos(2*pi*y)\n"
            "ux = .25\nuy = -.12\np = 1+.01*cos(2*pi*x)\n"
        )
    columns = f"multi = 1\nformat = text\nvariables = {', '.join(names)}\n"
    (case / "output.txt").write_text(
        "[state]\nfile = state.fields\nevery_steps = 1\n"
        + columns
        + "\n[clock]\nfile = clock.fields\nevery_time = .0017\n"
        + columns
    )

    def configure(**changes):
        options.update(changes)
        (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))

    configure(mesh="grid.mesh", backend="cpu")
    reference = case / "native"
    launch(aero, case, reference, launcher, 1)
    deadlines = sorted(stamp(p)[1] for p in reference.glob("clock_*.fields"))
    if deadlines != [0.0017, 0.0034]:
        raise AssertionError(f"Unexpected output times: {deadlines}")
    configure(mesh="grid.mdec", backend="cuda", recovery_every_steps=3)
    if recon == "weno5":
        configure(mpi_rank_weights="auto")
    out = case / "distributed"
    log = launch(aero, case, out, launcher, 2)
    check_gpu_weights(log, case / "grid.mdec", devices)
    placement = re.findall(
        r"MPI CUDA placement: rank (\d+) local_rank (\d+) device (\d+) node/PCI (\S+)", log
    )
    if (
        [int(row[2]) for row in placement] != list(map(int, devices.split(",")))
        or len({row[3] for row in placement}) != 2
        or "Active backend = CUDA (MPI)" not in log
    ):
        raise AssertionError(log)
    original = memory_rows(log)[-2:]
    budgets = re.findall(r"MPI GPU budget: rank (\d+) free_bytes (\d+) budget_bytes (\d+)", log)
    if len(budgets) != 2 or any(int(b) != int(free) - int(free) // 20 for _, free, b in budgets):
        raise AssertionError(log)
    error = check_output(reference, out, 2, count, dim, names, empty)
    print(f"OK two GPUs {scheme} {recon} empty={empty}, max field difference {error:.3g}")
    if not empty and scheme == "ssprk3":
        # Поровну делим исходный объём работы, чтобы проверить лимит памяти на любой паре карт.
        configure(mpi_rank_weights="1,1")
        manual = launch(aero, case, case / "manual", launcher, 2)
        if "MPI GPU calibration" in manual or [
            float(r["weight"]) for r in report_rows(manual, "mpi_rank\tweight")
        ] != [1, 1]:
            raise AssertionError(manual)
        original = memory_rows(manual)[-2:]
        # Уменьшаем предел памяти процесса 0 ниже первоначальной потребности шаблона WENO.
        limit = int(0.80 * original[0][3])
        configure(mpi_gpu_memory_mib=f"{limit/1048576},4096")
        tight = case / "limited"
        rows = memory_rows(launch(aero, case, tight, launcher, 2))
        _, first, end, need, budget, _ = rows[-2]
        if not (need <= budget == limit and 0 < end - first < original[0][2] - original[0][1]):
            raise AssertionError(rows)
        if not any(row[-1] == "rejected" for row in rows):
            raise AssertionError(f"Expected exact WENO reservation to refine the estimate: {rows}")
        error = check_output(reference, tight, 2, count, dim, names)
        print(
            f"OK GPU VRAM cap {limit} bytes, leaves {original[0][2]} -> {end}, "
            f"reserved {need} bytes, max field difference {error:.3g}"
        )
        configure(mpi_rank_weights="auto")
        capped = case / "auto-limited"
        capped_log = launch(aero, case, capped, launcher, 2)
        samples = [
            int(n) for n in re.findall(r"MPI GPU calibration: rank .*? sample_faces (\d+)", capped_log)
        ]
        if len(samples) != 2 or samples[0] != samples[1] or not 0 < samples[0] < 32768:
            raise AssertionError(capped_log)
        if memory_rows(capped_log)[-2][4] != limit:
            raise AssertionError(capped_log)
        error = check_output(reference, capped, 2, count, dim, names)
        print(
            f"OK auto weights within VRAM cap, sample {samples[0]} faces, max field difference {error:.3g}"
        )
        del options["mpi_gpu_memory_mib"]
        del options["mpi_rank_weights"]
        latest = max(stamp(p)[0] for p in reference.glob("*.fields"))
        configure(mesh="grid.mesh", backend="cpu", final_time=0.006, recovery_every_steps=0)
        # Оба исполнителя продолжают с одного сохранённого состояния и одного момента времени.
        baseline = case / "cpu-restart"
        baseline.mkdir()
        for checkpoint in out.glob("recovery-*.bin"):
            shutil.copyfile(checkpoint, baseline / checkpoint.name)
        serial_gpu = case / "gpu-restart"
        shutil.copytree(baseline, serial_gpu)
        launch(aero, case, baseline, launcher, 1, "--recover")
        configure(mesh="grid.mdec", backend="cuda")
        launch(aero, case, out, launcher, 2, "--recover")
        sub = f"resumed-{latest}-1"
        error = check_output(baseline / sub, out / sub, 2, count, dim, names)
        print(f"OK two GPUs restart, max field difference {error:.3g}")
        device = devices.split(",")[0]
        (case / "cuda.txt").write_text(f"devices = {device}\n")
        configure(mesh="grid.mesh", recovery_every_steps=2)
        serial_log = launch(aero, case, serial_gpu, launcher, 1, "--recover")
        if (
            f"Active backend = CUDA device {device}" not in serial_log
            or "Recovered step" not in serial_log
        ):
            raise AssertionError(serial_log)
        error = check_output(baseline / sub, serial_gpu / sub, 1, count, dim, names)
        print(f"OK two GPUs to one GPU restart, max field difference {error:.3g}")
    if empty:
        busy = next(row for row in original if row[2] > row[1])
        # Одна неделимая часть сетки помещается и без кэша узлового газа и нормалей.
        # Сначала отнимаем один байт от полной конфигурации и проверяем расчёт на экономном пути.
        configure(mpi_gpu_memory_mib=(busy[3] - 1) / 1048576)
        compact = case / "without-cache"
        reduced = memory_rows(launch(aero, case, compact, launcher, 2))[-2:]
        required = next(row[3] for row in reduced if row[2] > row[1])
        if required >= busy[3]:
            raise AssertionError((busy, reduced))
        error = check_output(reference, compact, 2, count, dim, names, empty)
        print(f"OK one leaf without optional buffers, reserved {required} bytes, max field difference {error:.3g}")
        # Теперь предел ниже измеренного минимума даже после отказа от кэша.
        configure(mpi_gpu_memory_mib=(required - 1) / 1048576)
        out = case / "too-small"
        failure = run([launcher, "-n", 2, aero, f"--input={case}", f"--output={out}"], success=False)
        if ("MPI GPU memory" not in failure or sorted(p.name for p in out.iterdir()) != ["aero.log"]
                or "MPI GPU memory" not in (out / "aero.log").read_text() or "Starting time integration" in failure):
            raise AssertionError(failure)
        print("OK indivisible leaf exceeds both device budgets before output")
        (case / "cuda.txt").write_text("devices = 0\n")
        failure = run(
            [launcher, "-n", 2, aero, f"--input={case}", f"--output={case/'bad'}"], success=False
        )
        if "one GPU per node-local MPI rank" not in failure:
            raise AssertionError(failure)


def main():
    parser = mpi_parser(__doc__, require_programs=True)
    parser.add_argument("--devices", default="1,0", help="two visible device IDs on the same node")
    args = parser.parse_args()
    with mpi_workspace("mpi-gpu-") as directory:
        for fixture, scheme, recon, empty in (
            ("sod_1d", "euler", "first_order", False),
            ("shear_jump_2d", "ssprk3", "weno5", False),
            ("sod_1d", "ssprk3", "ebr3", True),
        ):
            verify(
                Path(directory) / recon,
                args.aero.resolve(),
                args.meshdec.resolve(),
                args.mpiexec,
                args.devices,
                fixture,
                scheme,
                recon,
                empty,
            )


if __name__ == "__main__":
    main()
