"""Проверяет параллельную запись общих полей и их чтение по исходным номерам узлов."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
from pathlib import Path
import shutil
import struct

from mpi_euler import check_output, close, launch, stamp
from mpi_geometry import legacy_archive, run, sparse_line
from mpi_periodic import report_rows
from run import nodes, settings


def binary_snapshot(path):
    """Разбирает двоичное описание снимка и значения его полей для независимой сверки."""
    with path.open("rb") as f:

        def word():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(word()).decode("utf-8")

        if f.read(8) != b"AEROFLDB" or word() != 1:
            raise AssertionError(f"Bad field signature in {path}")
        dimension, mesh_hash = word(), f.read(32)
        counts = tuple(word() for _ in range(4))
        step, time = word(), struct.unpack("<d", f.read(8))[0]
        fields = {}
        for _ in range(word()):
            name = string()
            if (
                name in fields
                or f.read(1) != b"\x00"
                or word() != 1
                or string() != "value"
                or string() != ""
                or word() != counts[0]
            ):
                raise AssertionError(f"Bad NODE descriptor in {path}")
            fields[name] = struct.unpack(f"<{counts[0]}d", f.read(8 * counts[0]))
        if f.read(1):
            raise AssertionError(f"Trailing data in {path}")
        return (dimension, mesh_hash, counts, step, time), fields


def compare(reference, result, coords):
    """Сверяет ожидаемые значения с полученными и сообщает место расхождения."""
    a, expected = binary_snapshot(reference)
    b, actual = binary_snapshot(result)
    if (
        a[:4] != b[:4]
        or not close(a[4], b[4])
        or list(actual) != list(expected)
        or b[2][0] != len(coords)
    ):
        raise AssertionError(f"Wrong global metadata or variable order in {result}")
    copies = {}
    for i, point in enumerate(coords):
        key = tuple(round(x % 1, 12) for x in point)
        peer = copies.setdefault(key, i)
        for name, values in actual.items():
            if not close(values[i], expected[name][i]) or values[i] != values[peer]:
                raise AssertionError(f"Field {name} differs at original node {i} in {result}")
    if len(copies) == len(coords):
        raise AssertionError("Periodic fixture must contain identified vertices")


def verify(case, aero, meshdec, launcher):
    """Проверяет параллельную запись общих полей и их чтение по исходным номерам узлов."""
    shutil.copytree(ROOT / "qa/cases/shear_jump_2d", case)
    names = ["rho", "ux", "uy", "p", "E"]
    coords = nodes(case / "grid.mesh", 2)
    options = settings(case / "main.txt")
    options.update(time_scheme="ssprk3", reconstruction="weno5", omp_threads=1)

    def configure(mesh_name, end, recovery=0):
        options.update(mesh=mesh_name, final_time=end, recovery_every_steps=recovery)
        (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))

    init = case / "init.txt"
    init.write_text(
        "initial_type formula\nrho = 1+.08*sin(2*pi*x)+.03*cos(2*pi*y)\n"
        "ux = .2+.05*cos(2*pi*y)\nuy = .05\np = 1+.06*cos(2*pi*x)\n"
    )
    zones = case / "output.txt"
    global_zones = (
        "[latest]\nfile = shared/latest.fields\nevery_time = .001\nlayout = global\n"
        "variables = rho, ux, uy, p, E\n"
        "[history]\nfile = shared/hist.fields\nevery_steps = 1\nmulti = 1\n"
        "format = binary\nlayout = global\nvariables = E, rho, p\n"
    )
    local_zone = (
        "[local]\nfile = state.fields\nevery_steps = 1\nmulti = 1\n"
        "layout = local\nformat = text\nvariables = rho, ux, uy, p, E\n"
    )
    zones.write_text(global_zones)
    run([meshdec, case / "grid.mesh", case / "grid.mdec", "--compress", "--target-ranks=8"])
    run([meshdec, case / "grid.mesh", case / "one.mdec"])
    ref, mixed, empty = (case / name for name in ("native", "mixed", "empty"))
    configure("grid.mesh", 0.003)
    launch(aero, case, ref, launcher, 1)

    def compare_series(target):
        expected = sorted(f.name for f in (ref / "shared").iterdir())
        if sorted(f.name for f in (target / "shared").iterdir()) != expected:
            raise AssertionError("Global output schedule or set of published files differs")
        for name in expected:
            compare(ref / "shared" / name, target / "shared" / name, coords)
        metadata, _ = binary_snapshot(target / "shared/latest.fields")
        if metadata[-1] != 0.003:
            raise AssertionError("Latest snapshot was not replaced at its last deadline")

    configure("one.mdec", 0.003)
    launch(aero, case, empty, launcher, 4)
    compare_series(empty)
    if list(empty.glob("rank-*")):
        raise AssertionError("Global-only output wrote partition files")

    configure("grid.mdec", 0.003, recovery=1)
    zones.write_text(global_zones + local_zone)
    launch(aero, case, mixed, launcher, 4)
    compare_series(mixed)
    original = mixed / "shared/latest.fields"
    before = original.read_bytes()
    saved_step = binary_snapshot(original)[0][3]
    for rank in range(4):
        part = mixed / f"rank-{rank}-of-4"
        if not (part / "grid.mesh").is_file() or not list((part / "fields").glob("state_*.fields")):
            raise AssertionError("Missing local output in a run with mixed layouts")

    # После восстановления новые общие файлы должны оказаться в каталоге продолжения.
    # Сравниваем их с непрерывным решением в тот же момент физического времени.
    configure("grid.mesh", 0.004)
    extended = case / "extended"
    zones.write_text(global_zones)
    launch(aero, case, extended, launcher, 1)
    configure("grid.mdec", 0.004, recovery=1)
    launch(aero, case, mixed, launcher, 2, "--recover")
    resumed = mixed / f"resumed-{saved_step}-1" / "shared/latest.fields"
    compare(extended / "shared/latest.fields", resumed, coords)
    if original.read_bytes() != before:
        raise AssertionError("Resume overwrote the earlier segment")

    collision = case / "collision"
    (collision / "shared").mkdir(parents=True)
    first = collision / "shared/hist_000001.fields"
    first.write_bytes(b"old history")
    configure("grid.mdec", 0.003)
    zones.write_text(global_zones[global_zones.index("[history]") :])
    log = run([launcher, "-n", 2, aero, f"--input={case}", f"--output={collision}"], success=False)
    if (
        "publish" not in log
        or first.read_bytes() != b"old history"
        or list(first.parent.iterdir()) != [first]
    ):
        raise AssertionError(log)

    legacy_archive(case / "grid.mdec", case / "old.mdec", version=3)
    configure("old.mdec", 0.003)
    log = run([launcher, "-n", 2, aero, f"--input={case}", f"--output={case/'legacy'}"], success=False)
    if "SHA-256" not in log:
        raise AssertionError(log)

    # Каждый вариант солвера начинает новый расчёт из своего сохранённого снимка.
    zones.write_text(local_zone)

    def start_from(source):
        init.write_text(
            f"initial_type = file\npath = {source}/shared/latest.fields\n"
            "rho = rho\nux = ux\nuy = uy\np = p\n"
        )

    configure("grid.mesh", 0.0003)
    start_from("native")
    reference = case / "reference-new"
    launch(aero, case, reference, launcher, 1)
    configure("grid.mdec", 0.0003)
    start_from("mixed")
    actual = case / "new"
    launch(aero, case, actual, launcher, 2)
    check_output(reference, actual, 2, len(coords), 2, names)
    step, time = stamp(actual / "rank-0-of-2/fields/state_000001.fields")
    if step != 1 or not 0 < time <= 0.0003:
        raise AssertionError("New calculation did not start at time zero")
    print("OK global AEROFIELDS matches native output at every periodic node, including empty ranks")
    print("OK mixed layouts, resumed schedules, atomic publication, and fresh run from 4 ranks on 2")


def verify_batches(case, aero, meshdec, launcher):
    """Проверяет совместную запись, когда у процессов разные числа пакетов и несмежные номера узлов."""
    shutil.copytree(ROOT / "qa/cases/sod_1d", case)
    mesh = case / "grid.mesh"
    sparse_line(mesh, 16384)
    coords = nodes(mesh, 1)
    (case / "bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n")
    (case / "init.txt").write_text("initial_type formula\nrho = 1+.04*cos(2*pi*x)\nux = .21\np = 1\n")
    (case / "output.txt").write_text(
        "[all]\nfile = all.fields\nlayout = global\nevery_steps = 1\n" "variables = E, rho, ux, p\n"
    )
    options = settings(case / "main.txt")
    options.update(reconstruction="first_order", time_scheme="euler", omp_threads=1, final_time=1e-7)
    run([meshdec, mesh, case / "grid.mdec", "--target-ranks=16"])
    for source, ranks in (("mesh", 1), ("mdec", 2)):
        options["mesh"] = f"grid.{source}"
        if ranks == 2:
            options["mpi_rank_weights"] = "3,1"
        (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))
        log = launch(aero, case, case / source, launcher, ranks)
        if ranks == 2:
            # Одна периодическая пара склеивает две исходные вершины. В отчёте считаются степени свободы,
            # поэтому в данном неравенстве учитываем ещё один исходный номер.
            counts = [int(row["owned"]) for row in report_rows(log, "rank\tnode")]
            if not (min(counts) + 1 < 8192 < max(counts)):
                raise AssertionError(f"Unequal MPI-IO batch counts required: {counts}")
    compare(case / "mesh/all.fields", case / "mdec/all.fields", coords)
    print("OK multiple batches, noncontiguous source IDs and different per-rank batch counts")


def main():
    parser = mpi_parser(__doc__)
    args = parser.parse_args()
    with mpi_workspace("mpi-fields-") as work:
        programs = (args.aero.resolve(), args.meshdec.resolve(), args.mpiexec)
        verify(Path(work) / "case", *programs)
        verify_batches(Path(work) / "sparse", *programs)


if __name__ == "__main__":
    main()
