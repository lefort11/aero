"""Проверяет продолжение вычислений из контрольной точки при смене числа процессов и повреждении файлов."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import re
from pathlib import Path
import shutil
import struct
import zlib

from mpi_euler import free_patch, check_output, close, launch, stamp
from mpi_geometry import legacy_archive, run, sparse_line
from mpi_periodic import report_rows
from run import nodes, read_snapshot, settings


def conservative_file(source, target, count, names):
    """Записывает консервативное начальное поле для проверки чтения и восстановления."""
    fields = read_snapshot(source, count, 2, stamp(source)[1], names)
    header = source.read_text().split("\nFIELDS ")[0]
    rho = fields["rho"][1]
    columns = [
        fields["E"][1],
        [r * u for r, u in zip(rho, fields["uy"][1])],
        rho,
        [r * u for r, u in zip(rho, fields["ux"][1])],
    ]
    cells = int(re.search(r"(?m)^COUNTS\s+\d+\s+\d+\s+\d+\s+(\d+)", header)[1])
    target.write_text(
        header + '\nFIELDS 2\nFIELD "unused"\nLOCATION CELL\nCOMPONENTS 1 "value"\n'
        f'UNITS ""\nVALUES {cells}\nDATA ' + "0 " * cells + "\nEND_FIELD\n"
        'FIELD "U"\nLOCATION NODE\nCOMPONENTS 4 "E" "rho_uy" "rho" "rho_ux"\n'
        f'UNITS ""\nVALUES {count}\nDATA '
        + " ".join(repr(value) for col in columns for value in col)
        + "\nEND_FIELD\nEND\n"
    )


def verify(case, aero, meshdec, launcher):
    """Проверяет продолжение вычислений из контрольной точки при смене числа процессов и повреждении файлов."""
    shutil.copytree(ROOT / "qa/cases/shear_jump_2d", case)
    mesh = case / "grid.mesh"
    count = len(nodes(mesh, 2))
    names = ["rho", "ux", "uy", "p", "E"]
    options = settings(case / "main.txt")
    options.update(reconstruction="weno5", time_scheme="ssprk3", omp_threads=1)

    def configure(mesh_name, end, interval):
        options.update(mesh=mesh_name, final_time=end, recovery_every_steps=interval)
        (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))

    def zone(name, timing, fmt="text", multi=1):
        return (
            f"[{name}]\nfile = {name}.fields\n{timing}\nformat = {fmt}\nmulti = {multi}\n"
            f"variables = {', '.join(names)}\n\n"
        )

    # Меняющиеся давление и скорость у границы с опорными значениями помогают обнаружить ошибку:
    # при восстановлении исходное поле для условий нельзя подменить принятым состоянием.
    (case / "bc.txt").write_text(
        "[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n\n"
        + "".join(free_patch("y" + side) for side in ("min", "max"))
    )
    init = case / "init.txt"
    init.write_text(
        "initial_type formula\nrho = 1+.07*sin(2*pi*x)+.05*y\nux = .25\n"
        "uy = .12+.04*y\np = 1+.05*cos(2*pi*x)+.12*y\n"
    )
    formula = init.read_text()
    configure("grid.mesh", 0.0005, 0)
    output_settings = case / "output.txt"
    formats = ("text", "binary", "lzma")
    output_settings.write_text("".join(zone(fmt, "every_time = .0005", fmt, 0) for fmt in formats))
    seed = case / "seed"
    launch(aero, case, seed, launcher, 1)
    run([meshdec, mesh, case / "grid.mdec", "--compress", "--target-ranks=4"])

    def initial(fmt):
        init.write_text(f"initial_type file\npath seed/{fmt}.fields\n" "rho rho\nux ux\nuy uy\np p\n")

    output_settings.write_text(zone("state", "every_steps = 1") + zone("timed", "every_time = .001"))
    initial("text")
    configure("grid.mesh", 0.006, 0)
    ref = case / "reference"
    launch(aero, case, ref, launcher, 1)

    def reference_slice(directory, start, end):
        directory.mkdir()
        for f in ref.glob("*.fields"):
            if start < stamp(f)[0] <= end:
                shutil.copyfile(f, directory / f.name)
        return directory

    split = 0.003
    early = [stamp(f)[0] for f in ref.glob("state_*.fields") if stamp(f)[1] <= split]
    last = max(early)
    prefix = reference_slice(case / "prefix", 0, last)
    configure("grid.mdec", split, 1)
    out = case / "restart"
    for fmt in formats:
        initial(fmt)
        dest = out if fmt == "binary" else case / f"input-{fmt}"
        launch(aero, case, dest, launcher, 4)
        check_output(prefix, dest, 4, count, 2, names)
    conservative_file(seed / "text.fields", seed / "conservative.fields", count, names)
    init.write_text("initial_type = file\npath = seed/conservative.fields\nfield = U\n")
    launch(aero, case, case / "input-U", launcher, 2)
    check_output(prefix, case / "input-U", 2, count, 2, names)
    print("OK MPI reads primitive and conservative global fields from time zero (text, binary and LZMA)")

    # Контрольная точка обычной сетки должна читаться при другом разбиении архива на листья,
    # в том числе если часть процессов не владеет ни одним листом.
    initial("binary")
    configure("grid.mesh", split, 3)
    native_restart = case / "native-restart"
    launch(aero, case, native_restart, launcher, 1)
    run([meshdec, mesh, case / "one-leaf.mdec"])
    legacy_archive(case / "grid.mdec", case / "old.mdec", version=3)
    # Контрольная точка старого архива привязана к байтам именно этого файла.
    configure("old.mdec", split, 1)
    legacy = case / "legacy-restart"
    init.write_text(formula)
    launch(aero, case, legacy, launcher, 2)
    initial("text")
    configure("grid.mdec", split, 1)
    bad = seed / "text.fields"
    original = bad.read_text()
    bad.write_text(re.sub(r'(?<=MESH_SHA256\s)"[a-f0-9]{64}"', '"' + "0" * 64 + '"', original))
    failure = run(
        [launcher, "-n", 2, aero, f"--input={case}", f"--output={case/'wrong-mesh'}"], success=False
    )
    if "SHA-256 mismatch" not in failure:
        raise AssertionError(failure)
    init.unlink()
    shutil.rmtree(seed)
    slots = sorted(out.glob("recovery-*.bin"))
    if len(slots) != 2:
        raise AssertionError("Expected two shared recovery files")

    def saved_step(file):
        with file.open("rb") as f:
            f.seek(32)
            return struct.unpack("<Q", f.read(8))[0]

    slots.sort(key=saved_step)
    if [saved_step(f) for f in slots] != [last - 1, last]:
        raise AssertionError("Recovery slots did not alternate")

    def clone(target):
        target.mkdir()
        for f in slots:
            shutil.copyfile(f, target / f.name)
        return target

    # Проверяем явный выбор каждого слота даже при наличии более нового исправного шага.
    # Если сохранение единственное и приходится на final_time, для него используется противоположный слот.
    final_step = max(stamp(f)[0] for f in ref.glob("state_*.fields"))
    for slot in (0, 1):
        directory = clone(case / f"selected-{slot}")
        selected = directory / f"recovery-{slot}.bin"
        expected = saved_step(selected)
        before = selected.read_bytes()
        ranks = slot + 1
        configure("grid.mesh" if ranks == 1 else "grid.mdec", 0.006, final_step + 1)
        flags = ["--recover", "0"] if slot == 0 else ["--recover=1"]
        log = launch(aero, case, directory, launcher, ranks, *flags)
        if f"Recovered step {expected} at" not in log:
            raise AssertionError(log)
        reference = reference_slice(case / f"expected-selected-{slot}", expected, final_step)
        check_output(reference, directory / f"resumed-{expected}-1", ranks, count, 2, names)
        if (
            selected.read_bytes() != before
            or saved_step(directory / f"recovery-{1-slot}.bin") != final_step
        ):
            raise AssertionError("First save after explicit recovery did not rotate to the other file")

    torn = clone(case / "unfinished")
    corrupt = clone(case / "corrupt")
    blocked = clone(case / "blocked")
    older = slots[0].name
    newer = slots[1].name
    committed = slots[1].read_bytes()
    (torn / newer).write_bytes(bytes(120))
    with (corrupt / newer).open("r+b") as f:
        f.seek(-9, 2)
        value = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([value[0] ^ 128]))

    missing = clone(case / "missing")
    (missing / newer).unlink()
    configure("grid.mdec", 0.006, 1)
    slot = "0" if newer == "recovery-0.bin" else "1"
    for directory in (missing, torn, corrupt):
        log = run(
            [launcher, "-n", 2, aero, f"--input={case}", f"--output={directory}", "--recover", slot],
            success=False,
        )
        if "no valid checkpoint" not in log or newer not in log or list(directory.glob("resumed-*")):
            raise AssertionError(log)
    print("OK explicit recovery selection, rotation and rejection of missing or damaged selected files")

    for directory, ranks, expected in [(out, 2, last), (torn, 1, last - 1), (corrupt, 2, last - 1)]:
        # Останавливаемся в следующий момент вывода, чтобы проверить также переход с двух процессов на четыре.
        end = 0.005 if directory == out else 0.006
        configure("grid.mesh" if ranks == 1 else "grid.mdec", end, 1)
        log = launch(aero, case, directory, launcher, ranks, "--recover")
        if f"Recovered step {expected} at" not in log:
            raise AssertionError(log)
        end_step = max(stamp(f)[0] for f in ref.glob("state_*.fields") if stamp(f)[1] <= end)
        reference = reference_slice(case / f"expected-{directory.name}", expected, end_step)
        check_output(reference, directory / f"resumed-{expected}-1", ranks, count, 2, names)
        if directory == out:
            configure("grid.mdec", 0.006, 1)
            launch(aero, case, out, launcher, 4, "--recover")
            tail = reference_slice(case / "expected-tail", end_step, 2**64 - 1)
            check_output(tail, out / f"resumed-{end_step}-1", 4, count, 2, names)
    configure("one-leaf.mdec", 0.006, 1)
    launch(aero, case, native_restart, launcher, 4, "--recover")
    tail = reference_slice(case / "native-tail", last, 2**64 - 1)
    check_output(tail, native_restart / f"resumed-{last}-1", 4, count, 2, names, empty=True)
    print("OK WENO5/SSPRK3 restarts 4 -> 2 -> 4, 4 -> 1 and 1 -> 4 with empty partitions")

    # Создаём реальную ошибку открытия для записи в неактивном слоте.
    # После завершения процессов ранее зафиксированный слот должен читаться и совпадать побайтово.
    (blocked / older).unlink()
    (blocked / older).mkdir()
    configure("grid.mdec", 0.006, 1)
    log = run(
        [launcher, "-n", 2, aero, f"--input={case}", f"--output={blocked}", "--recover"], success=False
    )
    if "cannot prepare" not in log or (blocked / newer).read_bytes() != committed:
        raise AssertionError(log)
    (blocked / older).rmdir()
    launch(aero, case, blocked, launcher, 2, "--recover")
    check_output(tail, blocked / f"resumed-{last}-2", 2, count, 2, names)
    # Пустой или незавершённый слот и повреждённые данные должны вызвать согласованный отказ восстановления.
    # В этом случае запуск с начальными условиями привёл бы к скрытой потере результатов.
    (torn / newer).write_bytes(bytes(120))
    (torn / older).write_bytes(b"partial")
    configure("grid.mdec", 0.006, 1)
    log = run(
        [launcher, "-n", 4, aero, f"--input={case}", f"--output={torn}", "--recover"], success=False
    )
    if "no valid checkpoint" not in log:
        raise AssertionError(log)
    configure("old.mdec", 0.006, 1)
    legacy_log = launch(aero, case, legacy, launcher, 4, "--recover")
    if "Recovered step" not in legacy_log:
        raise AssertionError(legacy_log)
    output_settings.write_text(
        zone("conflict", "every_steps = 1", multi=0).replace("conflict.fields", "./recovery-0.bin")
    )
    configure("grid.mesh", 0.006, 1)
    log = run([aero, f"--input={case}", f"--output={case/'overlap'}"], success=False)
    if "field output overlaps recovery data" not in log:
        raise AssertionError(log)
    print("OK failed write preserves previous data, corrupt-slot fallback and recovery on v3 archives")


def verify_sparse_batches(case, aero, meshdec, launcher):
    """Проверяет восстановление контрольной точки при неодинаковых размерах пакетов MPI-IO."""
    shutil.copytree(ROOT / "qa/cases/sod_1d", case)
    # Соседние исходные номера относятся к разным пространственным областям.
    # У наиболее загруженного процесса больше 256 КиБ записей; остальные заканчивают передачу раньше.
    count = 8192
    mesh = case / "grid.mesh"
    sparse_line(mesh, count)
    (case / "init.txt").write_text("initial_type formula\nrho = 1 + .1*x\nux = .2\np = 1 + .15*x\n")
    (case / "output.txt").write_text("")
    options = settings(case / "main.txt")
    options.update(
        reconstruction="first_order", time_scheme="euler", recovery_every_steps=1, omp_threads=1
    )

    def configure(mesh_name, end, weights):
        options.update(mesh=mesh_name, final_time=end, mpi_rank_weights=weights)
        (case / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in options.items()))

    run([meshdec, mesh, case / "grid.mdec", "--target-ranks=16"])
    native, mpi = case / "native", case / "mpi"
    configure("grid.mesh", 1e-7, "1")
    launch(aero, case, native, launcher, 1)
    configure("grid.mesh", 2e-7, "1")
    launch(aero, case, native, launcher, 1, "--recover")
    configure("grid.mdec", 1e-7, "3,1")
    first_log = launch(aero, case, mpi, launcher, 2)
    (case / "init.txt").unlink()
    configure("grid.mdec", 2e-7, "1,1,4")
    next_log = launch(aero, case, mpi, launcher, 3, "--recover")
    record = struct.Struct("<QQ6dQ")
    for log in (first_log, next_log):
        sizes = [int(rank["owned"]) * record.size for rank in report_rows(log, "rank\tnode")]
        if not min(sizes) <= 256 * 1024 < max(sizes):
            raise AssertionError(f"Unequal MPI-IO batch counts required for this test: {sizes}")

    # По формату AERORCV1 сверяем исходные номера, начальное и принятое поля,
    # а также контрольную сумму каждой записи.
    def compare(slot):
        name = f"recovery-{slot}.bin"
        ref, result = [(directory / name).read_bytes() for directory in (native, mpi)]
        if (
            len(result) != len(ref)
            or len(result) != 120 + count * record.size
            or result[:120] != ref[:120]
            or result[:8] != b"AERORCV1"
            or struct.unpack_from("<Q", result, 32)[0] != slot + 1
            or zlib.crc32(result[:112]) != struct.unpack_from("<Q", result, 112)[0]
        ):
            raise AssertionError("Incompatible checkpoint header or extent")
        for i in range(count):
            pos = 120 + i * record.size
            a, b = record.unpack_from(ref, pos), record.unpack_from(result, pos)
            if (
                a[:2] != (i, i)
                or a[:2] != b[:2]
                or a[5:8] != b[5:8]
                or not all(close(x, y) for x, y in zip(a[2:5], b[2:5]))
                or zlib.crc32(result[pos : pos + record.size - 8]) != b[-1]
            ):
                raise AssertionError(f"Sparse checkpoint mismatch at node {i}")

    compare(0)
    compare(1)
    with (mpi / "recovery-1.bin").open("r+b") as f:
        f.seek(120)
        f.write(b"\xff")
    log = launch(aero, case, mpi, launcher, 3, "--recover")
    if "Recovered step 1 at" not in log or "damaged record at node 0" not in log:
        raise AssertionError(log)
    compare(1)
    print(
        "OK sparse multi-batch MPI-IO, unequal rank sizes, and rollback after a first-batch CRC failure"
    )


def main():
    parser = mpi_parser(__doc__)
    args = parser.parse_args()
    with mpi_workspace("mpi-recovery-") as work:
        programs = (args.aero.resolve(), args.meshdec.resolve(), args.mpiexec)
        verify(Path(work) / "case", *programs)
        verify_sparse_batches(Path(work) / "sparse", *programs)


if __name__ == "__main__":
    main()
