"""Сопоставляет геометрию контрольных объёмов в MPI и на исходной сетке."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import math
import re
from pathlib import Path
import shutil
import struct
import subprocess


def run(args, success=True):
    """Выполняет команду проверки с ограничением времени и проверяет ожидаемый исход."""
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=120,
    )
    if (result.returncode == 0) != success:
        raise RuntimeError(f"{args}\n{result.stdout}")
    return result.stdout


def sparse_line(mesh, count):
    """Создаёт линию с перемежающимися глобальными номерами, чтобы соседние записи файла попали разным процессам."""
    # Соседние глобальные номера чередуются между удалёнными участками линии.
    half = count // 2
    node_ids = [2 * (i % half) + i // half for i in range(count)]
    blocks = [
        ("X", ((i // 2 + (i % 2) * half) / (count - 1) for i in range(count))),
        ("Y", [0] * count),
        ("Z", [0] * count),
        ("FACE_OFFSETS", range(count + 1)),
        ("FACE_NODES", node_ids),
        ("CELL_OFFSETS", range(0, 2 * count, 2)),
        ("CELL_FACES", (f for i in range(count - 1) for f in (i, i + 1))),
        ("CELL_ORIENTATIONS", [-1, 1] * (count - 1)),
    ]
    with mesh.open("w") as f:
        f.write(
            f"AEROMESH 3 TEXT\nDIMENSION 1\nCOUNTS {count} {count} {count} {count-1} {2*(count-1)} 2 2\n"
        )
        for name, values in blocks:
            f.write(name + "\n" + " ".join(map(str, values)) + "\n")
        f.write(f'BOUNDARY_FACES 0 {count-1}\nBOUNDARY_PATCH 0 1\nPATCH_NAMES "xmin" "xmax"\nEND\n')


def geometry(directory, ranks):
    """Читает таблицы контрольных объёмов и площадок всех разделов для сравнения с целой сеткой."""
    records = {}
    owners = []
    for rank in range(ranks):
        owned = 0
        path = directory / f"geometry-{rank}-of-{ranks}.tsv"
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                continue
            fields = line.split()
            start = {"V": 2, "E": 3, "B": 4}[fields[0]]
            key = tuple(fields[:start])
            if key in records:
                raise AssertionError(f"Duplicate geometry owner for {key}")
            records[key] = tuple(map(float, fields[start:]))
            owned += fields[0] == "V"
        owners.append(owned)
    return records, owners


def compare(expected, actual):
    """Сверяет ожидаемые значения с полученными и сообщает место расхождения."""
    if expected.keys() != actual.keys():
        raise AssertionError(f"Missing or extra entities: {list(expected.keys() ^ actual.keys())[:5]}")
    for key, ref in expected.items():
        values = actual[key]
        scale = max(map(abs, ref))
        if len(values) != len(ref) or any(
            not math.isclose(x, y, rel_tol=3e-13, abs_tol=3e-13 * scale) for x, y in zip(ref, values)
        ):
            raise AssertionError(f"{key}: {ref} vs {values}")


def legacy_archive(current, target, version=1):
    """Строит из проверочного архива образ прежней версии для проверки совместимости."""
    data = bytearray(current.read_bytes())
    # В старом заголовке перед счётчиком узлов было дополнительное вещественное
    # поле. Увеличиваем все абсолютные адреса каталога и листьев на его размер.
    leaves, branches = struct.unpack_from("<QQ", data, 72)
    names, index, tree, payload = struct.unpack_from("<QQQQ", data, 96)
    directory = tree + 112 * branches
    if version == 1:
        keep = directory
    elif version == 2:
        nodes = struct.unpack_from("<Q", data, 32)[0]
        stars, halo = struct.unpack_from("<QQ", data, directory)
        keep = directory + 16 + 8 * (nodes + 1 + 2 * leaves + 1) + 16 * (stars + halo)
    elif version == 3:
        keep = payload - 64
    elif version == 4:
        keep = payload
    else:
        raise ValueError("Expected a legacy archive version 1 through 4")
    del data[keep:payload]
    data[32:32] = struct.pack("<d", 1.)
    struct.pack_into("<Q", data, 8, version)
    struct.pack_into("<QQQQ", data, 104, names + 8, index + 8, tree + 8, keep + 8)
    for i in range(leaves):
        pos = index + 8 + 48 * i
        offset = struct.unpack_from("<Q", data, pos)[0]
        struct.pack_into("<Q", data, pos, offset + 8 - (payload - keep))
    target.write_bytes(data)


def verify_case(work, case, aero, meshdec, launcher):
    """Сопоставляет геометрию одной постановки на исходной сетке и после разбиения."""
    source = ROOT / "qa" / "cases" / case
    params = work / case
    shutil.copytree(source, params)
    native = params / "grid.mesh"
    if case == "shear_jump_2d":
        # Масштабированная плоскость вложена в 3D так, что векторы площадок имеют ненулевую компоненту Z.
        words = native.read_text().split()
        count = int(words[words.index("COUNTS") + 1])
        xyz = [words.index(axis) + 1 for axis in "XYZ"]
        for i in range(count):
            x, y = (float(words[j + i]) for j in xyz[:2])
            words[xyz[2] + i] = repr(0.25 * x + 0.5 * y)
        words[words.index("LENGTH_SCALE_TO_METERS") + 1] = "0.375"
        native.write_text(" ".join(words))
    out = params / "results"
    def solver_command(directory, ranks=1):
        """У каждого независимого расчёта свой каталог полей и recovery."""
        mpi = [launcher, "-n", ranks] if ranks > 1 else []
        return [*mpi, aero, f"--input={params}", f"--output={directory}"]

    command = solver_command(out)
    run(command + ["--geometry-only"])
    reference, _ = geometry(out, 1)
    if case == "pulse_1d":
        run(command)
        reference_fields = (out / "final.fields").read_bytes()
    archive = params / "grid.mdec"
    compression = ["--compress"] if case != "wave_3d" else []
    run([meshdec, native, archive, "--target-ranks=4", *compression])
    # Исходный файл нужен только для эталона. Для распределённого запуска оставляем один архив.
    native.unlink()
    settings = params / "main.txt"
    settings.write_text(settings.read_text().replace("grid.mesh", "grid.mdec"))
    for ranks in (1, 2, 4):
        shutil.rmtree(out)
        run([launcher, "-n", ranks, *command, "--geometry-only"])
        result, owners = geometry(out, ranks)
        compare(reference, result)
        if not all(owners):
            raise AssertionError(f"Expected a real split: {owners}")
        print(f"OK {case}  processes={ranks}  owned vertices={owners}  records={len(result)}")
    if case == "pulse_1d":
        run(command)
        if (out / "final.fields").read_bytes() != reference_fields:
            raise AssertionError("Serial solve from archive differs from native periodic solve")
        print("OK single-process periodic SSPRK3 field output")
        # Архив v4 хранит SHA-256 и старое слово масштаба. Проверяем смещения
        # каталога на двух процессах, пока исходная сетка отсутствует.
        legacy_archive(archive, params / "v4.mdec", 4)
        run([launcher, "-n", 2, *command, "--geometry-only", "main:mesh=v4.mdec"])
        old_geometry, _ = geometry(out, 2)
        compare(reference, old_geometry)
        legacy_archive(archive, params / "old.mdec")
        original = settings.read_text()
        legacy = original.replace("grid.mdec", "old.mdec")
        settings.write_text(legacy)
        legacy_output = params / "legacy-serial"
        run(solver_command(legacy_output))
        if (legacy_output / "final.fields").read_bytes() != reference_fields:
            raise AssertionError("Legacy serial fields differ")
        mpi = [launcher, "-n", 2, *command]
        run(mpi + ["--geometry-only"])
        result, _ = geometry(out, 2)
        compare(reference, result)
        settings.write_text(re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = first_order", legacy))
        run(solver_command(params / "legacy-first-order", 2))
        for scheme in ("ebr3", "weno3", "weno5"):
            settings.write_text(
                re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = " + scheme, legacy)
            )
            if "regenerate the .mdec" not in run(mpi, success=False):
                raise AssertionError(f"Missing {scheme} diagnostic for legacy archive without halo")
        legacy_archive(archive, params / "v2.mdec", 2)
        v2 = original.replace("grid.mdec", "v2.mdec")
        for scheme in ("ebr3", "weno3"):
            settings.write_text(re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = " + scheme, v2))
            run(solver_command(params / ("v2-" + scheme), 2))
        for scheme in ("ebr5", "weno5"):
            settings.write_text(re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = " + scheme, v2))
            if "far halo index" not in run(mpi, success=False):
                raise AssertionError(f"Missing {scheme} diagnostic for v2")
        settings.write_text(original)
        archive.unlink()
        run([meshdec, source / "grid.mesh", archive])
        run([launcher, "-n", 4, *command, "--geometry-only"])
        result, owners = geometry(out, 4)
        compare(reference, result)
        if owners.count(0) != 3:
            raise AssertionError(f"Expected three empty partitions: {owners}")
        print("OK one leaf on four processes")
    if case == "wave_3d":
        # Неверные границы каталога и ссылки должны вызвать согласованный отказ всех процессов.
        data = bytearray(archive.read_bytes())
        payload = struct.unpack_from("<Q", data, 120)[0]
        if struct.unpack_from("<Q", data, payload)[0] != 0:
            raise AssertionError("Expected a vertex at start of Morton order")
        first_id = struct.unpack_from("<Q", data, payload + 8)[0]
        struct.pack_into("<Q", data, payload + 8, first_id + 1)
        archive.write_bytes(data)
        error = run([launcher, "-n", 4, *command, "--geometry-only"], success=False)
        if "global node owner" not in error:
            raise AssertionError(error)
        struct.pack_into("<Q", data, payload + 8, first_id)
        original_link = struct.unpack_from("<Q", data, payload + 48)[0]
        struct.pack_into("<Q", data, payload + 48, 2**64 - 1)
        archive.write_bytes(data)
        error = run([launcher, "-n", 4, *command, "--geometry-only"], success=False)
        if "nonexistent leaf" not in error:
            raise AssertionError(error)
        tree = struct.unpack_from("<Q", data, 112)[0]
        branches = struct.unpack_from("<Q", data, 80)[0]
        directory = tree + 112 * branches
        star_links, halo_links = struct.unpack_from("<QQ", data, directory)
        leaves = struct.unpack_from("<Q", data, 72)[0]
        node_count = struct.unpack_from("<Q", data, 32)[0]
        offsets = directory + 16 + 8 * (node_count + 1) + 16 * star_links
        struct.pack_into("<Q", data, payload + 48, original_link)
        first_end = struct.unpack_from("<Q", data, offsets + 8)[0]
        struct.pack_into("<Q", data, offsets + 8, 2**64 - 1)
        archive.write_bytes(data)
        if "invalid halo row extent" not in run(
            [launcher, "-n", 4, *command, "--geometry-only"], success=False
        ):
            raise AssertionError("Damaged precomputed halo accepted")
        struct.pack_into("<Q", data, offsets + 8, first_end)
        far = offsets + 8 * (2 * leaves + 1) + 16 * halo_links
        struct.pack_into("<Q", data, far + 16, 2**64 - 1)
        archive.write_bytes(data)
        mpi = [launcher, "-n", 4, *command]
        config = settings.read_text()
        settings.write_text(re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = ebr3", config))
        run(mpi)
        settings.write_text(re.sub(r"(?m)^reconstruction\s*=.*", "reconstruction = ebr5", config))
        if "invalid halo row extent" not in run(mpi, success=False):
            raise AssertionError("Damaged EBR5 halo accepted")
        print(
            "OK collective rejection of duplicate ownership, invalid links and damaged near/far halo rows"
        )


def main():
    parser = mpi_parser(__doc__)
    args = parser.parse_args()
    with mpi_workspace("mpi-geometry-") as work:
        for case in ("pulse_1d", "shear_jump_2d", "wave_3d"):
            verify_case(Path(work), case, args.aero.resolve(), args.meshdec.resolve(), args.mpiexec)


if __name__ == "__main__":
    main()
