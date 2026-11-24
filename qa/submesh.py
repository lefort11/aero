"""Сверяет срез с полным снимком той же задачи, включая разбиение по MPI."""
import argparse
import math
import os
import shutil
from pathlib import Path

from mpi_common import ROOT, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run


# Создаёт срез сетки и сравнивает интерполированные поля разных режимов запуска.
def check(case, mpi):
    # Находит утилиту, необходимую для проверки подсеток.
    def program(name):
        return ROOT / "bin" / (name + (".exe" if os.name == "nt" else ""))
    grid = case / "grid.mesh"
    for source in (ROOT / "qa/cases/wave_3d").iterdir():
        if source.suffix in (".txt", ".mesh"):
            shutil.copyfile(source, case / source.name)
    archive = case / "grid.mdec"
    run([program("meshdec"), grid, archive, "--target-ranks=2", "--compress"])
    zones = ["[volume]\nfile=volume.fields\nvariables=primary,mach\nevery_steps=1\nlayout=global"]
    mappings = {}
    for mode, offset in (("faces", ".5"), ("cut", ".53125")):
        path = case / f"{mode}.submesh"
        other = case / f"{mode}-archive.submesh"
        for base, dest in ((grid, path), (archive, other)):
            run([program("mesh_slice"), base, dest, mode, "0", "0", "1", offset])
        assert path.read_bytes() == other.read_bytes()
        with path.open("rb") as stream:
            assert stream.readline().strip() == b"AEROSLICE 1"
            count = int(stream.readline().split()[2])
            assert count == 225
            rows = [stream.readline().split() for _ in range(count)]
        mapping = [(int(a), int(b), float(w)) for a, b, w in rows]
        assert all(a == b and w == 0 for a, b, w in mapping) if mode == "faces" else all(
            a != b and 0 < w < 1 for a, b, w in mapping)
        mappings[mode] = mapping
        zones.append(f"[{mode}]\nfile={mode}.fields\nvariables=rho,uz,p,mach\nevery_steps=1\nsubmesh={path.name}")
    (case / "output.txt").write_text("\n".join(zones))
    for distributed in ([False, True] if mpi else [False]):
        dest = case / ("mpi" if distributed else "serial")
        launch = ["mpiexec", "-n", "2"] if distributed else []
        mesh = archive if distributed else grid
        run([*launch, program("aero"), f"--input={case}", f"--output={dest}", f"main:mesh={mesh}",
             "main:final_time=.001"])
        volume = binary_snapshot(dest / "volume.fields")[1]
        for i, mach in enumerate(volume["mach"]):
            speed = math.sqrt(sum(volume[axis][i]**2 for axis in ("ux", "uy", "uz")))
            assert abs(mach - speed / math.sqrt(1.4 * volume["p"][i] / volume["rho"][i])) < 1.e-13
        for mode, rows in mappings.items():
            field_path = dest / f"{mode}.fields"
            meta, fields = binary_snapshot(field_path)
            assert meta[0] == 2 and meta[2][0] == len(rows) and meta[2][3] == 196
            for name, field in fields.items():
                for actual, (a, b, w) in zip(field, rows):
                    expected = (1-w)*volume[name][a] + w*volume[name][b]
                    assert abs(actual-expected) < 1.e-13, (mode, name, actual, expected)
            run([program("fields_to_tecplot"), case / f"{mode}.submesh", field_path, dest / f"{mode}.plt"])
        print("OK submesh: both plane modes, interpolation, Tecplot, MPI =", distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi", action="store_true")
    with mpi_workspace("submesh-") as directory:
        check(Path(directory), parser.parse_args().mpi)
