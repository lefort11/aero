"""Сверяет срез с полным снимком той же задачи, включая разбиение по MPI."""
import measurements
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
    zones = ["[volume]\nfile=volume.fields\nvariables=primary,mach\nevery_steps=1\nlayout=global\nvisualization=vtk,tecplot"]
    zones.append("[local]\nfile=local.fields\nvariables=rho\nevery_steps=1\nvisualization=vtk")
    mappings = {}
    for mode, offset in (("faces", ".5"), ("cut", ".53125")):
        path = case / f"{mode}.submesh"
        other = case / f"{mode}-archive.submesh"
        for base, dest in ((grid, path), (archive, other)):
            run([program("mesh_slice"), base, dest, mode, "0", "0", "1", offset])
        assert measurements.compare(path.read_bytes(), "==", other.read_bytes())
        with path.open("rb") as stream:
            assert stream.readline().strip() == b"AEROSLICE 1"
            count = int(stream.readline().split()[2])
            assert measurements.compare(count, "==", 225)
            rows = [stream.readline().split() for _ in range(count)]
        mapping = [(int(a), int(b), float(w)) for a, b, w in rows]
        assert all(measurements.compare(a, "==", b) and measurements.compare(w, "==", 0) for a, b, w in mapping) if mode == "faces" else all(
            measurements.compare(a, "!=", b) and (measurements.chain(0, "<", w, "<", 1)) for a, b, w in mapping)
        mappings[mode] = mapping
        zones.append(f"[{mode}]\nfile={mode}.fields\nvariables=rho,uz,p,mach\nevery_steps=1\nsubmesh={path.name}\nvisualization=vtk,tecplot")
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
            assert measurements.error_below(abs(mach - speed / math.sqrt(1.4 * volume["p"][i] / volume["rho"][i])), 1.e-13)
        for mode, rows in mappings.items():
            field_path = dest / f"{mode}.fields"
            meta, fields = binary_snapshot(field_path)
            assert measurements.compare(meta[0], "==", 2) and measurements.compare(meta[2][0], "==", len(rows)) and measurements.compare(meta[2][3], "==", 196)
            for name, field in fields.items():
                for actual, (a, b, w) in zip(field, rows):
                    expected = (1-w)*volume[name][a] + w*volume[name][b]
                    assert measurements.error_below(abs(actual-expected), 1.e-13), (mode, name, actual, expected)
        # Побайтовая сверка охватывает общую нумерацию MPI и оба способа интерполяции.
        for name in ("volume", *mappings):
            geometry = grid if name == "volume" else case / f"{name}.submesh"
            for ext in ("plt", "vtk"):
                converted = dest / f"converted.{ext}"
                run([program("visfield"), geometry, dest / f"{name}.fields", converted])
                assert measurements.compare(converted.read_bytes(), "==", (dest / f"{name}.{ext}").read_bytes())
        for part in (sorted(dest.glob("rank-*-of-*")) if distributed else [dest]):
            fields = part / "fields" if distributed else part
            local_grid = part / "grid.mesh" if distributed else grid
            converted = dest / "converted-local.vtk"
            run([program("visfield"), local_grid, fields / "local.fields", converted])
            assert measurements.compare(converted.read_bytes(), "==", (fields / "local.vtk").read_bytes())
            assert not (fields / "local.plt").exists()
        print("OK submesh, interpolation, local/global Tecplot and VTK, MPI =", distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi", action="store_true")
    with mpi_workspace("submesh-") as directory:
        check(Path(directory), parser.parse_args().mpi)
