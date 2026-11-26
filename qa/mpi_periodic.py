"""Проверяет склейку периодических узлов, поток через шов и полноту шаблонов на границе процессов."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import math
import os
from pathlib import Path
import random
import re
import shutil

from mdec_header import read_mdec_header
from mpi_geometry import geometry, run
from mpi_euler import (
    free_patch,
    check_output,
    check_report,
    launch,
    mesh_setting,
    stamp,
    verify as verify_euler,
)
from run import nodes, read_snapshot, reconstruction_variables, settings, write_settings


def square(mesh):
    """Строит квадратную сетку для проверки периодических швов."""
    mesh.write_text("""AEROMESH 2 TEXT
DIMENSION 2
COUNTS 4 4 8 1 4 4 4
LENGTH_SCALE_TO_METERS 1
X 0 1 1 0
Y 0 0 1 1
Z 0 0 0 0
FACE_OFFSETS 0 2 4 6 8
FACE_NODES 0 1 1 2 2 3 3 0
CELL_OFFSETS 0 4
CELL_FACES 0 1 2 3
CELL_ORIENTATIONS 1 1 1 1
BOUNDARY_FACES 0 1 2 3
BOUNDARY_PATCH 2 1 3 0
PATCH_NAMES "xmin" "xmax" "ymin" "ymax"
END
""")


def short_line(mesh):
    """Строит короткую линию для проверки периодических образов."""
    mesh.write_text("""AEROMESH 2 TEXT
DIMENSION 1
COUNTS 9 9 9 8 16 2 2
LENGTH_SCALE_TO_METERS 1
X 0 .125 .25 .375 .5 .625 .75 .875 1
Y 0 0 0 0 0 0 0 0 0
Z 0 0 0 0 0 0 0 0 0
FACE_OFFSETS 0 1 2 3 4 5 6 7 8 9
FACE_NODES 0 1 2 3 4 5 6 7 8
CELL_OFFSETS 0 2 4 6 8 10 12 14 16
CELL_FACES 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8
CELL_ORIENTATIONS -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1 -1 1
BOUNDARY_FACES 0 8
BOUNDARY_PATCH 0 1
PATCH_NAMES "xmin" "xmax"
END
""")


def contact_trace(scheme, values, h, cfl):
    """Вычисляет след на контакте по заданной формуле реконструкции для независимой сверки."""
    a, b, c, d, e = values
    if scheme in ("ebr5", "mp5", "MPWENO5"):
        trace = (2 * a - 13 * b + 47 * c + 27 * d - 3 * e) / 60
        if scheme == "ebr5":
            return trace
        # Проверяем MP5 в значениях узлов независимо от упаковки наклонов в солвере.
        alpha = min(4, max(0, 1 / cfl - 1))
        def minmod(*x):
            return max(0, min(x)) + min(0, max(x))
        delta = 2 * (trace - c)
        bound = minmod(d-c, alpha*(c-b))
        if scheme == "MPWENO5":
            far, near, center, opposite = b-a, c-b, d-c, e-d
            beta = (13/12*(near-far)**2 + .25*(3*near-far)**2,
                    13/12*(center-near)**2 + .25*(center+near)**2,
                    13/12*(opposite-center)**2 + .25*(opposite-3*center)**2)
            if abs(beta[0]-beta[2]) >= min(beta):
                # Независимая нормировка по величине наибольшего индикатора.
                # На постоянном поле все три полинома возвращают одну константу.
                if max(beta) == 0:
                    return c
                peak = max(beta)
                beta = tuple(v/peak for v in beta)
                eps = 1e-40
                tau = abs(.5*(beta[0]+beta[2])-beta[1])
                variation = math.sqrt(sum((beta[i]-beta[(i+1)%3])**2 for i in range(3)))
                hden = sum(w*v for w,v in zip((.1,.6,.3),beta)) + variation + eps
                chi = variation/hden
                peak = max(beta)+eps
                factors = [peak*(v+eps+tau)-.8*(v+eps)*(peak+tau) for v in beta]
                weights = [w * (hden**2*(v+eps+tau**2/(v+eps))
                                  +(v+eps)*chi*(2-chi)*hden**2*v/(v+eps+variation))
                           *factors[(i+1)%3]*factors[(i+2)%3]
                           for i,(w,v) in enumerate(zip((.1,.6,.3),beta))]
                slopes = ((5*near-2*far)/3,(near+2*center)/3,(4*center-opposite)/3)
                delta = sum(w*v for w,v in zip(weights,slopes))/sum(weights)
                trace = c+.5*delta
            acceptable = min(0,bound) <= delta <= max(0,bound)
        else:
            acceptable = delta * (delta - bound) < 1e-10
        if acceptable:
            return trace
        def curvature(x, y):
            return minmod(x, y, 4*x-y, 4*y-x)
        prev, mid, nxt = a-2*b+c, b-2*c+d, c-2*d+e
        md = (c+d-curvature(mid,nxt)) / 2
        lc = c + (c-b)/2 + 4/3*curvature(prev,mid)
        ul = c + alpha*(c-b)
        low = max(min(c,d,md),min(c,ul,lc))
        high = min(max(c,d,md),max(c,ul,lc))
        return min(high,max(low,trace))
    if scheme == "weno3":
        candidates = ((3 * c - b) / 2, (c + d) / 2)
        beta = ((c - b) ** 2, (d - c) ** 2)
        ideal = (1 / 3, 2 / 3)
    else:
        candidates = ((2 * a - 7 * b + 11 * c) / 6, (-b + 5 * c + 2 * d) / 6, (2 * c + 5 * d - e) / 6)
        beta = (
            13 / 12 * (a - 2 * b + c) ** 2 + 0.25 * (a - 4 * b + 3 * c) ** 2,
            13 / 12 * (b - 2 * c + d) ** 2 + 0.25 * (b - d) ** 2,
            13 / 12 * (c - 2 * d + e) ** 2 + 0.25 * (3 * c - 4 * d + e) ** 2,
        )
        ideal = (0.1, 0.6, 0.3)
    weights = [w / (v + h * h * 1e-40) ** 2 for w, v in zip(ideal, beta)]
    return sum(w * q for w, q in zip(weights, candidates)) / sum(weights)


def check_contact(ref, logical, names, reconstruction, cfl):
    """Сверяет эталонный перенос контакта со значениями в узлах."""
    path = next(p for p in ref.glob("*.fields") if stamp(p)[0] == 1)
    _, dt = stamp(path)
    actual = read_snapshot(path, len(logical), 1, dt, names)["rho"][1]
    n = len(logical) - 1
    rho = [1 + 0.25 * math.cos(math.pi * (x - 0.25)) ** 4 for x, _, _ in logical[:-1]]
    # Схема Роу переносит этот контакт с единичной скоростью. Донорные узлы берём через периодический шов.
    flux = [
        contact_trace(reconstruction, [rho[(i + k) % n] for k in range(-2, 3)], 1 / n, cfl) for i in range(n)
    ]
    for i, value in enumerate(actual):
        j = i % n
        expected = rho[j] - dt * n * (flux[j] - flux[j - 1])
        if not math.isclose(value, expected, rel_tol=3e-14):
            raise AssertionError(
                f"Incorrect {reconstruction} contact flux at node {i}: {value} vs {expected}"
            )


def warp(mesh, logical):
    """Деформирует координаты тестовой сетки, сохраняя её топологию."""
    # Парные грани совмещаются сдвигом, но прилегающие ячейки имеют разную ширину и разные векторы площадок.
    # После перенумерации у некоторых рёбер шва также меняется направление.
    words = mesh.read_text().split()
    dst = list(range(len(logical)))
    random.Random(17).shuffle(dst)
    xyz = [words.index(a) + 1 for a in "XYZ"]
    reordered = [None] * len(dst)
    for i, (x, y, z) in enumerate(logical):
        a = x + 0.015 * math.sin(math.pi * x) * math.cos(2 * math.pi * y)
        b = y + 0.01 * math.cos(2 * math.pi * x) * math.sin(math.pi * y)
        for start, value in zip(xyz, (a, b, 0.25 * a + 0.5 * b)):
            words[start + dst[i]] = repr(value)
        reordered[dst[i]] = (x, y, z)
    for i in range(words.index("FACE_NODES") + 1, words.index("CELL_OFFSETS")):
        words[i] = str(dst[int(words[i])])
    words[words.index("LENGTH_SCALE_TO_METERS") + 1] = ".375"
    mesh.write_text(" ".join(words))
    return reordered


def check_seam(out, ranks, logical, axes, names, dim, volumes, closed, split):
    """Сопоставляет периодические копии и проверяет объёмы и связи по обе стороны шва."""
    copies = []
    for rank in range(ranks):
        part = out / f"rank-{rank}-of-{ranks}"
        mapping = [
            tuple(map(int, row.split())) for row in (part / "nodes.tsv").read_text().splitlines()[1:]
        ]
        copies.append((part / "fields", mapping))
    first_integral = None
    for path in sorted(copies[0][0].glob("*.fields")):
        time = stamp(path)[1]
        states, processes = {}, {}
        rho, energy = [], []
        for rank, (directory, mapping) in enumerate(copies):
            fields = read_snapshot(directory / path.name, len(mapping), dim, time, names)
            for local, node, owner in mapping:
                if owner != rank:
                    continue
                xyz = logical[node]
                key = tuple(0.0 if i in axes and v == 1.0 else v for i, v in enumerate(xyz))
                state = tuple(fields[n][1][local] for n in names)
                if states.setdefault(key, state) != state:
                    raise AssertionError(f"Periodic copies differ at step {path.name}, node {node}")
                processes.setdefault(key, set()).add(rank)
                rho.append(volumes[node] * state[0])
                energy.append(volumes[node] * state[-1])
        if split and not any(len(s) > 1 for s in processes.values()):
            raise AssertionError("Expected periodic copies on different MPI processes")
        if closed:
            total = math.fsum(rho), math.fsum(energy)
            if first_integral is None:
                first_integral = total
            if not all(math.isclose(a, b, rel_tol=4e-14) for a, b in zip(total, first_integral)):
                raise AssertionError(f"Periodic mass/energy drift: {total} vs {first_integral}")


def report_rows(log, prefix):
    """Разбирает табличную часть отчёта после заданного заголовка."""
    lines = log.split("Starting time integration")[0].splitlines()
    first = next(i for i, v in enumerate(lines) if v.startswith(prefix + "\t"))
    columns = lines[first].split("\t")
    rows = []
    for line in lines[first + 1 :]:
        values = line.split("\t")
        if not values[0].isdigit():
            break
        if len(values) != len(columns):
            raise AssertionError(f"Incomplete partition report row: {line}")
        rows.append(dict(zip(columns, values)))
    return rows


def check_partition_report(log, serial, ranks, archive, out, logical, axes, dim, geometry_rows):
    """Проверяет, что отчёт разделов согласуется с геометрией и сведениями архива."""
    parts = report_rows(log, "rank\tnode")
    hosts = report_rows(log, "node\thost")
    if [int(p["rank"]) for p in parts] != list(range(ranks)):
        raise AssertionError("Missing or duplicate MPI rank in partition report")

    def total(key):
        return sum(int(p[key]) for p in parts)

    leaves = read_mdec_header(archive).leaves
    reference = re.search(r"Computational nodes = (\d+)\s+edges = (\d+)", serial)
    if (total("owned"), total("edges"), total("leaves")) != (*map(int, reference.groups()), leaves):
        raise AssertionError("Distributed workload differs from serial grid or archive")
    for field in ("state", "flux"):
        if total(field + "_send_B") != total(field + "_recv_B"):
            raise AssertionError(f"Unbalanced {field} exchange")
    for p in parts:
        if int(p["state_recv_B"]) != int(p["halo"]) * (dim + 2) * 8:
            raise AssertionError("State import bytes differ from halo size")
        if int(p["leaves"]) == 0 and any(
            int(p[k])
            for k in (
                "owned",
                "halo",
                "edges",
                "state_send_B",
                "state_recv_B",
                "flux_send_B",
                "flux_recv_B",
            )
        ):
            raise AssertionError("Empty partition reports work or traffic")
    # Независимо считаем межпроцессные связи по исходным рёбрам и выгруженным номерам владельцев.
    # На периодической линии склеены только два её конца.
    if not axes or dim == 1:
        owners = {}
        for rank in range(ranks):
            path = out / f"rank-{rank}-of-{ranks}" / "nodes.tsv"
            for line in path.read_text().splitlines()[1:]:
                _, node, owner = map(int, line.split())
                owners[node] = owner
        if axes:
            ends = [i for i, xyz in enumerate(logical) if xyz[0] in (0.0, 1.0)]
            owners[max(ends)] = owners[min(ends)]
        edges, sent, received = ([0] * ranks for _ in range(3))
        for key in geometry_rows:
            if key[0] == "E":
                left, right = (owners[int(i)] for i in key[1:])
                edges[left] += 1
                if left != right:
                    sent[left] += (dim + 3) * 8
                    received[right] += (dim + 3) * 8
        for i, p in enumerate(parts):
            if (int(p["edges"]), int(p["flux_send_B"]), int(p["flux_recv_B"])) != (
                edges[i],
                sent[i],
                received[i],
            ):
                raise AssertionError(f"Incorrect computed edges or flux exchange on rank {i}")
    if {n["node"] for n in hosts} != {p["node"] for p in parts} or len(hosts) != len(
        {n["node"] for n in hosts}
    ):
        raise AssertionError("Incorrect node groups")
    for n in hosts:
        members = [p for p in parts if p["node"] == n["node"]]
        if int(n["ranks"]) != len(members) or n["node"] != members[0]["rank"] or not n["host"]:
            raise AssertionError("Incorrect node leader or number of ranks")
        for src, dst in (
            ("stencil_peak_KiB", "stencil_peaks_sum_KiB"),
            ("startup_peak_KiB", "startup_peaks_sum_KiB"),
            ("ready_rss_KiB", "ready_rss_sum_KiB"),
        ):
            values = [p[src] for p in members]
            expected = "n/a" if "n/a" in values else str(sum(map(int, values)))
            if n[dst] != expected or (os.name == "nt" and expected in ("n/a", "0")):
                raise AssertionError(f"Incorrect node memory summary for {src}")
        for p in members:
            for field in ("stencil_peak_KiB", "ready_rss_KiB"):
                if "n/a" not in (p[field], p["startup_peak_KiB"]) and int(p[field]) > int(
                    p["startup_peak_KiB"]
                ):
                    raise AssertionError("Inconsistent RSS high-water mark")
        if "n/a" not in (n["ram_total_KiB"], n["ram_available_KiB"]) and int(
            n["ram_available_KiB"]
        ) > int(n["ram_total_KiB"]):
            raise AssertionError("Available RAM exceeds total RAM")


def verify(
    work,
    title,
    fixture,
    dim,
    axes,
    aero,
    meshdec,
    launcher,
    tiny=False,
    empty=False,
    reconstruction="first_order",
    archive=None,
    narrow=False,
):
    """Проверяет склейку периодических узлов, поток через шов и полноту шаблонов на границе процессов."""
    case = work / title
    shutil.copytree(ROOT / "qa/cases" / fixture, case)
    native = case / "grid.mesh"
    if narrow:
        short_line(native)
    elif tiny:
        square(native)
    logical = nodes(native, dim)
    tilted = dim == 2 and not tiny
    if tilted:
        logical = warp(native, logical)
    bc = ""
    for i, axis in enumerate("xyz"[:dim]):
        if i in axes:
            shift = [int(i == j) for j in range(3)]
            if tilted:
                shift[2] = (0.25, 0.5)[i]
            bc += (
                f"[{axis}min]\nbc_type periodic\npair {axis}max\ntranslation "
                + " ".join(map(str, shift))
                + "\n\n"
            )
        else:
            for side in ("min", "max"):
                bc += free_patch(axis + side)
    (case / "bc.txt").write_text(bc)
    if dim == 2:
        (case / "init.txt").write_text(
            "initial_type formula\nrho = 1 + .1*cos(2*pi*x) + .03*sin(2*pi*y)\n"
            "ux = .6\nuy = -.2\np = 1 + .07*sin(2*pi*x)\n"
        )
    names = ["rho", *("u" + c for c in "xyz"[:dim]), "p", "E"]
    (case / "output.txt").write_text(
        "[history]\nfile = state.fields\nevery_steps = 1\nmulti = 1\n"
        "format = text\nvariables = " + ", ".join(names) + "\n"
    )
    if tiny:
        # Единственная ячейка периодична по обоим направлениям. Получается один узел и два ребра-петли.
        final = 0.35
    else:
        final = (0.02, 0.006, 0.012)[dim - 1]
    options = settings(case / "main.txt")
    options.update(
        mesh="grid.mesh",
        reconstruction=reconstruction,
        euler_solver="lf" if tilted else "roe",
        reconstruction_variables="flux" if dim == 2 and "flux" in reconstruction_variables(reconstruction) else "characteristic",
        report_every=1,
        final_time=final,
        omp_threads=2,
    )

    def configure(scheme):
        options["time_scheme"] = scheme
        write_settings(case / "main.txt", options)

    configure("euler")
    geom = case / "geometry"
    run([aero, f"--input={case}", f"--output={geom}", "--geometry-only"])
    data, _ = geometry(geom, 1)
    volumes = [data["V", str(i)][0] for i in range(len(logical))]
    for scheme in ("euler", "ssprk3"):
        configure(scheme)
        ref = case / f"ref-{scheme}"
        (case / f"{scheme}.log").write_text(run([aero, f"--input={case}", f"--output={ref}"]))
    if narrow:
        check_contact(case / "ref-euler", logical, names, reconstruction, float(options["cfl"]))
    if archive is None:
        run(
            [meshdec, native, case / "grid.mdec", "--compress", *([] if empty else ["--target-ranks=4"])]
        )
        archive = case / "grid.mdec"
    native.unlink()
    options["mesh"] = os.path.relpath(archive, case).replace(os.sep, "/")
    for scheme in ("euler", "ssprk3"):
        configure(scheme)
        ref = case / f"ref-{scheme}"
        for ranks in ((4,) if empty else (2,) if tiny else (2, 4)):
            out = case / f"{scheme}-{ranks}"
            log = launch(aero, case, out, launcher, ranks)
            serial = (case / f"{scheme}.log").read_text()
            check_report(serial, log)
            check_partition_report(log, serial, ranks, archive, out, logical, axes, dim, data)
            if reconstruction != "first_order":
                pattern = r"downgraded reconstruction sides = (\d+)"
                if re.search(pattern, serial)[1] != re.search(pattern, log)[1]:
                    raise AssertionError("The partition introduced incomplete reconstruction rows")
            error = check_output(ref, out, ranks, len(logical), dim, names, empty)
            check_seam(
                out,
                ranks,
                logical,
                axes,
                names,
                dim,
                volumes,
                len(axes) == dim,
                bool(axes) and not empty,
            )
            print(f"OK {title} {scheme} ranks={ranks} max field difference={error:.3g}")
    if dim == 1 and axes and reconstruction == "first_order":
        command = [launcher, "-n", 4, aero, f"--input={case}", f"--output={case/'invalid'}"]
        (case / "bc.txt").write_text(bc.replace("translation 1 0 0", "translation 1.01 0 0"))
        if "unmatched node" not in run(command, success=False):
            raise AssertionError("Missing periodic translation diagnostic")
        (case / "bc.txt").write_text(bc)
        (case / "init.txt").write_text("initial_type formula\nrho = 1+x\nux = 0\np = 1\n")
        if "periodic mismatch" not in run(command, success=False):
            raise AssertionError("Missing periodic initial state diagnostic")


def verify_layer_face(work, aero, meshdec, launcher, reconstruction):
    """Проверяет реконструкцию у грани, вершины которой входят в нужный слой шаблона."""
    # Вокруг вершины 1 невыпуклое окружение. Луч вдоль +X пересекает узел 4 при x = 0.5
    # и внешнюю грань (3,5) при x = 2. Оба конца грани находятся в первом слое,
    # а её единственная ячейка (3,5,4) не содержит ни одного конца ребра (0,1).
    case = work / "concave-star"
    shutil.copytree(ROOT / "qa/cases/shear_jump_2d", case)
    mesh = case / "grid.mesh"
    mesh.write_text("""AEROMESH 2 TEXT
DIMENSION 2
COUNTS 7 13 26 7 21 5 1
LENGTH_SCALE_TO_METERS 1
X -1 0 0 2 .5 2 0
Y 0 0 -1 -1 0 1 1
Z 0 0 0 0 0 0 0
FACE_OFFSETS 0 2 4 6 8 10 12 14 16 18 20 22 24 26
FACE_NODES 1 0 0 2 2 1 2 3 3 1 3 4 4 1 4 5 5 1 5 6 6 1 6 0 3 5
CELL_OFFSETS 0 3 6 9 12 15 18 21
CELL_FACES 0 1 2 2 3 4 4 5 6 6 7 8 8 9 10 10 11 0 12 7 5
CELL_ORIENTATIONS 1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 -1 1 -1 -1
BOUNDARY_FACES 1 3 9 11 12
BOUNDARY_PATCH 0 0 0 0 0
PATCH_NAMES "wall"
END
""")
    (case / "bc.txt").write_text("[wall]\nbc_type wall\n")
    (case / "init.txt").write_text(
        "initial_type formula\nrho = 1 + .05*x*x + .12*y*y + .04*x\n"
        "ux = -.45\nuy = .02\np = 1 + .025*x*x\n"
    )
    names = ["rho", "ux", "uy", "p", "E"]
    (case / "output.txt").write_text(
        "[history]\nfile = state.fields\nevery_steps = 1\nmulti = 1\n"
        "format = text\nvariables = " + ", ".join(names) + "\n"
    )
    options = settings(case / "main.txt")
    options.update(mesh="grid.mesh", reconstruction=reconstruction, final_time=0.003, report_every=1)
    write_settings(case / "main.txt", options)
    ref = case / "reference"
    serial = run([aero, f"--input={case}", f"--output={ref}"])
    run([meshdec, mesh, case / "grid.mdec", "--target-ranks=7"])
    mesh.unlink()
    mesh_setting(case, "grid.mdec")
    out = case / "out"
    check_report(serial, launch(aero, case, out, launcher, 7))
    error = check_output(ref, out, 7, 7, 2, names)
    print(f"OK concave star on seven ranks max field difference={error:.3g}")


def verify_unmatched_faces(work, aero, meshdec, launcher):
    """Проверяет отказ при несовпадающих гранях периодической пары."""
    # Восемь углов совпадают после сдвига. На двух периодических сторонах выбраны разные диагонали,
    # поэтому попарного соответствия треугольников нет.
    case = work / "unmatched-faces"
    shutil.copytree(ROOT / "qa/cases/wave_3d", case)
    native = case / "grid.mesh"
    native.write_text("""AEROMESH 2 TEXT
DIMENSION 3
COUNTS 8 8 28 1 8 8 6
LENGTH_SCALE_TO_METERS 1
X 0 1 1 0 0 1 1 0
Y 0 0 1 1 0 0 1 1
Z 0 0 0 0 1 1 1 1
FACE_OFFSETS 0 3 6 9 12 16 20 24 28
FACE_NODES 0 4 7 0 7 3 1 2 5 2 6 5 0 1 5 4 3 7 6 2 0 3 2 1 4 5 6 7
CELL_OFFSETS 0 8
CELL_FACES 0 1 2 3 4 5 6 7
CELL_ORIENTATIONS 1 1 1 1 1 1 1 1
BOUNDARY_FACES 0 1 2 3 4 5 6 7
BOUNDARY_PATCH 0 0 1 1 2 3 4 5
PATCH_NAMES "xmin" "xmax" "ymin" "ymax" "zmin" "zmax"
END
""")
    options = settings(case / "main.txt")
    options.update(reconstruction="first_order", backend="cpu")
    write_settings(case / "main.txt", options)
    command = [aero, f"--input={case}", f"--output={case/'out'}"]
    run(command + ["--geometry-only"])
    if "unmatched face" not in run(command, success=False):
        raise AssertionError("Invalid nonconforming serial fixture")
    run([meshdec, native, case / "grid.mdec", "--target-ranks=4"])
    native.unlink()
    mesh_setting(case, "grid.mdec")
    if "unmatched face" not in run([launcher, "-n", 4, *command], success=False):
        raise AssertionError("MPI failed to diagnose nonconforming periodic faces")
    print("OK collective rejection of different periodic face subdivisions")


# Разбирает параметры запуска, выполняет набор QA-проверок и возвращает статус результата.
def main():
    parser = mpi_parser(__doc__)
    group = parser.add_mutually_exclusive_group()
    for order in ("ebr3", "ebr5", "weno3", "weno5", "mp5", "MPWENO5"):
        group.add_argument("--" + order.lower(), dest="reconstruction", action="store_const", const=order)
    args = parser.parse_args()
    programs = args.aero.resolve(), args.meshdec.resolve(), args.mpiexec
    with mpi_workspace("mpi-periodic-") as directory:
        work = Path(directory)
        if args.reconstruction:
            order = {"reconstruction": args.reconstruction}
            verify(work, "line", "pulse_1d", 1, {0}, *programs, **order)
            verify(work, "plane", "shear_jump_2d", 2, {0, 1}, *programs, **order)
            if args.reconstruction != "ebr3":
                verify(work, "narrow", "pulse_1d", 1, {0}, *programs, narrow=True, **order)
            if args.reconstruction in ("weno3", "weno5", "mp5", "MPWENO5"):
                verify_layer_face(work, *programs, args.reconstruction)
                for scheme in ("euler", "ssprk3"):
                    verify_euler(
                        work,
                        "sod-cut-" + scheme,
                        "sod_1d",
                        1,
                        0.002,
                        "characteristic",
                        "roe",
                        *programs,
                        scheme,
                        cut=True,
                        **order,
                    )
            shared = work / "plane/grid.mdec"
            verify(work, "mixed", "shear_jump_2d", 2, {0}, *programs, archive=shared, **order)
            verify(work, "physical", "shear_jump_2d", 2, set(), *programs, archive=shared, **order)
            verify(work, "cube", "wave_3d", 3, {0, 1, 2}, *programs, **order)
            verify(
                work, "one-leaf", "shear_jump_2d", 2, {0, 1}, *programs, tiny=True, empty=True, **order
            )
            return
        verify(work, "line", "pulse_1d", 1, {0}, *programs)
        verify(work, "plane", "shear_jump_2d", 2, {0, 1}, *programs)
        verify(work, "mixed", "shear_jump_2d", 2, {0}, *programs)
        verify(work, "cube", "wave_3d", 3, {0, 1, 2}, *programs)
        verify(work, "loops", "shear_jump_2d", 2, {0, 1}, *programs, tiny=True)
        verify(work, "one-leaf", "shear_jump_2d", 2, {0, 1}, *programs, tiny=True, empty=True)
        verify_unmatched_faces(work, *programs)


if __name__ == "__main__":
    main()
