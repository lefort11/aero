"""Сверяет каждый принятый шаг и поля MPI-расчёта с результатом на целой сетке."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import math
from pathlib import Path
import re
import shutil

from mpi_geometry import run
from run import nodes, read_snapshot, settings


def free_patch(name):
    """Задаёт открытую границу с тремя режимами и текущим внутренним состоянием в качестве источника."""
    return f"[{name}]\nbc_type free\nsuperin\ninflow base_p\noutflow base_u\n\n"


def launch(aero, case, output, launcher, ranks, *extra):
    """Запускает солвер на указанном числе MPI-процессов с отдельными каталогами входа и результата."""
    return run([launcher, "-n", ranks, aero, f"--input={case}", f"--output={output}", *extra])


def mesh_setting(case, mesh):
    """Меняет путь к сетке в локальной копии постановки."""
    path = case / "main.txt"
    path.write_text(re.sub(r"(?m)^mesh = .*", f"mesh = {mesh}", path.read_text()))


def common_cut(case, work, aero, launcher, coords):
    """Находит границу, общую для двух разбиений и ближайшую к середине линии."""
    cuts = []
    for ranks in (2, 4):
        out = work / f"geometry-{ranks}"
        launch(aero, case, out, launcher, ranks, "--geometry-only")
        owners = {}
        for rank in range(ranks):
            rows = (out / f"geometry-{rank}-of-{ranks}.tsv").read_text().splitlines()
            for row in rows:
                if row.startswith("V\t"):
                    owners[int(row.split()[1])] = rank
        cuts.append({i for i in range(len(coords) - 1) if owners[i] != owners[i + 1]})
    shared = set.intersection(*cuts)
    return min(shared, key=lambda i: abs(coords[i][0] - 0.5))


def stamp(path):
    """Извлекает номер принятого шага и его физическое время из снимка."""
    text = path.read_text(encoding="ascii")
    step = int(re.search(r"(?m)^STEP\s+(\d+)", text)[1])
    time = float(re.search(r"(?m)^TIME\s+(\S+)", text)[1])
    return step, time


def close(a, b):
    """Сравнивает числа с относительным и абсолютным допусками MPI-проверок."""
    return math.isclose(a, b, rel_tol=3e-13, abs_tol=3e-14)


def check_report(reference, actual):
    """Сверяет строки отчёта об интегрировании с результатом на целой сетке."""
    rows = re.compile(r"(?m)^(?:Flow report|Energy residual|rho\s|u[xyz]\s|p\s|E\s|Completed).*")
    ref, result = [
        rows.findall(log.partition("Starting time integration")[2]) for log in (reference, actual)
    ]
    if not ref or not ref[0].startswith("Flow report") or len(ref) != len(result):
        raise AssertionError("Missing global report or different number of steps")
    for a, b in zip(ref, result):
        if a == b:
            continue
        numbers = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
        if re.sub(numbers, "#", a) != re.sub(numbers, "#", b) or not all(
            close(float(x), float(y)) for x, y in zip(re.findall(numbers, a), re.findall(numbers, b))
        ):
            raise AssertionError(f"Global report mismatch:\n{a}\n{b}")


def check_output(reference, out, ranks, count, dim, names, empty=False):
    """Сопоставляет локальные номера с исходными и сравнивает поля каждого узла во все моменты вывода."""
    paths = sorted(reference.glob("*.fields"))
    if not paths:
        raise AssertionError("No reference snapshots")
    owners = {}
    copies = []
    for rank in range(ranks):
        if ranks == 1:
            directory, mapping = out, list(range(count))
        else:
            part = out / f"rank-{rank}-of-{ranks}"
            if not (part / "grid.mesh").is_file():
                raise AssertionError("Missing local mesh")
            directory, mapping = part / "fields", []
            rows = (part / "nodes.tsv").read_text().splitlines()[1:]
            for row in rows:
                local, node, owner = map(int, row.split())
                if local != len(mapping) or not 0 <= node < count or not 0 <= owner < ranks:
                    raise AssertionError(f"Bad node map: {row}")
                if rank == owner:
                    if node in owners:
                        raise AssertionError(f"Duplicate owner for node {node}")
                    owners[node] = rank
                mapping.append(node)
        if {p.name for p in directory.glob("*.fields")} != {p.name for p in paths}:
            raise AssertionError(f"Output schedule mismatch on rank {rank}")
        copies.append((directory, mapping))
    if ranks > 1 and (len(owners) != count or len(set(owners.values())) != (1 if empty else ranks)):
        raise AssertionError("Missing owner or incorrect number of active partitions")
    max_error = 0.0
    for path in paths:
        step, time = stamp(path)
        expected = read_snapshot(path, count, dim, time, names)
        for directory, mapping in copies:
            snap = directory / path.name
            s, t = stamp(snap)
            if s != step or not math.isclose(t, time, rel_tol=3e-14):
                raise AssertionError(f"Step/time mismatch: {(s,t)} vs {(step,time)}")
            actual = read_snapshot(snap, len(mapping), dim, time, names)
            for key in names:
                for i, global_id in enumerate(mapping):
                    a, b = expected[key][1][global_id], actual[key][1][i]
                    max_error = max(max_error, abs(a - b))
                    if not close(a, b):
                        raise AssertionError(f"step={s}, node={global_id}, {key}: {a} vs {b}")
    return max_error


def cut_balance(path, coords, cut, reverse, scheme):
    """Проверяет баланс переноса в окрестности разреза между разделами."""
    dt = stamp(path)[1]
    left, right = (len(coords) - 2, len(coords) - 1) if reverse else (0, 1)
    expected_dt = 0.25 * (coords[right][0] - coords[left][0]) / (2 * math.sqrt(1.4))
    if not close(dt, expected_dt):
        raise AssertionError(f"First CFL step: {dt} vs {expected_dt}")
    f = read_snapshot(path, len(coords), 1, dt, ("rho", "ux", "p", "E"))
    states = [(1.0, 1.0), (0.125, 0.1)]
    if reverse:
        states.reverse()
    # При потоке первого порядка на каждой стадии влияние распространяется ещё на один узел в обе стороны.
    depth = 3 if scheme == "ssprk3" else 1
    rates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(cut - depth + 1, cut + depth + 1):
        side = int(i > cut)
        rho, p = states[side]
        volume = (coords[i + 1][0] - coords[i - 1][0]) / 2
        density, velocity, energy = [f[key][1][i] for key in ("rho", "ux", "E")]
        for k, change in enumerate((density - rho, density * velocity, energy - p / (1.4 - 1))):
            rates[side][k] += change * volume / dt
        if scheme == "ssprk3" and i in (cut - 2, cut + 3) and close(density, rho):
            raise AssertionError(f"Third stage did not propagate the discontinuity to node {i}")
    pressure_jump = states[0][1] - states[1][1]
    if (
        rates[0][0] * pressure_jump >= 0
        or rates[1][0] * pressure_jump <= 0
        or not all(close(a + b, ref) for (a, b), ref in zip(zip(*rates), (0.0, pressure_jump, 0.0)))
    ):
        raise AssertionError(f"First-step conservation failure across Sod interface: {rates}")


def verify(
    work,
    title,
    fixture,
    dim,
    final,
    variables,
    flux,
    aero,
    meshdec,
    launcher,
    scheme,
    cut=False,
    reverse=False,
    empty=False,
    reconstruction="first_order",
):
    """Сверяет каждый принятый шаг и поля MPI-расчёта с результатом на целой сетке."""
    case = work / title
    shutil.copytree(ROOT / "qa/cases" / fixture, case)
    native = case / "grid.mesh"
    if dim == 2:
        words = native.read_text().split()
        xyz = [words.index(axis) + 1 for axis in "XYZ"]
        count = int(words[words.index("COUNTS") + 1])
        for i in range(count):
            x, y = (float(words[j + i]) for j in xyz[:2])
            words[xyz[2] + i] = repr(0.25 * x + 0.5 * y)
        native.write_text(" ".join(words))
    coords = nodes(native, dim)
    names = ["rho", *("u" + c for c in "xyz"[:dim]), "p", "E"]
    options = settings(case / "main.txt")
    options.update(
        mesh="grid.mdec",
        reconstruction=reconstruction,
        reconstruction_variables=variables,
        euler_solver=flux,
        time_scheme=scheme,
        final_time=str(final),
        flowreport_every=20 if title == "sod" else 1,
        omp_threads=2 if dim > 1 else 1,
    )
    (case / "main.txt").write_text("".join(f"{key} = {v}\n" for key, v in options.items()))
    if dim > 1:
        (case / "bc.txt").write_text(
            "".join(free_patch(a + side) for a in "xyz"[:dim] for side in ("min", "max"))
        )
    schedule = f"every_time = {final}\nmulti = 0" if title == "sod" else "every_steps = 1\nmulti = 1"
    output = (
        f"[history]\nfile = state.fields\n{schedule}\nformat = text\nvariables = "
        + ", ".join(names)
        + "\n"
    )
    if cut:
        output += (
            f"\n[timed]\nfile = timed.fields\nevery_time = {final/2}\nmulti = 1\n"
            "format = text\nvariables = " + ", ".join(names) + "\n"
        )
    (case / "output.txt").write_text(output)
    archive = case / "grid.mdec"
    run([meshdec, native, archive, "--compress", *([] if empty else ["--target-ranks=4"])])
    if cut:
        i = common_cut(case, case, aero, launcher, coords)
        location = (coords[i][0] + coords[i + 1][0]) / 2
        init = case / "init.txt"
        init.write_text(
            init.read_text().replace("x < 0.5", ("x > " if reverse else "x < ") + repr(location))
        )
    mesh_setting(case, "grid.mesh")
    reference = case / "native"
    log = run([aero, f"--input={case}", f"--output={reference}"])
    if cut:
        if reconstruction == "first_order":
            cut_balance(reference / "state_000001.fields", coords, i, reverse, scheme)
        times = [stamp(path)[1] for path in sorted(reference.glob("timed_*.fields"))]
        if times != [final / 2, final]:
            raise AssertionError(f"Missed physical-time outputs: {times}")
    mesh_setting(case, "grid.mdec")
    native.unlink()
    for ranks in ((4,) if empty else (1, 2, 4)):
        target = case / f"out-{ranks}"
        actual = launch(aero, case, target, launcher, ranks)
        check_report(log, actual)
        error = check_output(reference, target, ranks, len(coords), dim, names, empty)
        print(
            f"OK {scheme} {title}  ranks={ranks}  max field difference={error:.3g}  {actual.splitlines()[-1]}"
        )
    if cut and not reverse and scheme == "ssprk3" and reconstruction == "first_order":
        config = case / "main.txt"
        config.write_text(config.read_text().replace("flowreport_every = 1", "flowreport_every = 0"))
        quiet = case / "no-reports"
        actual = launch(aero, case, quiet, launcher, 4)
        if "Flow report" in actual:
            raise AssertionError("Unexpected flow report")
        check_output(reference, quiet, 4, len(coords), dim, names)

        # При этом CFL первая стадия Эйлера на одной стороне разреза даёт нефизическое состояние.
        # Все процессы должны завершиться до публикации первого шага.
        config.write_text(
            re.sub(r"(?m)^cfl = .*", "cfl = 100", config.read_text()).replace(
                f"final_time = {final}", "final_time = 0.12"
            )
        )
        (case / "output.txt").write_text(
            "[history]\nfile = state.fields\nevery_steps = 1\nmulti = 1\n"
            "format = text\nvariables = rho\n"
        )
        failed = case / "failed-stage"
        log = run([launcher, "-n", 4, aero, f"--input={case}", f"--output={failed}"], success=False)
        if "Invalid flux at global node" not in log or any(failed.rglob("*.fields")):
            raise AssertionError(
                f"Expected a collective intermediate-stage failure before output:\n{log}"
            )
        print("OK SSPRK3 without reports and collective rejection of a failed intermediate stage")
    if empty:
        command = [launcher, "-n", 4, aero, f"--input={case}", f"--output={target}"]
        (case / "init.txt").write_text("initial_type file\npath missing.fields\n")
        rejected = run(command, success=False)
        if "Cannot open fields for reading" not in rejected:
            raise AssertionError(rejected)


def main():
    parser = mpi_parser(__doc__)
    parser.add_argument("--time-scheme", choices=("euler", "ssprk3"), default="euler")
    args = parser.parse_args()
    programs = (args.aero.resolve(), args.meshdec.resolve(), args.mpiexec, args.time_scheme)
    with mpi_workspace(f"mpi-{args.time_scheme}-") as work:
        root = Path(work)
        verify(root, "sod-cut", "sod_1d", 1, 0.002, "conservative", "roe", *programs, cut=True)
        verify(
            root, "sod-cut-reversed", "sod_1d", 1, 0.002, "flux", "lf", *programs, cut=True, reverse=True
        )
        verify(root, "sod", "sod_1d", 1, 0.12, "conservative", "roe", *programs)
        verify(root, "tilted", "shear_jump_2d", 2, 0.015, "characteristic", "lf", *programs)
        verify(root, "wave", "wave_3d", 3, 0.018, "flux", "roe", *programs)
        verify(root, "one-leaf", "sod_1d", 1, 0.002, "conservative", "lf", *programs, empty=True)


if __name__ == "__main__":
    main()
