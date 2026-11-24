"""Сверяет каждый принятый шаг и поля MPI-расчёта с результатом на целой сетке."""

from mpi_common import report_rows, ROOT, mpi_parser, mpi_workspace
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
    rows = re.compile(r"(?m)^(?:STEP |t = |Energy |Stabilization |(?:rho|u[xyz]|p|E)'?\s|Completed).*")
    ref, result = [
        rows.findall(log.partition("Starting time integration")[2]) for log in (reference, actual)
    ]
    if not ref or not ref[0].startswith("STEP ") or len(ref) != len(result):
        raise AssertionError("Missing global report or different number of steps")
    for a, b in zip(ref, result):
        if a.startswith('STEP '):  # Время компьютера может различаться.
            a, b = a[:53], b[:53]
        if a == b:
            continue
        numbers = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
        # Ширина заполнителя таблицы меняется при переходе от экспоненциальной
        # записи к обычной. Сверяем текстовые столбцы и сами числа отдельно.
        if re.sub(numbers, "#", a).split() != re.sub(numbers, "#", b).split() or not all(
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
    # В начальном разрыве скорость нулевая. Для каждого отрезка берём большую
    # из двух скоростей звука, в том числе на стыке левого и правого состояний.
    states = [(1.0, 1.0), (0.125, 0.1)]
    if reverse:
        states.reverse()
    sounds = [math.sqrt(1.4 * p / rho) for rho, p in states]
    expected_dt = min(
        .25 * (coords[i + 1][0] - coords[i][0]) / max(sounds[i > cut], sounds[i + 1 > cut])
        for i in range(len(coords) - 1)
    )
    if not close(dt, expected_dt):
        raise AssertionError(f"First CFL step: {dt} vs {expected_dt}")
    f = read_snapshot(path, len(coords), 1, dt, ("rho", "ux", "p", "E"))
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


def check_mp_alpha(aero, case, launcher, ranks, overrides):
    """Проверяет коэффициент, с которым солвер действительно построил MP-реконструкцию."""
    for reconstruction in ("mp5", "MPWENO5"):
        for time, cfl, alpha in (("euler", .5, 1), ("ssprk3", 2, 0), ("bdf1", .5, 4),
                                 ("bdf2", 2, 4), ("steady", .25, 4)):
            options = [f"main:reconstruction={reconstruction}", f"main:time_scheme={time}",
                       f"main:cfl={cfl}", "main:final_time=1e-5"]
            if time == "steady":
                options.extend(["implicit:steady_tol=0.9", "implicit:steady_max_steps=25"])
            target = case / f"alpha-{reconstruction}-{time}"
            log = launch(aero, case, target, launcher, ranks, *overrides, *options)
            actual = re.search(r"(?m)^MP alpha = (\S+)", log)
            if not actual or float(actual[1]) != alpha or "Completed " not in log:
                raise AssertionError(f"Wrong MP alpha for {reconstruction}, {time}, CFL={cfl}")
    print(f"OK MP5 and MPWENO5 alpha for five time schemes on {ranks} ranks")


def check_steady(aero, case, launcher, ranks, overrides, coords, names):
    """Сверяет оба критерия по начальному оттоку и по последнему полю MPI."""
    options = ("main:time_scheme=steady", "main:final_time=0", "implicit:simplified_newton=on", "implicit:jacobian_update_every=3")
    history = case / "output.txt"
    original = history.read_bytes()
    init = case / "init.txt"
    initial_settings = init.read_bytes()
    try:
        # Срок очередного снимка заведомо позже выхода на стационар.
        history.write_text("[state]\nfile = final.fields\nevery_steps = 1000\nmulti = 1\n"
                           "format = text\nvariables = " + ",".join(names) + "\n")
        absolute = "implicit:steady_criterion=absolute"
        start = case / "steady-initial"
        log = launch(aero, case, start, launcher, 1, *options, absolute, "implicit:steady_tol=1e20")
        initial_rate = float(re.search(r"absolute spatial residual = (\S+)", log)[1])
        if initial_rate <= 0 or stamp(next(start.glob("*.fields"))) != (0, 0.):
            raise AssertionError("Absolute criterion did not accept the initial field")
        # Один порог задаём в долях начального оттока, другой — в его единицах.
        # Обе проверки должны принять один и тот же шаг и сохранить одно поле.
        relative_output = case / "steady-relative"
        relative_log = launch(aero, case, relative_output, launcher, 1, *options,
                              "implicit:steady_tol=0.5", "implicit:steady_max_steps=100")
        target = case / "steady-absolute"
        abs_log = launch(aero, case, target, launcher, ranks, *overrides, *options, absolute,
                         "implicit:steady_tol=" + repr(initial_rate * .5), "implicit:steady_max_steps=100")
        def last(log, criterion):
            rows = list(report_rows(log, r"^Steady\s+" + criterion + r" residual = (\S+)"))
            # В таблице семь значащих цифр. Для строгой сверки критериев
            # берём невязку из итоговой строки с полной точностью вывода.
            final = re.search(r"(?m)^Steady state converged, " + criterion + r" spatial residual = (\S+)", log)
            return int(rows[-1][0]), float(final[1])
        accepted, ratio = last(relative_log, "relative")
        step, rate = last(abs_log, "absolute")
        if accepted != step or not 0. < ratio <= .5 or not math.isclose(ratio, rate/initial_rate, rel_tol=2e-8):
            raise AssertionError(f"Steady criteria disagree: relative {ratio} at step {accepted}, "
                                 f"absolute {rate} at step {step}, initial {initial_rate}")
        check_output(relative_output, target, ranks, len(coords), 1, names)
        # Те же физические потоки, но контрольные объёмы продвигаются с разными шагами.
        # Сравниваем одно и то же принятое состояние между CPU и разбиением.
        local_options = [*options, "implicit:local_time_step=on", "implicit:steady_tol=.5",
                         "implicit:steady_max_steps=100"]
        local_ref, local_mpi = case / "local-steady-reference", case / "local-steady-mpi"
        bounded = [*local_options, "implicit:local_time_step_max_ratio=4"]
        local_log = launch(aero, case, local_ref, launcher, 1, *bounded)
        other_log = launch(aero, case, local_mpi, launcher, ranks, *overrides, *bounded)
        for log in (local_log, other_log):
            ranges = list(report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)"))
            if not ranges or not all(0. < float(lo) < float(hi) <= 4.00001*float(lo) for _, lo, hi in ranges):
                raise AssertionError("Invalid bounded local time steps on a nonuniform mesh")
        if last(local_log, "relative")[0] != last(other_log, "relative")[0]:
            raise AssertionError("Decomposition changed the local steady convergence step")
        check_output(local_ref, local_mpi, ranks, len(coords), 1, names)
        # При коэффициенте 1 все разделы должны получить общий минимум всей сетки.
        # Сверяем поля с обычной схемой даже на узлах, удалённых от самого малого объёма.
        capped = case / "local-steady-uniform-step"
        capped_log = launch(aero, case, capped, launcher, ranks, *overrides, *local_options,
                            "implicit:local_time_step_max_ratio=1")
        if last(capped_log, "relative")[0] != accepted:
            raise AssertionError("Uniform cap changed convergence compared with a global step")
        check_output(relative_output, capped, ranks, len(coords), 1, names)
        limit = case / "steady-limit"
        failed = run([launcher, "-n", ranks, aero, f"--input={case}", f"--output={limit}", *overrides,
                      *options, "implicit:steady_tol=1e-12", "implicit:steady_max_steps=1"], success=False)
        if "pseudo-step limit reached" not in failed or not list(limit.rglob("*.fields")):
            raise AssertionError("Steady must save the last field and report nonconvergence")
        # У однородного газа баланс точный. Относительный критерий должен
        # сохранить начальное поле и завершиться, не деля ноль на ноль.
        init.write_text("initial_type formula\nrho = 1\nux = 0\np = 1\n")
        uniform = case / "steady-uniform"
        equilibrium = launch(aero, case, uniform, launcher, ranks, *overrides, *options)
        if "relative spatial residual = 0" not in equilibrium or "Completed 0 steps" not in equilibrium:
            raise AssertionError("Uniform steady field should require zero pseudo-steps")
        print(f"OK steady: CPU relative and {ranks}-rank absolute, step {step}, residual ratio {ratio:.5g}")
    finally:
        init.write_bytes(initial_settings)
        history.write_bytes(original)


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
    simplified=False,
    devices=(),
    matrix_test=None,
):
    """Сверяет каждый принятый шаг и поля MPI-расчёта с результатом на целой сетке."""
    case = work / title
    shutil.copytree(ROOT / "qa/cases" / fixture, case)
    if scheme in ("bdf1","bdf2"):
        config = (ROOT / "templates/implicit.txt").read_text(encoding="utf-8")
        if scheme == "bdf2":
            config = config.replace("jacobian_flux = LF", "jacobian_flux = Roe")
            config = config.replace("preconditioner = BJ", "preconditioner = CSGS")
            config = config.replace("solver_tol = 1e-8", "solver_tol = 1e-12")
        if simplified:
            config = config.replace("simplified_jacobian = off", "simplified_jacobian = on")
        (case / "implicit.txt").write_text(config, encoding="utf-8")
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
        report_every=20 if title == "sod" else 1,
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
    if matrix_test:
        result = run([launcher, "-n", len(devices) or 2, matrix_test, native, archive, *map(str, devices)])
        print(result.strip())
    if cut:
        cut_node = common_cut(case, case, aero, launcher, coords)
        location = (coords[cut_node][0] + coords[cut_node + 1][0]) / 2
        init = case / "init.txt"
        init.write_text(
            init.read_text().replace("x < 0.5", ("x > " if reverse else "x < ") + repr(location))
        )
    mesh_setting(case, "grid.mesh")
    reference = case / "native"
    log = run([aero, f"--input={case}", f"--output={reference}"])
    if cut:
        if reconstruction == "first_order" and scheme not in ("bdf1","bdf2"):
            cut_balance(reference / "state_000001.fields", coords, cut_node, reverse, scheme)
        times = [stamp(path)[1] for path in sorted(reference.glob("timed_*.fields"))]
        if times != [final / 2, final]:
            raise AssertionError(f"Missed physical-time outputs: {times}")
    mesh_setting(case, "grid.mdec")
    native.unlink()
    ranks_to_test = (4,) if empty else (1, 2, 4)
    if devices:
        ranks_to_test = (len(devices),)
        (case / "cuda.txt").write_text("devices=" + ",".join(map(str, devices)))
    for ranks in ranks_to_test:
        target = case / f"out-{ranks}"
        overrides = ("main:backend=cuda",) if devices else ()
        actual = launch(aero, case, target, launcher, ranks, *overrides)
        if scheme in ("bdf1","bdf2"):
            pattern = r"^Newton\s+\d+/\d+ iterations\s+relative residual = (\S+)\s+\[OK\]$"
            expected, observed = [list(report_rows(text, pattern)) for text in (log, actual)]
            if not expected or [s for s, _ in expected] != [s for s, _ in observed]:
                raise AssertionError("Missing Newton residual for an accepted step")
            for (step, reference_norm), (_, norm) in zip(expected, observed):
                if not (0 <= float(norm) <= 1e-6 and
                        math.isclose(float(norm), float(reference_norm), rel_tol=1e-4, abs_tol=1e-11)):
                    raise AssertionError(f"Newton step {step}: {reference_norm} vs {norm}")
        check_report(log, actual)
        error = check_output(reference, target, ranks, len(coords), dim, names, empty)
        print(
            f"OK {scheme} {title}  ranks={ranks}  max field difference={error:.3g}  {actual.splitlines()[-1]}"
        )
    if cut and scheme == "bdf1" and not simplified:
        check_steady(aero, case, launcher, ranks_to_test[-1], overrides, coords, names)
        check_mp_alpha(aero, case, launcher, ranks_to_test[-1], overrides)
    if cut and scheme == "bdf2" and not simplified:
        # Останавливаемся на запланированном сроке вывода, поэтому последовательность dt
        # совпадает с непрерывным прогоном. Меняем число процессов между записью и чтением.
        restart = case / "restart"
        launch(aero, case, restart, launcher, 1, f"main:final_time={final/2}", "main:recovery_every_steps=1")
        snapshots = sorted(restart.glob("state_*.fields"))
        last, split = stamp(snapshots[-1])
        if split != final/2 or last < 2:
            raise AssertionError("Expected at least two accepted steps before restart")
        resumed = launch(aero, case, restart, launcher, ranks_to_test[-1], "--recover", *overrides)
        if f"Recovered step {last} at" not in resumed:
            raise AssertionError(resumed)
        expected_tail = case / "expected-tail"
        expected_tail.mkdir()
        for path in reference.glob("*.fields"):
            if stamp(path)[0] > last:
                shutil.copyfile(path, expected_tail / path.name)
        error = check_output(expected_tail, restart / f"resumed-{last}-1", ranks_to_test[-1], len(coords), dim, names)
        print(f"OK BDF2 checkpoint 1 -> {ranks_to_test[-1]} ranks, max field difference={error:.3g}")

        # У контрольной точки версии 1 есть только U_n. Первый шаг продолжения BDF2
        # должен совпасть с одним шагом неявного Эйлера из той же контрольной точки.
        legacy = case / "legacy-bdf1"
        launch(aero, case, legacy, launcher, 1, "main:time_scheme=bdf1", f"main:final_time={split}",
               "main:recovery_every_steps=1")
        migrated = case / "legacy-to-bdf2"
        migrated.mkdir()
        for checkpoint in legacy.glob("recovery-*.bin"):
            shutil.copyfile(checkpoint, migrated / checkpoint.name)
        first_step = f"main:final_time={split+1e-5}"
        launch(aero, case, legacy, launcher, 1, "--recover", first_step, "main:time_scheme=bdf1")
        launch(aero, case, migrated, launcher, ranks_to_test[-1], "--recover", first_step, *overrides)
        before = next(legacy.glob("resumed-*"))
        after = next(migrated.glob("resumed-*"))
        error = check_output(before, after, ranks_to_test[-1], len(coords), dim, names)
        print(f"OK BDF2 startup from version 1, max field difference={error:.3g}")
    if cut and not reverse and scheme == "ssprk3" and reconstruction == "first_order":
        config = case / "main.txt"
        config.write_text(config.read_text().replace("report_every = 1", "report_every = 0"))
        quiet = case / "no-reports"
        actual = launch(aero, case, quiet, launcher, 4)
        if re.search(r"(?m)^STEP ", actual):
            raise AssertionError("Unexpected report")
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
        # После первой стадии именно левый узел разрыва теряет допустимое
        # состояние. Номер берём из фактического разбиения этой сетки.
        bad_node = rf"(?m)^aero: rank \d+: Invalid Euler stage state at global node {cut_node}$"
        if not re.search(bad_node, log) or any(failed.rglob("*.fields")):
            raise AssertionError(
                f"Expected a collective intermediate-stage failure before output:\n{log}"
            )
        print("OK SSPRK3 without reports and collective rejection of a failed intermediate stage")
    if cut and scheme in ("bdf1","bdf2") and not simplified:
        limited = case / "newton-limit"
        ranks = ranks_to_test[-1]
        log = run([launcher, "-n", ranks, aero, f"--input={case}", f"--output={limited}",
                   "implicit:newton_max_iter=1", "implicit:newton_tol=1e-14", *overrides])
        residuals = re.findall(r"(?m)^Newton\s+1/1 iterations\s+relative residual = (\S+)\s+\[ITERATION LIMIT\]$", log)
        times = sorted(stamp(path)[1] for path in limited.rglob("timed_*.fields"))
        if not any(float(value) > 1.e-14 for value in residuals) or times != [final / 2] * ranks + [final] * ranks:
            raise AssertionError(f"{scheme} did not publish the final field after limited Newton steps:\n{log}")
    if empty:
        command = [launcher, "-n", 4, aero, f"--input={case}", f"--output={case / 'missing-init'}"]
        (case / "init.txt").write_text("initial_type file\npath missing.fields\n")
        rejected = run(command, success=False)
        if "Cannot open fields for reading" not in rejected:
            raise AssertionError(rejected)


# Разбирает параметры запуска, выполняет набор QA-проверок и возвращает статус результата.
def main():
    parser = mpi_parser(__doc__)
    parser.add_argument("--time-scheme", choices=("euler", "ssprk3", "bdf1", "bdf2"), default="euler")
    parser.add_argument("--matrix-test", type=Path, help="Executable for row-wise A*x comparison")
    parser.add_argument("--devices", help="Compare a CPU reference with these MPI GPUs, for example 0,1")
    args = parser.parse_args()
    programs = (args.aero.resolve(), args.meshdec.resolve(), args.mpiexec, args.time_scheme)
    with mpi_workspace(f"mpi-{args.time_scheme}-") as work:
        root = Path(work)
        if args.time_scheme in ("bdf1","bdf2"):
            devices = tuple(map(int, args.devices.split(","))) if args.devices else ()
            for simplified in (False, True):
                label = "frozen" if simplified else "full"
                verify(root, f"sod-cut-{label}", "sod_1d", 1, .004, "characteristic", "lf", *programs,
                       cut=True, simplified=simplified, devices=devices,
                       matrix_test=args.matrix_test.resolve() if args.matrix_test and not simplified else None)
            verify(root, "weno-roe", "sod_1d", 1, .005, "characteristic", "roe", *programs,
                   reconstruction="weno3", devices=devices)
            verify(root, "tilted", "shear_jump_2d", 2, .003, "characteristic", "lf", *programs,
                   simplified=True, devices=devices)
            verify(root, "wave", "wave_3d", 3, .006, "flux", "roe", *programs, devices=devices)
            verify(root, "one-leaf", "sod_1d", 1, .002, "characteristic", "lf", *programs,
                   empty=True, devices=devices)
            return
        verify(root, "sod-cut", "sod_1d", 1, 0.004, "characteristic", "roe", *programs, cut=True)
        verify(
            root, "sod-cut-reversed", "sod_1d", 1, 0.004, "flux", "lf", *programs, cut=True, reverse=True
        )
        verify(root, "sod", "sod_1d", 1, 0.12, "characteristic", "roe", *programs)
        verify(root, "tilted", "shear_jump_2d", 2, 0.015, "characteristic", "lf", *programs)
        verify(root, "wave", "wave_3d", 3, 0.018, "flux", "roe", *programs)
        verify(root, "one-leaf", "sod_1d", 1, 0.002, "characteristic", "lf", *programs, empty=True)


if __name__ == "__main__":
    main()
