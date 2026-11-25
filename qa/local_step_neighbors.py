"""Проверяет ограничение отношения локальных псевдошагов на соседних узлах."""

import math
import shutil
from pathlib import Path

from mpi_common import ROOT, mpi_parser, mpi_workspace, report_rows
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from viscosity import max_difference


def prepare(case):
    """Первый отрезок линии в 256 раз короче остальных. Скорость звука всюду одинакова."""
    short = 1 / 2048
    positions = [0.] + [short + i / 8 for i in range(8)]
    sparse_line(case / "grid.mesh", len(positions), positions)
    (case / "main.txt").write_text(
        "mesh=grid.mesh\nbackend=cpu\nomp_threads=2\ntime_scheme=steady\n"
        "cfl_method=height\ncfl=.05\nreconstruction=first_order\nreconstruction_variables=characteristic\n"
        "euler_solver=lf\nreport_every=1\n"
    )
    (case / "gas.txt").write_text("gamma=1.4\n")
    # Градиент давления запускает движение. При постоянном p/rho первый шаг
    # определяется только длиной меньшего из отрезков, прилегающих к узлу.
    (case / "init.txt").write_text("initial_type=formula\nrho=1+.2*x\np=1+.2*x\nux=0\n")
    (case / "bc.txt").write_text("[xmin, xmax]\nbc_type novisc_wall\n")
    (case / "output.txt").write_text(
        "[final]\nfile=final.fields\nvariables=rho,ux,p\nevery_steps=3\nformat=binary\nlayout=global\n"
    )
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")


def solve(case, args, label, *, limit=None, global_limit=None, mpi=False, gpu=False, residual_limit=False):
    steps = 68 if residual_limit else 3
    command = [args.mpiexec, "-n", "2"] if mpi else []
    command += [args.aero, f"--input={case}", f"--output={case / label}",
                "implicit:local_time_step=auto",
                f"implicit:steady_max_steps={steps}", "implicit:steady_tol=1e-14"]
    if residual_limit:
        command += ["implicit:local_time_step_residual_limit=on", "implicit:local_time_step_probe_node=7"]
    if global_limit is not None:
        command.append(f"implicit:local_time_step_max_ratio={global_limit}")
    if limit is not None:
        command.append(f"implicit:local_time_step_neighbor_ratio={limit}")
    if mpi:
        command.append("main:mesh=grid.mdec")
    if gpu:
        command.append("main:backend=cuda")
    log = run(command, success=False)
    if "Steady: pseudo-step limit reached before convergence" not in log:
        raise AssertionError(log)

    # На первом шаге допустимые множители по пути равны 1,1,2,4,...,128.
    # При одном общем потолке 2 вместо 128 получилось бы 2, а при одном
    # обходе соседей осталось бы 256. Передача через границу MPI тоже обязательна.
    ranges = list(report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)"))
    if [int(step) for step, _, _ in ranges] != list(range(1, steps+1)):
        raise AssertionError(log)
    _, smallest, largest = ranges[0]
    measured = float(largest) / float(smallest)
    before = min(256, global_limit or 256)
    # К самому дальнему узлу от ближайшего малого шага ведут семь рёбер.
    expected = min(before, limit ** 7) if limit else before
    if not math.isclose(measured, expected, rel_tol=2.e-6):
        raise AssertionError(f"{label}: неверный разброс шагов {measured:g}")
    if limit:
        rows = list(report_rows(log, r"^LTS edge ratio\s+(\S+)\s+->\s+(\S+)\s+limit = (\S+)"))
        if ([int(step) for step, *_ in rows] != list(range(1, steps+1))
                or not math.isclose(float(rows[0][1]), before, rel_tol=2.e-6)
                or not math.isclose(float(rows[0][2]), min(before, limit), rel_tol=2.e-6)
                or any(not 1 <= float(after) <= limit * (1 + 1.e-6) or float(setting) != limit
                       for _, _, after, setting in rows)):
            raise AssertionError(log)
    if residual_limit:
        # До первой полной серии из 32 принятых шагов история только накапливается.
        # На этой неоднородной линии затем растёт невязка лишь в части узлов.
        counts = [int(n) for _, n in report_rows(log, r"^LTS residual limit  nodes = (\d+)")]
        if len(counts) != steps or any(counts[:32]) or not 0 < max(counts[32:]) < 9:
            raise AssertionError(f"{label}: неверная активация узлового ограничения {counts}")
        probe = list(report_rows(log, r"^LTS probe +node = 7 +dt = (\S+) +gas rate = (\S+) +SA rate = (\S+) +cap/CFL step = (\S+)"))
        if len(probe) != steps:
            raise AssertionError(f"{label}: неполная история выбранного узла")
        for (_, dt, gas, sa, cap), (_, minimum, maximum) in zip(probe, ranges):
            if not (float(minimum) <= float(dt) <= float(maximum) and float(gas) > 0
                    and float(sa) == 0 and 0 < float(cap) <= 1):
                raise AssertionError(f"{label}: неверный шаг или невязка узла 7")
    metadata, fields = binary_snapshot(case / label / "final.fields")
    if metadata[3] != steps or set(fields) != {"rho", "ux", "p"}:
        raise AssertionError(f"{label}: неполный снимок последнего шага")
    return fields


def check_stencil(case, args):
    """Дальние узлы шаблона WENO5 лежат по обе стороны раздела MPI."""
    shutil.copyfile(ROOT / "templates/scheme.txt", case / "scheme.txt")
    baseline = {}
    for label, mpi, gpu in [("cpu", False, False), ("mpi", True, False), ("gpu", False, True)]:
        if (mpi and not args.mpi) or (gpu and not args.gpu):
            continue
        out = case / ("stencil-" + label)
        command = ([args.mpiexec, "-n", "2"] if mpi else []) + [
            args.aero, f"--input={case}", f"--output={out}", "main:reconstruction=weno5",
            "implicit:local_time_step=auto", "implicit:local_time_step_residual_limit=stencil",
            "implicit:local_time_step_probe_node=2",
            "implicit:steady_max_steps=96", "implicit:steady_tol=1e-14",
            "main:mesh=grid.mdec" if mpi else "main:mesh=grid.mesh",
            "main:backend=cuda" if gpu else "main:backend=cpu"]
        log = run(command, success=False)
        if "pseudo-step limit reached before convergence" not in log:
            raise AssertionError(log)
        # Наблюдение начинается с одинаковых шагов. Следующий диапазон
        # вычисляется только после полного блока из 32 принятых шагов.
        ratios = reported_limits(log)
        assert set(ratios) == set(range(1, 97))
        assert all(ratios[k] == 1. for k in range(1, 33)), ratios
        assert all(ratios[k] >= 1. and (k % 32 == 1 or ratios[k] == ratios[k-1])
                   for k in ratios), ratios
        counts = [int(n) for _, n in report_rows(log, r"^LTS residual limit  nodes = (\d+)")]
        caps = [int(n) for _, n in report_rows(log, r"nodes with stored cap = (\d+)")]
        assert len(caps) == len(counts) and all(0 <= active <= cap <= 9 for active,cap in zip(counts,caps))
        assert len(counts) == 96 and not any(counts[:32]) and max(counts) > 0, counts
        # Номер относится к одному из узлов шаблона; при разбиении другой
        # сработавший узел может быть выбран представителем того же события.
        sources = list(report_rows(log, r"^LTS probe stencil  gas response at node = (\d+)"))
        reviews = [int(step) for step, node in sources]
        assert reviews and all(step % 32 == 0 for step in reviews), sources
        assert all(0 <= int(node) < 9 for _, node in sources), sources
        _, fields = binary_snapshot(out / "final.fields")
        if label == "cpu":
            assert any(int(node) != 2 for _, node in sources), sources
            baseline, expected, expected_caps, expected_reviews = fields, counts, caps, reviews
        else:
            delta = max_difference(baseline, fields)
            assert reviews == expected_reviews and counts == expected and caps == expected_caps and delta < 2.e-10, (label, delta, counts)
            print(f"PASS WENO5 stencil {label}: counts match, maximum field difference {delta:.3g}")


def reported_limits(log):
    """Сверяет фактический разброс dt с объявленным пределом на каждом принятом шаге."""
    limits = {int(k): float(v) for k, v in report_rows(log, r"^LTS step ratio limit = (\S+)")}
    ranges = list(report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)"))
    assert {int(k) for k, _, _ in ranges} == limits.keys()
    for k, small, large in ranges:
        assert float(large) <= float(small) * limits[int(k)] * (1. + 2e-6), (k, small, large)
    return limits


def check_cfl_after_limit(case, args, label="cpu", mpi=False, gpu=False):
    """При K=1 проверяет физический шаг, пробу узла и историю после смены CFL."""
    command = ([args.mpiexec, "-n", "2"] if mpi else []) + [
        args.aero, f"--input={case}", "implicit:local_time_step=auto",
        "implicit:steady_max_steps=4", "implicit:steady_tol=1e-14",
        "implicit:local_time_step_probe_node=7", "main:control_every=1",
        "main:mesh=grid.mdec" if mpi else "main:mesh=grid.mesh",
        "main:backend=cuda" if gpu else "main:backend=cpu"]
    for mode in ("work", "cfl"):
        out = case / f"scale-{mode}-{label}"
        out.mkdir()
        (out / "control.txt").write_text("cfl=0.25\n")
        log = run([*command, f"--output={out}", "main:cfl=100",
                   f"implicit:local_time_step_residual_limit={mode}"], success=False)
        assert "pseudo-step limit reached before convergence" in log, log
        rows = list(report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)\s+CFL = (\S+)"))
        assert [float(cfl) for _, _, _, cfl in rows] == [100., .25, .25, .25], log
        assert set(reported_limits(log).values()) == {1.}, log
        steps = [float(lo) for _, lo, _, _ in rows]
        assert steps == [float(hi) for _, _, hi, _ in rows]
        assert steps == [float(dt) for _, dt in report_rows(log, r"^LTS probe +node = 7 +dt = (\S+)")]
        assert math.isclose(steps[0], 100 / (2048 * math.sqrt(1.4)), rel_tol=2.e-6), steps
        _, fields = binary_snapshot(out / "final.fields")
        if mode == "work":
            reference_steps, reference_fields = steps, fields
        else:
            assert all(math.isclose(a, b, rel_tol=2.e-6) for a, b in zip(steps, reference_steps)), steps
            assert max_difference(reference_fields, fields) < 2.e-10
    print(f"PASS LTS cfl at CFL 100 and after switching to 0.25 ({label})")


def spread_record(directory, step):
    """Читает строку регулятора из контрольной точки указанного принятого шага."""
    for path in directory.glob("recovery-?.bin"):
        data = path.read_bytes()
        if int.from_bytes(data[32:40], "little") == step:
            # Восемь последних байт — контрольная сумма служебной секции.
            record = data[:-8].rsplit(b" step_spread ", 1)[-1].split()
            assert record[1] == b"work1", (step, record[:2])
            return record[0].decode("ascii"), list(map(float, record[2:10]))
    raise AssertionError(f"Нет контрольной точки шага {step} в {directory}")


def check_spread_restart(case, args, mode):
    """После LO начинаем историю EBR5 заново и продолжаем её с середины блока."""
    case.mkdir()
    prepare(case)
    points = [i/16 for i in range(17)]
    sparse_line(case / "grid.mesh", len(points), points)
    (case / "bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n")
    (case / "init.txt").write_text("initial_type=formula\nrho=1+.04*sin(4*pi*x)\nux=1\np=1\n")
    shutil.copyfile(ROOT / "templates/scheme.txt", case / "scheme.txt")
    if args.mpi:
        run([args.meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2"])
    if args.gpu:
        (case / "cuda.txt").write_text(f"devices={args.device}\n")
    # После 40 шагов LO регулятор уже вышел из первого блока. В точке 91
    # у EBR5 завершён один блок и накоплено 19 наблюдений следующего.
    switch, split, end = 40, 91, 106
    for label in ["cpu"] + (["mpi"] if args.mpi else []) + (["gpu"] if args.gpu else []):
        parallel = label == "mpi"
        solver = [args.mpiexec, "-n", "2", args.aero] if parallel else [args.aero]
        common = [*solver, f"--input={case}", "main:cfl=1", "implicit:local_time_step=auto",
                  f"implicit:local_time_step_residual_limit={mode}", "implicit:steady_tol=1e-14",
                  "main:mesh=grid.mdec" if parallel else "main:mesh=grid.mesh",
                  "main:backend=cuda" if label == "gpu" else "main:backend=cpu"]
        whole, out = case / ("whole-" + label), case / ("continued-" + label)

        # Продвигает тестовый расчёт на следующий интервал проверки.
        def advance(directory, until, *options):
            log = run([*common, f"--output={directory}", f"implicit:steady_max_steps={until}", *options],
                      success=False)
            assert "pseudo-step limit reached before convergence" in log, log
            if mode == "work":
                # Управление по работе BiCGSTAB ограничивает весь диапазон dt.
                # У него нет помеченных контрольных объёмов и источников в шаблоне.
                assert "LTS residual limit" not in log and "nodes with stored cap" not in log, log
                assert "LTS probe stencil" not in log, log
            return log

        automatic = advance(whole, end, f"main:steps_fo={switch}", "main:reconstruction=ebr5")
        first = advance(out, switch, "main:reconstruction=first_order")
        high_order = advance(out, split, "--recover", "main:reconstruction=ebr5")
        scheme, (_, _, cap, count, first_sum, second_sum, work, previous_work) = spread_record(out, split)
        assert scheme.split("/", 1)[0] == "ebr5" and count == (split-switch) % 32, (scheme, count)
        assert first_sum and second_sum and work >= count and previous_work >= 32 and cap >= 1.

        # Чтение управляемого верхнего предела не начинает новую серию:
        # значение 100 выше текущего и не меняет разрешённого в тесте шага.
        (out / "control.txt").write_text("local_time_step_max_ratio = 100\n")
        more = advance(out, end, "--recover", "main:reconstruction=ebr5")
        assert "Control accepted: local_time_step_max_ratio" in more, more
        limits, expected = reported_limits(first + high_order + more), reported_limits(automatic)
        assert limits.keys() == expected.keys() == set(range(1, end+1))
        assert limits[switch] > 1. and all(limits[k] == 1. for k in range(switch+1, switch+33)), limits
        assert limits[split+1] == limits[split] and limits[switch+65] != limits[split], limits
        for k, ratio in expected.items():
            assert math.isclose(limits[k], ratio, rel_tol=2e-6), (label, k, limits[k], ratio)
        # Выбранная зона записывает один общий файл независимо от числа рангов.
        # Сравниваем именно последний принятый слой обеих веток.
        resumed, = out.glob(f"resumed-{split}-*")
        files = [next(directory.rglob("final.fields")) for directory in (whole, resumed)]
        (a, reference), (b, continued) = map(binary_snapshot, files)
        assert a[3] == b[3] == end
        delta = max_difference(reference, continued)
        assert delta < 2.e-10, (label, delta)
        print(f"PASS LTS {mode} after LO to EBR5 and mid-block restart ({label}), maximum field difference {delta:.3g}")


# Разбирает параметры запуска, выполняет набор QA-проверок и возвращает статус результата.
def check_recovery_and_probe(case, args):
    """Пересчёт невязки после смены BC и диагностика на границе LO/EBR5."""
    case.mkdir()
    prepare(case)
    shutil.copyfile(ROOT / "templates/scheme.txt", case / "scheme.txt")
    if args.mpi:
        run([args.meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2"])
    (case / "cuda.txt").write_text(f"devices={args.device}\n")
    for label in ["cpu"] + (["mpi"] if args.mpi else []) + (["gpu"] if args.gpu else []):
        command = ([args.mpiexec, "-n", "2"] if label == "mpi" else []) + [
            args.aero, f"--input={case}", "implicit:local_time_step=auto",
            "main:mesh=grid.mdec" if label == "mpi" else "main:mesh=grid.mesh",
            "main:backend=cuda" if label == "gpu" else "main:backend=cpu"]
        for mode in ("work", "stencil"):
            out = case / (label + "-" + mode)
            (case / "init.txt").write_text("initial_type=formula\nrho=1\np=1\nux=0\n")
            (case / "bc.txt").write_text("[xmin, xmax]\nbc_type novisc_wall\n")
            settings = [f"--output={out}", f"implicit:local_time_step_residual_limit={mode}",
                        "implicit:steady_criterion=absolute", "implicit:steady_tol=1e-8",
                        "implicit:steady_max_steps=1"]
            initial = run([*command, *settings])
            assert "Completed 0 steps" in initial, initial
            (case / "bc.txt").write_text(
                "[xmin]\nbc_type free\nreference_state rho=1 p=1 ux=.1\n"
                "superin external\ninflow external\noutflow external\n"
                "[xmax]\nbc_type novisc_wall\n")
            resumed = run([*command, *settings, "--recover"], success=False)
            assert "pseudo-step limit reached before convergence" in resumed and "STEP 1 " in resumed, resumed
        (case / "bc.txt").write_text("[xmin, xmax]\nbc_type novisc_wall\n")
        (case / "init.txt").write_text("initial_type=formula\nrho=1+.2*x\np=1+.2*x\nux=0\n")
        log = run([*command, f"--output={case / (label + '-probe')}",
                   "main:reconstruction=ebr5", "main:steps_fo=3",
                   "implicit:local_time_step_residual_limit=stencil",
                   "implicit:local_time_step_probe_node=7", "implicit:steady_max_steps=5",
                   "implicit:steady_tol=1e-14"], success=False)
        rates = {int(k): float(v) for k, v in report_rows(log, r"^LTS probe +node = 7 +dt = \S+ +gas rate = (\S+)")}
        assert set(rates) == set(range(1, 6)) and all(math.isfinite(v) and v > 0 for v in rates.values()), rates
        print(f"PASS recovery BC change and LO transition probe ({label})")


def main():
    parser = mpi_parser(__doc__)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    with mpi_workspace("lts-neighbors-") as work:
        case = Path(work)
        prepare(case)
        check_cfl_after_limit(case, args)
        check_recovery_and_probe(case / "regressions", args)
        original = solve(case, args, "default")
        expected = solve(case, args, "limited", limit=2)
        if max_difference(original, expected) < 1.e-7:
            raise AssertionError("Ограничение не изменило решение на неоднородной сетке")
        # Предел 1 делает шаги на всей линии одинаковыми. Независимый эталон
        # даёт уже существующее глобальное ограничение при том же значении.
        uniform = solve(case, args, "global-1", global_limit=1)
        if max_difference(uniform, solve(case, args, "neighbor-1", limit=1)) > 2.e-11:
            raise AssertionError("При отношении 1 не воспроизведён единый шаг")
        combined = solve(case, args, "combined", limit=2, global_limit=5)
        feedback = solve(case, args, "feedback", limit=2, residual_limit=True)
        if args.mpi:
            run([args.meshdec, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2"])
        if args.gpu:
            (case / "cuda.txt").write_text(f"devices={args.device}\n")
        for label, use_mpi, use_gpu in (("MPI", args.mpi, False), ("GPU", False, args.gpu)):
            if use_mpi or use_gpu:
                check_cfl_after_limit(case, args, label, use_mpi, use_gpu)
                fields = solve(case, args, label, limit=2, mpi=use_mpi, gpu=use_gpu)
                delta = max_difference(expected, fields)
                if delta > 2.e-11:
                    raise AssertionError(f"{label}: максимальное расхождение полей {delta:g}")
                controlled = solve(case, args, label + "-feedback", limit=2, mpi=use_mpi, gpu=use_gpu, residual_limit=True)
                feedback_delta = max_difference(feedback, controlled)
                if feedback_delta > 2.e-10:
                    raise AssertionError(f"{label}: различаются поля после двух проверок местной невязки")
                other = solve(case, args, label + "-combined", limit=2, global_limit=5, mpi=use_mpi, gpu=use_gpu)
                combined_delta = max_difference(combined, other)
                if combined_delta > 2.e-11:
                    raise AssertionError(f"{label}: расхождение при двух ограничениях {combined_delta:g}")
                print(f"PASS LTS neighbor ratio {label}, maximum field difference {delta:.3g}, combined {combined_delta:.3g}, residual limit {feedback_delta:.3g}")
        check_stencil(case, args)
        for mode in ("stencil", "work", "cfl"):
            check_spread_restart(case / ("startup-restart-" + mode), args, mode)
        print("PASS LTS neighbor ratio on a graded line")


if __name__ == "__main__":
    main()
