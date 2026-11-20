"""Сверяет непрерывный расчёт с остановкой по команде и продолжением из recovery."""
import shutil
import re
import subprocess
from pathlib import Path

from mpi_common import ROOT, mpi_parser, mpi_workspace, report_rows
from mpi_geometry import run
from mpi_fields import binary_snapshot


def check_cli(args, work):
    """Проверяет справку именно собранного солвера, в том числе число сообщений в MPI."""
    case = work / "cli"
    shutil.copytree(ROOT / "qa/cases/pulse_1d", case)
    aero = args.aero.resolve()
    def launch(arguments):
        return subprocess.run(arguments, cwd=case, capture_output=True, text=True, errors="replace", timeout=45)

    help_text = launch([aero, "--help"])
    assert help_text.returncode == 0 and "Usage: aero " in help_text.stdout and not help_text.stderr, help_text
    short = launch([aero, "-h"])
    assert short.returncode == 0 and short.stdout == help_text.stdout, short
    for options, reason in (([], ""), (["--input=."], "missing --output"), (["--output=."], "missing --input"),
                            (["--input="], "expected --input"), (["main:cfl="], "invalid parameter override")):
        result = launch([aero, *options])
        assert result.returncode != 0 and reason in result.stderr and help_text.stdout in result.stderr, result
    parallel = launch([args.mpiexec, "-n", "2", aero, "--input=."])
    messages = parallel.stdout + parallel.stderr
    assert parallel.returncode != 0 and messages.count("Usage: aero ") == messages.count("missing --output") == 1, parallel
    assert not list(case.glob("recovery-*.bin"))
    # Параметры из командной строки по-прежнему запускают расчёт прямо в каталоге постановки.
    result = launch([aero, "main:final_time=0", "gas.txt:gamma=1.4"])
    assert result.returncode == 0 and (case / "recovery-0.bin").is_file(), result
    result = launch([aero, "main:cfl=invalid"])
    assert result.returncode != 0 and "Usage: aero " not in result.stderr and "cfl" in result.stderr.lower(), result
    print("OK command-line help, case overrides and two MPI ranks", flush=True)


def check_cfl_inputs(args, work):
    """Один набор значений проходит через стартовый файл и через команду после первого шага."""
    case = work / "cfl-inputs"
    case.mkdir()
    sample = ROOT / "qa/cases/pulse_1d"
    for path in sample.glob("*.txt"):
        shutil.copyfile(path, case / path.name)
    common = [args.aero, f"--input={case}", f"main:mesh={sample / 'grid.mesh'}",
              "main:stabilization=cfl", "main:final_time=.006"]
    base = dict(cfl_init=".25", cfl_min=".1", cfl_max=".5", cfl_coeff="2", cfl_num_steps="2")
    cases = [
        (dict(cfl_init=".1"), True), (dict(cfl_init="5e-1"), True),
        (dict(cfl_min=".25", cfl_max=".25", cfl_num_steps="1"), True),
        (dict(cfl_init=".09"), False), (dict(cfl_init=".51"), False),
        (dict(cfl_min="0"), False), (dict(cfl_min=".6"), False),
        (dict(cfl_coeff="1"), False), (dict(cfl_num_steps="0"), False),
        (dict(cfl_num_steps="1.5"), False), (dict(cfl_coeff="nan"), False),
        (dict(cfl_max="1e999"), False), (dict(cfl_init=".25x"), False),
    ]
    def entries(values, live):
        keys = {"cfl_init": "cfl"} if live else {}
        return "".join(f"{keys.get(k, k)}={v}\n" for k, v in values.items())

    for index, (override, valid) in enumerate(cases):
        values = dict(base, **override)
        for live in (False, True):
            dest = case / f"out-{index}-{live}"
            dest.mkdir()
            (case / "cfl.txt").write_text(entries(base if live else values, False))
            if live:
                (dest / "control.txt").write_text(entries(values, True))
            log = run([*common, f"--output={dest}"], success=live or valid)
            if live:
                assert ("Control applied" if valid else "Control rejected") in log, log
            elif not valid:
                assert "cfl.txt" in log and "Starting time integration" not in log, log
    # Через CLI новые параметры действуют с первого шага. Через control.txt
    # они вступают в силу после шага 1 и начинают отсчёт успешных шагов заново.
    (case / "cfl.txt").write_text(entries(base, False))
    schedule = {"cfl_num_steps": "1", "cfl_coeff": "1.5"}
    for live in (False, True):
        dest = case / ("schedule-control" if live else "schedule-cli")
        dest.mkdir()
        options = [*common, f"--output={dest}", "main:report_every=1", "main:final_time=.05"]
        if live:
            (dest / "control.txt").write_text(entries(schedule, True))
        else:
            options += [f"cfl:{key}={value}" for key, value in schedule.items()]
        log = run(options)
        cfl = [float(value) for _, value in report_rows(log, r"^t = .*?\bCFL = (\S+)")]
        expected = [.25, .25, .375, .5] if live else [.25, .375, .5]
        assert cfl[:len(expected)] == expected, log
    print(f"OK paired CFL validation ({len(cases)} cases) and CLI/live CFL growth", flush=True)


def solver_command(args, case, parallel, mesh):
    """Готовит один и тот же расчёт на CPU или на двух разделах MPI."""
    if not parallel:
        return [args.aero], mesh
    archive = case / "grid.mdec"
    run([args.meshdec, mesh, archive, "--target-ranks=2", "--compress"])
    cmd = [args.mpiexec, "-n", "2", args.aero]
    if args.devices:
        (case / "cuda.txt").write_text(f"devices={args.devices}\n")
        cmd.append("main:backend=cuda")
    return cmd, archive


def check_steady_steps(args, work, parallel):
    """Переключает псевдошаг в ходе расчёта и проверяет режим после recovery."""
    case = work / f"steady-{parallel}"
    sample = ROOT / "qa/cases/sod_1d"
    case.mkdir()
    for path in sample.glob("*.txt"):
        shutil.copyfile(path, case / path.name)
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")
    dest = case / "out"
    dest.mkdir()
    cmd, mesh = solver_command(args, case, parallel, sample / "grid.mesh")
    cmd += [f"--input={case}", f"--output={dest}", f"main:mesh={mesh}", "main:time_scheme=steady",
            "main:reconstruction=first_order", "main:report_every=1", "implicit:steady_max_steps=3",
            "implicit:steady_tol=1e-15", "implicit:simplified_newton=on", "implicit:jacobian_update_every=3"]

    def phase(start, command, expected, initial="off", reject=False):
        (dest / "control.txt").write_text(command)
        options = [f"implicit:local_time_step={initial}"]
        if start:
            options.append("--recover")
        log = run(cmd + options, success=False)
        assert "pseudo-step limit reached before convergence" in log, log
        assert ("Control rejected" if reject else "Control applied") in log, log
        # В первом отчёте действует прежний режим. В следующих двух проверяем
        # фактически принятые dt: 0 в ожиданиях обозначает единый шаг.
        ratios = {int(i): float(hi)/float(lo) for i, lo, hi in
                  report_rows(log, r"^dt_min = (\S+)\s+dt_max = (\S+)")}
        ratios.update((int(i), 0.) for (i,) in report_rows(log, r"^Pseudo-time = .*?\bdt = "))
        assert set(ratios) == set(range(start + 1, start + 4)), log
        for i, value in enumerate(expected, start + 1):
            if value is None:
                assert ratios[i] > 1.06, log
            else:
                assert abs(ratios[i] - value) < 2e-6, log

    phase(0, "local_time_step=on\nlocal_time_step_max_ratio=1.05\n", [0., 1.05, 1.05])
    phase(3, "local_time_step_max_ratio=off\n", [1.05, None, None])
    phase(6, "local_time_step=off\n", [None, 0., 0.])
    # Ошибка в ограничении отменяет и включение режима. Проверяем также, что
    # при старте восстановлен сохранённый off, хотя в параметрах теперь on.
    phase(9, "local_time_step=on\nlocal_time_step_max_ratio=.5\n", [0., 0., 0.], initial="on", reject=True)
    print(f"OK live steady time steps, limits and recovery, MPI={parallel}", flush=True)


def check(args, work, parallel, scheme):
    case = work / f"{parallel}-{scheme}"
    case.mkdir()
    sample = ROOT / "qa/cases" / ("pulse_1d" if scheme == "bdf2" else "sod_1d")
    for path in sample.glob("*.txt"):
        shutil.copyfile(path, case / path.name)
    for name in ("implicit", "cfl"):
        shutil.copyfile(ROOT / f"templates/{name}.txt", case / f"{name}.txt")
    zone = "layout=global\nformat=binary\nmulti=0\nvariables=rho,ux,p,E\n"
    (case / "output.txt").write_text(f"[end]\nfile=end.fields\nevery_time=.012\n{zone}"
                                     f"[rare]\nfile=rare.fields\nevery_steps=99999\n{zone}"
                                     "[history]\nfile=history.fields\nevery_time=1\nmulti=1\n"
                                     "layout=global\nformat=binary\nvariables=rho,ux,p,E\n")
    cmd, mesh = solver_command(args, case, parallel, sample / "grid.mesh")
    cmd += [f"--input={case}", f"main:mesh={mesh}", f"main:time_scheme={scheme}",
            "main:stabilization=on", "main:reconstruction=mpweno5", "main:final_time=.012"]
    if scheme == "bdf2":
        cmd += ["main:reconstruction=first_order"]
    if scheme == "ssprk3":
        cmd += ["cfl:cfl_init=4", "cfl:cfl_max=4"]  # На первом шаге появляются реальные метки отказов.
    changes = ("upw_coeff=.8\nmp6_hybrid=.4\ncfl_min=.02\ncfl_max=.4\ncfl=.18\n"
               "cfl_num_steps=4\ncfl_coeff=1.2\nreset_fail\n")
    for name, commands in (("full", changes), ("split", changes + "STOP\n")):
        dest = case / name
        dest.mkdir()
        (dest / "control.txt").write_text("misspelled = not-a-number\n" + commands)
        log = run([*cmd, f"--output={dest}"])
        assert "Control applied after step 1, CFL = " in log, log
        assert log.count("unknown parameter 'misspelled'") == 1, log
        for line in commands.splitlines():
            expected = line.replace("=", " = ")
            assert log.splitlines().count("Control accepted: " + expected) == 1, log
        accepted = int(re.search(r"Completed (\d+) steps", log)[1])
        latest = binary_snapshot(dest / "end.fields")
        assert latest[0][3] == accepted and binary_snapshot(dest / "rare.fields") == latest
        assert list(dest.glob("history_*.fields")) == [dest / f"history_{accepted:06d}.fields"]
        assert binary_snapshot(dest / f"history_{accepted:06d}.fields") == latest
        assert len(list(dest.glob("recovery-*.bin"))) == 1
        if scheme == "ssprk3":
            assert "RETRY step 1" in log and re.search(r"Control applied[^\n]+first-order cells = 0", log)
        if name == "split":
            assert "STOP: final fields and recovery saved at step 1" in log
            assert binary_snapshot(dest / "rare.fields")[0][-2] == 1
            run([*cmd, f"--output={dest}", "--recover"])
    expected = binary_snapshot(case / "full/end.fields")
    # Продолжение без новых шагов тоже публикует все зоны в новом каталоге,
    # подтверждая, что итоговая контрольная точка читается полностью.
    log = run([*cmd, f"--output={case / 'full'}", "--recover"])
    assert f"Recovered step {expected[0][3]}" in log
    assert binary_snapshot(next((case / "full").glob("resumed-*/rare.fields"))) == expected
    actual = binary_snapshot(next((case / "split").glob("resumed-*/end.fields")))
    assert actual[0] == expected[0]
    for key in expected[1]:
        assert max(abs(a-b) for a,b in zip(actual[1][key], expected[1][key])) < 1e-11
    if not parallel and scheme == "ssprk3":
        # Пока солвер работает, посылаем вторую команду после отклонения первой.
        dest = case / "live"
        dest.mkdir()
        (dest / "control.txt").write_text("cfl=.1\nSTOP\ncfl_coeff=0\n")
        launch = [*cmd, f"--output={dest}", "main:control_every=25", "main:final_time=10"]
        with subprocess.Popen(launch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              errors="replace") as process:
            try:
                for line in process.stdout:
                    if "Control rejected" in line:
                        pending = dest / "control.tmp"
                        pending.write_text("STOP\n")
                        pending.rename(dest / "control.txt")
                        break
                tail = process.communicate(timeout=60)[0]
                match = re.search(r"STOP: final fields and recovery saved at step (\d+)", tail)
                assert process.returncode == 0 and match and int(match[1]) % 25 == 0, tail
            finally:
                if process.poll() is None:
                    process.kill()
    print(f"OK runtime control {scheme}, MPI={parallel}", flush=True)


def check_zero_step(args, work):
    """Из точки (0, 0) можно продолжить даже при отключённых расписании и управлении."""
    sample = ROOT / "qa/cases/pulse_1d"
    out = work / "zero-step"
    options = [args.aero, f"--input={sample}", f"--output={out}", "main:recovery_every_steps=0",
               "main:control_every=0"]
    run([*options, "main:final_time=0"])
    assert (out / "recovery-0.bin").is_file()
    log = run([*options, "main:final_time=.006", "--recover"])
    assert "Recovered step 0 at time 0" in log and "Final time = 0.006" in log, log



if __name__ == "__main__":
    parser = mpi_parser(__doc__)
    parser.add_argument("--devices", help="GPU IDs for MPI")
    args = parser.parse_args()
    with mpi_workspace("run-control-") as tmp:
        check_cli(args, Path(tmp))
        check_zero_step(args, Path(tmp))
        check_cfl_inputs(args, Path(tmp))
        for mpi in (False, True):
            check_steady_steps(args, Path(tmp), mpi)
            for time_scheme in ("ssprk3", "bdf2"):
                check(args, Path(tmp), mpi, time_scheme)
