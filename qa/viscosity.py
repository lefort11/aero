"""Сквозная проверка AES: затухание сдвига, стенка и сверка полных полей CPU, GPU и MPI."""
import measurements
import math
import re
from pathlib import Path
import shutil

from mpi_common import ROOT, backend_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run
from run import nodes


END = .025
AMPLITUDE = .001
REYNOLDS = 20.


def setup(case, mpi, dim=2):
    """На периодической сетке сдвиг ux(y) затухает за счёт молекулярной вязкости."""
    sample = ROOT / "qa/cases" / ("momentum_2d" if dim == 2 else "wave_3d")
    for name in ("grid.mesh", "bc.txt"):
        shutil.copy(sample / name, case / name)
    (case / "main.txt").write_text(
        "mesh=grid.mesh\nbackend=cpu\nviscosity=normal\nreconstruction=first_order\n"
        "reconstruction_variables=characteristic\neuler_solver=lf\nupw_coeff=0\n"
        f"time_scheme=ssprk3\nfinal_time={END}\ncfl=.45\n"
    )
    (case / "gas.txt").write_text(f"gamma=1.4\nmach=.5\nreynolds={REYNOLDS}\nprandtl=.72\nt_amb=300\n")
    (case / "init.txt").write_text(
        f"initial_type=formula\nrho=1\np=1/(1.4*.5^2)\nux={AMPLITUDE}*sin(2*pi*y)\nuy=0\n"
    )
    variables = "rho,ux,uy,p,E" if dim == 2 else "rho,ux,uy,uz,p,E"
    (case / "output.txt").write_text(
        f"[state]\nfile=state.fields\nformat=binary\nlayout=global\nevery_time={END}\nvariables={variables}\n"
    )
    (case / "cuda.txt").write_text("devices=0,1\n")
    shutil.copy(ROOT / "templates/implicit.txt", case / "implicit.txt")
    if mpi:
        run([mpi, case / "grid.mesh", case / "grid.mdec", "--target-ranks=2", "--compress"])
    return nodes(case / "grid.mesh", dim)


def solve(case, args, name, gpu=False, distributed=False, options=()):
    target = case / name
    command = [args.aero.resolve(), f"--input={case}", f"--output={target}", *options]
    if distributed:
        command = [args.mpiexec, "-n", "2", *command, "main:mesh=grid.mdec"]
    if gpu:
        command.extend(["main:backend=cuda", "cuda:devices=" + (args.devices if distributed else args.devices.split(",")[0])])
    log = run(command)
    if "Active backend = " + ("CUDA" if gpu else "CPU") not in log or "Completed " not in log:
        raise AssertionError(log)
    metadata, fields = binary_snapshot(target / "state.fields")
    if "implicit:newton_max_iter=1" in options and "implicit:newton_without_predictor=on" in options:
        # В этой проверке отключаем предиктор явно, чтобы неоднородное поле
        # действительно требовало сборки и решения системы Ньютона.
        relative = [float(s) for s in re.findall(r"Newton .*relative residual = ([\d.eE+-]+)",log)]
        if measurements.compare(metadata[3], "<", 2) or measurements.compare(len(relative), "!=", metadata[3]) or not all(measurements.compare(r, "<=", 1.e-5) for r in relative):
            raise AssertionError(f"{name}: incomplete BDF steps or Newton convergence reports: {relative}")
        print(f"{name}: {metadata[3]} BDF steps, largest relative Newton residual {max(relative):.3g}",flush=True)
    return fields


def max_difference(left, right):
    if measurements.compare(left.keys(), "!=", right.keys()) or any(measurements.compare(len(left[k]), "!=", len(right[k])) for k in left):
        sizes = lambda fields: {key: len(values) for key, values in fields.items()}
        raise AssertionError(f"Saved field sizes differ: left={sizes(left)}, right={sizes(right)}")
    difference = [abs(a-b) for k in left for a, b in zip(left[k], right[k])]
    if not all(math.isfinite(x) for x in difference):
        key, i, a, b = next((key, i, a, b) for key in left
                            for i, (a, b) in enumerate(zip(left[key], right[key])) if not math.isfinite(a-b))
        raise AssertionError(f"Nonfinite field difference: {key}[{i}] left={a!r}, right={b!r}, |delta|={abs(a-b)!r}")
    return max(difference)


def compare_backends(case, args, name, reference, options=()):
    for gpu, mpi in ((False, True), (True, False), (True, True)):
        if (gpu and not args.gpu) or (mpi and not args.mpi):
            continue
        if gpu and mpi and len(args.devices.split(",")) < 2:
            continue
        label = name + ("-gpu" if gpu else "-cpu") + ("-mpi" if mpi else "")
        delta = max_difference(reference, solve(case, args, label, gpu, mpi, options))
        if measurements.error_exceeds(delta, 2.e-10):
            raise AssertionError(f"{label}: field difference {delta:.3g}")
        print(f"{label}: maximum field difference {delta:.3g}", flush=True)


def check_thermal_3d(case,args):
    """На гексаэдрах проверяем изменение вязкости с температурой и обмен через три периодических шва."""
    case.mkdir()
    coords = setup(case,args.meshdec.resolve() if args.mpi else None,dim=3)
    (case / "init.txt").write_text(
        "initial_type=formula\nrho=1+.12*sin(2*pi*x)\n"
        "ux=.07*sin(2*pi*y)\nuy=.05*cos(2*pi*z)\nuz=.06*cos(2*pi*x)\np=1.6+.17*cos(2*pi*(x+y+z))\n"
    )
    # В SSPRK3 одиночный процесс готовит вязкость при обновлении узла. MPI повторяет
    # расчёт после обмена, поэтому сравнение полей выявляет устаревшие значения со стадии.
    transport = ["main:viscosity=sutherland", "gas:sutherland_s=111", "gas:reynolds=4"]
    cases = (("ssprk3", ["main:cfl=.45"]),
             ("bdf2", ["main:cfl=4", "implicit:preconditioner=CSGS", "implicit:newton_tol=1e-9"]))
    # На равномерной периодической сетке объёмы одинаковы. Каждый узел шва берём
    # один раз. Аналитические средние синусов равны нулю, их квадратов — 1/2.
    interior = [i for i, point in enumerate(coords) if all(x < 1. for x in point)]
    initial_energy = 1.6/.4 + .25*(.07**2+.05**2+.06**2)
    # Вычисляет среднее контрольных значений поля.
    def average(values):
        return math.fsum(values) / len(interior)
    for scheme, settings in cases:
        options = [*transport, f"main:time_scheme={scheme}", *settings]
        name = "thermal-" + scheme
        fields = solve(case, args, name, options=options)
        density = fields["rho"]
        mass = average(density[i] for i in interior)
        energy = average(fields["E"][i] for i in interior)
        momentum = [average(density[i]*fields[k][i] for i in interior) for k in ("ux","uy","uz")]
        drift = max(abs(mass-1.), abs(energy-initial_energy), *(abs(m) for m in momentum))
        if not measurements.compare(drift, "<", 2.e-10):
            raise AssertionError(f"Periodic 3D conservation error {drift:.3g}")
        print(f"3D {scheme.upper()}: maximum conservation error {drift:.3g}", flush=True)
        compare_backends(case, args, name, fields, options)


# Проверяет вязкую невязку, теплопроводность и согласованность CPU, GPU и MPI.
def verify(case, args):
    coords = setup(case, args.meshdec.resolve() if args.mpi else None)
    inviscid = solve(case, args, "off", options=["main:viscosity=off"])
    viscous = solve(case, args, "shear")
    # На декартовой сетке AES сводится к центральной второй разности. Сравниваем
    # с собственной частотой дискретного оператора; нагрев порядка амплитуды в квадрате.
    count_y = len({y for _, y, _ in coords}) - 1
    frequency = 2 * count_y * math.sin(math.pi / count_y)
    decay = math.exp(-END * frequency**2 / REYNOLDS)
    wave = [AMPLITUDE * math.sin(2*math.pi*y) for _, y, _ in coords]
    error = max(abs(a - decay*b) for a, b in zip(viscous["ux"], wave))
    if measurements.error_exceeds(error, 2.e-8) or measurements.error_exceeds(max(abs(a-b) for a, b in zip(inviscid["ux"], wave)), 1.e-11):
        raise AssertionError(f"Unexpected shear damping: max error {error:.3g}")
    measured = sum(a*b for a, b in zip(viscous["ux"], wave)) / sum(b*b for b in wave)
    print(f"Shear: amplitude ratio {measured:.7f} (expected {decay:.7f}), maximum velocity error {error:.3g}", flush=True)
    compare_backends(case, args, "shear", viscous)

    # Большой CFL усиливает роль вязкого оператора. Для линейного сдвига одной
    # ньютоновской поправки достаточно. Ограничение не даёт скрыть пропущенные блоки
    # большим числом итераций; BDF2 проходит также шаги с предыдущим слоем.
    for scheme in ("bdf1","bdf2"):
        options = [f"main:time_scheme={scheme}", "main:cfl=3", "main:report_every=1",
                   "implicit:newton_max_iter=1", "implicit:newton_without_predictor=on", "implicit:newton_tol=1e-5"]
        if scheme == "bdf2":
            options += ["implicit:preconditioner=CSGS","implicit:jacobian_update_every=3",
                        "implicit:simplified_jacobian=on"]
        reference = solve(case,args,scheme,options=options)
        error = max(abs(a - decay*b) for a, b in zip(reference["ux"],wave))
        if measurements.error_exceeds(error, 5.e-7):
            raise AssertionError(f"{scheme}: incorrect viscous damping, error {error:.3g}")
        compare_backends(case,args,scheme,reference,options)

    # На стенке начальный поток имеет касательную скорость. Её принудительно
    # обнуляем, давление сохраняем. Дальше через обе стенки нет переноса энергии.
    (case / "bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
                                 "[ymin]\nbc_type wall\n[ymax]\nbc_type wall\n")
    (case / "init.txt").write_text("initial_type=freestream\nphi=0\ntheta=0\n")
    for scheme in ("bdf1","bdf2"):
        options = ["main:viscosity=sutherland", "gas:sutherland_s=111", f"main:time_scheme={scheme}"]
        name = "wall-" + scheme
        wall = solve(case, args, name, options=options)
        weighted_change = 0.
        weight_sum = 0.
        for i, (x,y,_) in enumerate(coords):
            at_wall = y in (0.,1.)
            if at_wall and (measurements.compare(wall["ux"][i], "!=", 0.) or measurements.compare(wall["uy"][i], "!=", 0.)):
                raise AssertionError("Nonzero wall velocity")
            weight = (.5 if x in (0.,1.) else 1.) * (.5 if at_wall else 1.)
            initial_energy = 1/(1.4*.5**2*.4) + (0. if at_wall else .5)
            weighted_change += weight*(wall["E"][i]-initial_energy)
            weight_sum += weight
        drift = abs(weighted_change / weight_sum)
        if not measurements.compare(drift, "<", 3.e-8):
            raise AssertionError(f"Energy changed at an adiabatic stationary wall: {drift:.3g}")
        print(f"{name}: velocity zero, mean energy change {drift:.3g}", flush=True)
        compare_backends(case, args, name, wall, options)

    check_thermal_3d(case / "thermal-3d",args)



if __name__ == "__main__":
    parser = backend_parser(__doc__)
    with mpi_workspace("aes-") as workspace:
        verify(Path(workspace), parser.parse_args())
