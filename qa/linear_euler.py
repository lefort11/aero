"""Сверяет пульсации и энергию с точным переносом акустических и энтропийных возмущений."""

import math
from pathlib import Path
import shutil
from mpi_common import ROOT, mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from run import nodes

END = 0.032
FREQUENCY = 2 * math.pi


def exact(x, t):
    """Фон rho0=2, ux0=1/4 и c0=1. Две звуковые компоненты движутся в разные стороны."""
    plus = 0.017 * math.sin(FREQUENCY * (x - 1.25 * t))
    minus = 0.009 * math.cos(FREQUENCY * (x + 0.75 * t))
    entropy = 0.03 * math.cos(2 * FREQUENCY * (x - 0.25 * t))
    rho, u, p = entropy + plus + minus, (plus - minus) / 2, plus + minus
    return {"rho": rho, "ux": u, "p": p, "E": p / 0.4 + 0.5 * u + 0.03125 * rho}


def analytic_error(fields, coordinates):
    return max(abs(fields[k][j] - exact(x, END)[k]) for j, x in enumerate(coordinates) for k in fields)


def prepare(case):
    mesh = case / "grid.mesh"
    shutil.copyfile(ROOT / "qa/cases/sod_1d/grid.mesh", mesh)
    (case / "gas.txt").write_text("gamma = 1.4\n")
    (case / "linear.txt").write_text("rho0 = 2\nux0 = .25\np0 = 1.4285714285714286\n")
    (case / "bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n")
    (case / "init.txt").write_text(
        "initial_type formula\nrho=.03*cos(4*pi*x)+.017*sin(2*pi*x)+.009*cos(2*pi*x)\n"
        "ux=(.017*sin(2*pi*x)-.009*cos(2*pi*x))/2\np=.017*sin(2*pi*x)+.009*cos(2*pi*x)\n"
    )
    (case / "main.txt").write_text(
        "Equations=Linear\nmesh=grid.mesh\nbackend=cpu\ncfl=.35\n"
        "reconstruction=first_order\nreconstruction_variables=conservative\n"
        f"euler_solver=lf\ntime_scheme=ssprk3\nfinal_time={END}\nreport_every=1\nomp_threads=2\n"
    )
    (case / "output.txt").write_text(
        f"[all]\nfile=all.fields\nevery_time={END}\nlayout=global\nmulti=0\n"
        "format=binary\nvariables=rho,ux,p,E\n"
    )
    shutil.copyfile(ROOT / "templates/implicit.txt", case / "implicit.txt")
    return [point[0] for point in nodes(mesh, 1)]


def solve(command, case, dest, options):
    log = run([*command, f"--input={case}", f"--output={dest}", *options])
    gpu = "main:backend=cuda" in options
    backend = "CUDA" if gpu else "CPU"
    if len(command) > 1:
        backend += f" (MPI), processes = {command[-2]}"
    if "Equations = Linear" not in log or "Completed " not in log or "Active backend = " + backend not in log:
        raise AssertionError(log)
    return binary_snapshot(dest / "all.fields")


def check_restart(aero, case):
    """BDF2 должен восстановить оба слоя и отклонить запуск с другим фоном."""
    opts = [
        "main:time_scheme=BDF2",
        "main:reconstruction=ebr3",
        "implicit:jacobian_flux=Roe",
        "implicit:preconditioner=CSGS",
        "main:recovery_every_steps=1000",
    ]
    directory = case / "restart"
    checkpoint, saved_fields = solve([aero], case, directory, opts)
    restart = [
        aero,
        f"--input={case}",
        f"--output={directory}",
        "--recover",
        *opts,
        f"main:final_time={2*END}",
    ]
    failed = run(restart + ["linear:ux0=-.25"], success=False)
    if "incompatible mesh or gas" not in failed:
        raise AssertionError(failed)
    run(restart)
    resumed = binary_snapshot(directory / f"resumed-{checkpoint[3]}-1" / "all.fields")
    continuous = solve([aero], case, case / "continuous", opts + [f"main:final_time={2*END}"])
    if resumed != continuous:
        raise AssertionError("BDF2 recovery changed fields or snapshot metadata")
    # При чтении скалярных полей пульсации возвращаются с исходным знаком.
    # Здесь проверяем начальное состояние до первого шага интегратора.
    (case / "init.txt").write_text("initial_type file\npath restart/all.fields\nrho=rho\nux=ux\np=p\n")
    _, imported = solve(
        [aero],
        case,
        case / "import",
        ["main:time_scheme=steady", "implicit:steady_criterion=absolute", "implicit:steady_tol=1e6"],
    )
    if imported != saved_fields:
        raise AssertionError("Loading perturbations from a field file changed the initial state")
    print("OK field import, BDF2 restart and rejection of incompatible background", flush=True)


def check_wall(aero, mpi, distributed, case, coords):
    """Стоячая звуковая волна имеет нулевую нормальную скорость на обеих стенках."""
    (case / "bc.txt").write_text("[xmin]\nbc_type wall\n[xmax]\nbc_type wall\n")
    (case / "init.txt").write_text("initial_type formula\nrho=.01*cos(pi*x)\nux=0\np=.01*cos(pi*x)\n")
    bad = run([aero, f"--input={case}", f"--output={case/'wall-bad'}"], success=False)
    if "tangent" not in bad:
        raise AssertionError(bad)
    opts = distributed + [
        "linear:ux0=0",
        "main:reconstruction=ebr3",
        "main:time_scheme=bdf2",
        "implicit:jacobian_flux=Roe",
    ]
    _, fields = solve(mpi, case, case / "wall", opts)
    error = 0
    for i, x in enumerate(coords):
        p = 0.01 * math.cos(math.pi * x) * math.cos(math.pi * END)
        u = 0.005 * math.sin(math.pi * x) * math.sin(math.pi * END)
        for key, value in (("p", p), ("rho", p), ("ux", u), ("E", p / 0.4)):
            error = max(error, abs(fields[key][i] - value))
    if error >= 3.0e-6:
        raise AssertionError(f"Standing wave error {error}")
    print(f"OK acoustic reflection at walls: analytic {error:.3g}", flush=True)


def check_shear(args, aero, mpi, distributed, case, dim):
    """Поперечная скорость и энтропия переносятся фоном. Давление постоянно."""
    sample = ROOT / "qa/cases" / ("momentum_2d" if dim == 2 else "wave_3d")
    shutil.copyfile(sample / "grid.mesh", case / "grid.mesh")
    shutil.copyfile(sample / "bc.txt", case / "bc.txt")
    run(
        [
            args.meshdec.resolve(),
            case / "grid.mesh",
            case / f"shear-{dim}d.mdec",
            "--target-ranks=2",
            "--compress",
        ]
    )
    (case / "linear.txt").write_text(
        "rho0=2\np0=1.4285714285714286\nux0=.25\nuy0=-.3\n" + ("uz0=.15\n" if dim == 3 else "")
    )
    (case / "init.txt").write_text(
        "initial_type formula\nrho=.02*cos(2*pi*y)\nux=.04*sin(2*pi*y)\n"
        "uy=0\np=-.003\n" + ("uz=-.01*sin(2*pi*y)\n" if dim == 3 else "")
    )
    names = ["rho", "ux", "uy"] + (["uz"] if dim == 3 else []) + ["p", "E"]
    (case / "output.txt").write_text(
        f"[all]\nfile=all.fields\nevery_time={END}\nlayout=global\n"
        "format=binary\nvariables=" + ",".join(names) + "\n"
    )
    variables = "characteristic" if dim == 2 else "flux"
    opts = [
        "main:reconstruction=MPWENO5",
        "scheme:mp6_hybrid=1",
        f"main:reconstruction_variables={variables}",
        "main:euler_solver=roe",
    ]
    reference = solve([aero], case, case / f"shear-cpu{dim}", opts)
    meta, fields = solve(
        mpi, case, case / f"shear-mpi{dim}", opts + distributed + [f"main:mesh=shear-{dim}d.mdec"]
    )
    if meta != reference[0]:
        raise AssertionError("Different shear snapshot metadata")
    errors = []
    for i, (_, y, _) in enumerate(nodes(case / "grid.mesh", dim)):
        phase = FREQUENCY * (y + 0.3 * END)
        rho, ux, uz = 0.02 * math.cos(phase), 0.04 * math.sin(phase), -0.01 * math.sin(phase)
        energy = -0.003 / 0.4 + 2 * (0.25 * ux + (0.15 * uz if dim == 3 else 0))
        energy += 0.5 * (0.25**2 + 0.3**2 + (0.15**2 if dim == 3 else 0)) * rho
        exact_fields = dict(rho=rho, ux=ux, uy=0, uz=uz, p=-0.003, E=energy)
        errors.extend(abs(fields[k][i] - exact_fields[k]) for k in names)
    diff = max(abs(a - b) for k in names for a, b in zip(reference[1][k], fields[k]))
    if not max(errors) < 2.0e-5 or not diff < 2.0e-10:
        raise AssertionError(f"{dim}D shear: analytic {max(errors)}, MPI difference {diff}")
    print(
        f"OK {dim}D shear, hybrid MPWENO5/{variables}: analytic {max(errors):.3g}, MPI {diff:.3g}", flush=True
    )


def check_exact_report(aero, mpi, case, distributed):
    """На периоде длины 1 ошибка (1+cos(2πx))*t имеет известные L2 и максимум."""
    init = case / "init.txt"
    original = init.read_bytes()
    init.write_text("initial_type formula\nrho=1+t\nux=0\np=1\n")
    (case / "exact.txt").write_text("rho=1+t*(1+cos(2*pi*x))\nux=0\np=1\n")
    for model in ("Euler", "Linear"):
        for label, command, extra in (("cpu", [aero], []), ("mpi", mpi, distributed)):
            out = case / f"exact-{model}-{label}"
            log = run([*command, f"--input={case}", f"--output={out}", f"main:Equations={model}",
                       "main:exact_solution=on", "main:report_every=0", *extra])
            report = (out / "sol_error.txt").read_text()
            if report not in log or report.count("Solution error") != 1:
                raise AssertionError("Expected one identical final error report in the file and console")
            name, l2, linf = report.splitlines()[2].split()
            if name != "rho" or abs(float(l2)-END*math.sqrt(1.5)) > 1e-12 or abs(float(linf)-2*END) > 1e-12:
                raise AssertionError(report)
    init.write_bytes(original)
    print("OK Euler and Linear exact norms, CPU and MPI", flush=True)


def check_modes(args, case):
    coords = prepare(case)
    aero = args.aero.resolve()
    run([args.meshdec.resolve(), case / "grid.mesh", case / "grid.mdec", "--target-ranks=2", "--compress"])
    devices = args.devices.split(",") if args.devices else []
    if devices:
        (case / "cuda.txt").write_text("devices=" + ",".join(devices))
    mpi = [args.mpiexec, "-n", len(devices) or 2, aero]
    distributed = ["main:mesh=grid.mdec"] + (["main:backend=cuda"] if devices else [])
    check_exact_report(aero, mpi, case, distributed)
    # Проверяем каждый путь реконструкции. На одной сетке сравниваем обычный CPU и MPI.
    variants = [
        ("first_order", "conservative", "lf", "euler"),
        ("first_order", "characteristic", "roe", "bdf1"),
        ("ebr3", "flux", "lf", "bdf2"),
        ("ebr5", "characteristic", "roe", "ssprk3"),
        ("weno3", "conservative", "lf", "ssprk3"),
        ("weno5", "characteristic", "roe", "ssprk3"),
        ("mp5", "flux", "roe", "ssprk3"),
        ("MPWENO5", "conservative", "roe", "ssprk3"),
    ]
    for i, (scheme, variables, flux, time) in enumerate(variants):
        options = [
            f"main:reconstruction={scheme}",
            f"main:reconstruction_variables={variables}",
            f"main:euler_solver={flux}",
            f"main:time_scheme={time}",
        ]
        if time == "bdf1":
            options += ["implicit:jacobian_flux=Roe", "implicit:simplified_jacobian=on"]
        reference, expected = solve([aero], case, case / f"serial{i}", options)
        meta, actual = solve(mpi, case, case / f"mpi{i}", options + distributed)
        if meta != reference or meta[4] != END or meta[3] < 2:
            raise AssertionError("Snapshot metadata or integration interval mismatch")
        analytic = analytic_error(actual, coords)
        if scheme == "weno3":
            coarse = case / "coarse.mesh"
            sparse_line(coarse, 102)
            _, coarse_fields = solve([aero], case, case / "coarse", options + ["main:mesh=coarse.mesh"])
            coarse_error = analytic_error(coarse_fields, [point[0] for point in nodes(coarse, 1)])
            if coarse_error <= 2 * analytic:
                raise AssertionError(f"WENO3 refinement: {coarse_error} -> {analytic}")
            print(f"OK WENO3 grid refinement: {coarse_error:.3g} -> {analytic:.3g}", flush=True)
        diff = max(abs(u - v) for k in expected for u, v in zip(actual[k], expected[k]))
        if not analytic < {"first_order": 0.001, "weno3": 0.0002}.get(scheme, 0.00012) or not diff < 2.0e-10:
            raise AssertionError(f"{scheme}/{variables}: analytic error {analytic}, MPI difference {diff}")
        print(
            f"OK linear {scheme}/{variables}/{flux}/{time}: analytic {analytic:.3g}, MPI {diff:.3g}",
            flush=True,
        )
    check_restart(aero, case)
    (case / "init.txt").write_text("initial_type formula\nrho=0\nux=0\np=0\n")
    meta, fields = solve(mpi, case, case / "zero", distributed + ["main:time_scheme=steady"])
    if meta[3] or any(v != 0 for col in fields.values() for v in col):
        raise AssertionError("Zero perturbations must remain exactly zero")
    check_wall(aero, mpi, distributed, case, coords)
    for dim in (2, 3):
        check_shear(args, aero, mpi, distributed, case, dim)


if __name__ == "__main__":
    parser = mpi_parser(__doc__)
    parser.add_argument("--devices", help="Two GPUs for the MPI run, for example 0,1")
    with mpi_workspace("linear-euler-") as tmp:
        check_modes(parser.parse_args(), Path(tmp))
