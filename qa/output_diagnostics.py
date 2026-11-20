"""Проверяет пульсации, dp/dt, ротор, выборку Cp и продолжение истории на другой раскладке."""
import math
import re
from pathlib import Path

from mpi_common import mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run
from viscosity import setup, max_difference


def check_pressure_rhs(coords, fields, linear):
    """Независимый баланс по снимку на прямоугольной сетке: первый порядок, LF с upw_coeff=0."""
    xs, ys = (sorted({point[axis] for point in coords}) for axis in (0, 1))
    dx, dy = xs[1]-xs[0], ys[1]-ys[0]
    nodes = {(x, y): i for i, (x, y, _) in enumerate(coords)}
    def flux(i, axis):
        rho, u, v, p, energy = (fields[k][i] for k in ("rho", "ux", "uy", "p", "E"))
        un = (u, v)[axis]
        if linear:
            # Фон (0.4, 0), gamma*p0=1.4. Нужен только поток уравнения давления.
            return [(.4 if axis == 0 else 0.)*p + 1.4*un]
        return [rho*un, rho*u*un + (p if axis == 0 else 0.),
                rho*v*un + (p if axis == 1 else 0.), (energy+p)*un]

    error = 0.
    for i, (x, y, _) in enumerate(coords):
        ix, iy = xs.index(x), ys.index(y)
        # Концы по x отождествлены периодичностью. Для внутренних граней
        # среднее узловых потоков даёт обычную центральную разность.
        nx = len(xs)-1
        left = flux(nodes[xs[(ix-1) % nx], y], 0)
        right = flux(nodes[xs[(ix+1) % nx], y], 0)
        rhs = [(l-r)/(2*dx) for l, r in zip(left, right)]
        here = flux(i, 1)
        faces = []
        for j in (iy-1, iy+1):
            if 0 <= j < len(ys):
                neighbor = flux(nodes[x, ys[j]], 1)
                faces.append([.5*(a+b) for a, b in zip(neighbor, here)])
            else:
                # Через твёрдую стенку передаётся только нормальная сила давления.
                faces.append([0.] if linear else [0., 0., fields["p"][i], 0.])
        height = dy/2 if iy in (0, len(ys)-1) else dy
        rhs = [a + (lo-hi)/height for a, lo, hi in zip(rhs, *faces)]
        if linear:
            expected = rhs[0]
        else:
            u, v = fields["ux"][i], fields["uy"][i]
            expected = .4*(rhs[3] - u*rhs[1] - v*rhs[2] + .5*(u*u+v*v)*rhs[0])
        actual = fields["dp_dt"][i]
        assert math.isfinite(actual) and abs(actual-expected) < 2.e-11, (i, actual, expected)
        error = max(error, abs(actual-expected))
    return error


def check(case, args, linear=False):
    coords = setup(case, args.meshdec.resolve() if args.mpi else None)
    (case / "bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
                                 "[ymin]\nbc_type novisc_wall\n[ymax]\nbc_type adiabatic_wall\n")
    if linear:
        with (case / "main.txt").open("a") as main:
            main.write("Equations=Linear\n")
        (case / "linear.txt").write_text("rho0=1.2\np0=1\nux0=.4\nuy0=0\n")
        initial_gas = "rho=.01*cos(2*pi*x)\np=.01*cos(2*pi*x)\n"
        reference_pressure, dynamic_pressure = 0., .5*1.2*.4**2
    else:
        initial_gas = "rho=1\np=2+.1*cos(2*pi*x)\n"
        reference_pressure, dynamic_pressure = 1/(1.4*.5**2), .5
    (case / "init.txt").write_text("initial_type=formula\nux=.1*y\nuy=0\n"+initial_gas)
    (case / "output.txt").write_text(
        "[flow]\nfile=state.fields\nformat=binary\nlayout=global\nevery_steps=1\nmulti=1\n"
        "variables=primary,fluctuations,dp/dt,vorticity\n"
        "[surface]\nfile=wall.fields\nformat=binary\nlayout=global\nevery_steps=1\nvariables=Cp\nwall_label=ymin\n")
    def solve(folder, end, gpu=False, mpi=False, recover=False, scheme='ssprk3', viscosity='off'):
        command=[args.aero.resolve(), f"--input={case}", f"--output={folder}",f"main:final_time={end}",
                 f"main:viscosity={viscosity}",f"main:time_scheme={scheme}","main:cfl=.45","main:mpi_timing=1"]
        if mpi:
            command=[args.mpiexec,"-n","2",*command,"main:mesh=grid.mdec"]
        if gpu:
            command += ["main:backend=cuda","cuda:devices="+(args.devices if mpi else args.devices.split(",")[0])]
        if recover:
            command += ["--recover"]
        log = run(command)
        assert "Active backend = " + ("CUDA" if gpu else "CPU") in log
        steps = int(re.search(r"Completed (\d+) steps", log)[1])
        calls = int(re.search(r"(?m)^flux_compute\s+(\d+)", log)[1])
        if scheme in ("euler", "ssprk3") and not recover:
            # Единственная дополнительная сборка нужна на самом последнем слое.
            stages = 1 if scheme == "euler" else 3
            assert calls == stages*steps+1, (scheme, steps, calls)
        flux_counts[folder] = calls
        path=next(folder.glob("resumed-*")) if recover else folder
        return [binary_snapshot(f) for f in sorted(path.glob("state_*.fields"))],binary_snapshot(path/"wall.fields")

    flux_counts = {}
    initial, _ = solve(case/"initial",0)
    ref, surface = solve(case/"complete",.033)
    assert len(ref)>2
    old=initial[0][1]
    assert max(abs(value) for value in old["dp_dt"]) > .01
    rhs_error = check_pressure_rhs(coords, old, linear)
    for i,(_,y,_) in enumerate(coords):
        if 0.<y<1.:
            assert abs(old["vorticity"][i]-.1)<1.e-13
    quantities=("rho","ux","uy","p","E")
    integrals={q:[0.]*len(coords) for q in quantities}
    start=0.
    for meta, fields in ref:
        t=meta[4]
        rhs_error = max(rhs_error, check_pressure_rhs(coords, fields, linear))
        for q in quantities:
            for i,value in enumerate(fields[q]):
                integrals[q][i]+=.5*(old[q][i]+value)*(t-start)
                expected=value-integrals[q][i]/t
                assert abs(fields[q+"_prime"][i]-expected)<1.e-13
        old,start=fields,t
    wall_ids=[i for i,(_,y,_) in enumerate(coords) if y==0.]
    points=surface[1]
    assert list(points["node_id"])==wall_ids
    for j,i in enumerate(wall_ids):
        assert tuple(points[k][j] for k in ("x","y","z"))==tuple(coords[i])
        assert abs(points["Cp"][j]-(old["p"][i]-reference_pressure)/dynamic_pressure)<1.e-13
    model = "Linear" if linear else "Euler"
    print(f"{model}: maximum dp_dt difference from discrete RHS {rhs_error:.3g}", flush=True)

    two_gpus = len(args.devices.split(",")) >= 2
    for gpu,mpi in ((False,True),(True,False),(True,True)):
        if (gpu and not args.gpu) or (mpi and not args.mpi): continue
        if gpu and mpi and not two_gpus: continue
        records, wall=solve(case/f"full-{gpu}-{mpi}",.033,gpu,mpi)
        assert len(records)==len(ref)
        dp_error = 0.
        for (meta,a),(other,b) in zip(ref,records):
            assert meta[:4]==other[:4] and abs(meta[4]-other[4])<1.e-14
            assert max_difference(a,b)<2.e-10
            dp_error = max(dp_error, max(abs(l-r) for l, r in zip(a["dp_dt"], b["dp_dt"])))
        print(f"{model}, GPU={gpu}, MPI={mpi}: maximum dp_dt difference from CPU {dp_error:.3g}", flush=True)
        assert wall[0][:4]==surface[0][:4] and max_difference(wall[1],points)<1.e-12
    # Останавливаемся в уже принятой точке, чтобы длины шагов совпали с непрерывным расчётом.
    split=case/"split"
    solve(split,ref[1][0][4])
    records,wall=solve(split,.033,args.gpu and (not args.mpi or two_gpus),args.mpi,True)
    assert max_difference(records[-1][1],old)<2.e-10
    assert max_difference(wall[1],points)<2.e-12
    print("OK all saved derived fields and resumed time averages",flush=True)

    # Проверяем также одиночный столбец без истории средних. На одинаковых шагах
    # BDF имеет готовые потоки принятого состояния, в том числе для последнего снимка.
    output = case / "output.txt"
    config = output.read_text()
    for scheme in ("euler", "bdf1", "bdf2"):
        by_backend = []
        for gpu in ([False, True] if args.gpu else [False]):
            target = case / f"{scheme}-{gpu}"
            output.write_text(config.replace("primary,fluctuations,dp/dt,vorticity", "primary,dp_dt"))
            records, _ = solve(target, .012, gpu=gpu, scheme=scheme)
            for _, fields in records:
                check_pressure_rhs(coords, fields, linear)
            by_backend.append(records)
            if scheme.startswith("bdf"):
                plain = case / f"{scheme}-{gpu}-plain"
                output.write_text(config.replace("primary,fluctuations,dp/dt,vorticity", "primary"))
                baseline, _ = solve(plain, .012, gpu=gpu, scheme=scheme)
                assert flux_counts[target] == flux_counts[plain]
                assert len(records) == len(baseline)
                for (meta, fields), (other, reference) in zip(records, baseline):
                    assert meta == other and all(fields[k] == v for k, v in reference.items())
        if len(by_backend) == 2:
            cpu, gpu_records = by_backend
            assert len(cpu) == len(gpu_records)
            for (_, a), (_, b) in zip(cpu, gpu_records):
                assert max_difference(a, b) < 2.e-10
        print(f"OK {model} {scheme}: pressure RHS and flux reuse", flush=True)

    if not linear:
        # Газ в начальный момент покоится. Поток энергии вызван только теплопроводностью,
        # а производная кинетической энергии равна нулю. Один шаг явного Эйлера
        # даёт независимую проверку dp/dt(t=0) через приращение полной энергии.
        (case / "init.txt").write_text("initial_type=formula\nrho=1\nux=0\nuy=0\np=2+.1*cos(2*pi*x)\n")
        output.write_text(config.replace("primary,fluctuations,dp/dt,vorticity", "primary,dp_dt"))
        dt = .0001
        for gpu in ([False, True] if args.gpu else [False]):
            def heat_snapshot(end):
                records, _ = solve(case / f"heat-{gpu}-{end}", end, gpu=gpu, scheme="euler", viscosity="normal")
                assert len(records) == 1 and records[0][0][3] == int(end > 0)
                return records[0][1]
            start = heat_snapshot(0)
            end = heat_snapshot(dt)
            rates = start["dp_dt"]
            assert max(map(abs, rates)) > .01
            for rate, e0, e1 in zip(rates, start["E"], end["E"]):
                assert abs(rate - .4*(e1-e0)/dt) < 2.e-11
            print(f"OK Euler, GPU={gpu}: initial dp_dt from viscous heat flux", flush=True)


if __name__=="__main__":
    parser=mpi_parser(__doc__)
    parser.add_argument("--mpi",action="store_true")
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--devices",default="0,1")
    args=parser.parse_args()
    with mpi_workspace("output-diagnostics-") as directory:
        check(Path(directory),args)
        linear_case = Path(directory) / "linear"
        linear_case.mkdir()
        check(linear_case,args,linear=True)
