"""Проверка порядка конвекции SA и сравнение полей RANS/DDES на разных исполнителях."""
import math
import struct
import zlib
from pathlib import Path

from mpi_common import backend_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run, sparse_line
from run import nodes
from turbulence import configure
from viscosity import max_difference, solve


def smooth(x):
    a = 2*math.pi*x
    return .008*(1 + .22*math.sin(a) + .07*math.cos(2*a))


def derivative(x):
    a = 2*math.pi*x
    return .008*2*math.pi*(.22*math.cos(a) - .14*math.sin(2*a))


def seed_line(case, args, x, speed, name, profile=smooth):
    """В контрольной точке шага 0 заменяем только nu_tilde заданной аналитической функцией."""
    (case/"init.txt").write_text(f"initial_type=formula\nrho=1\np=1\nux={speed}\n")
    solve(case,args,name,options=["main:final_time=0"])
    data = bytearray((case/name/"recovery-0.bin").read_bytes())
    # Газ уже записан солвером. Для одномерной точки без BDF и средних запись
    # содержит два номера, по три компоненты текущего и начального газа, nu_tilde и CRC.
    assert data[:7] == b"AEROSAV" and (data[7]-ord("1")) & 5 == 0
    stride = 80
    for i in range(len(x)):
        start = 120 + i*stride
        node, representative = struct.unpack_from("<QQ",data,start)
        assert node == i
        struct.pack_into("<d",data,start+64,profile(x[representative]))
        struct.pack_into("<Q",data,start+72,zlib.crc32(data[start:start+72]))
    (case/(name+".bin")).write_bytes(data)


def transport_error(case,args,x,speed):
    """Проверяем реальный сдвиг профиля за несколько шагов SSPRK3."""
    # При таком масштабе поля и Re вклад физической диффузии за t=0.2
    # много меньше ошибки дискретизации. В однородном периодическом газе источник SA равен нулю.
    scale, end = 1e-7, .2
    seed_line(case,args,x,speed,"pulse",lambda pos: scale*smooth(pos))
    (case/"init.txt").write_text("initial_type=recovery\npath=pulse.bin\n")
    options = ["main:time_scheme=ssprk3","main:reconstruction=ebr3","main:final_time=.2",
               "gas:reynolds=1e12","turbulence:sa_high_order=on"]
    w = solve(case,args,"transport",options=options)["nu_tilde"]
    unique = [i for i,pos in enumerate(x) if pos < 1.]
    assert min(w) > 0.
    return math.sqrt(sum((w[i]/scale-smooth(x[i]-speed*end))**2 for i in unique)/len(unique))


def spatial_errors(case, args, cells, mpweno_only=False):
    case.mkdir()
    mesh = case/"grid.mesh"
    sparse_line(mesh,cells+1)
    x = [p[0] for p in nodes(mesh,1)]
    (case/"bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n")
    (case/"main.txt").write_text("mesh=grid.mesh\nbackend=cpu\nviscosity=normal\nTurbulence=On\n"
        "reconstruction=weno5\nreconstruction_variables=characteristic\neuler_solver=roe\n"
        "upw_coeff=1\ntime_scheme=euler\nfinal_time=.00001\ncfl=.4\n")
    (case/"gas.txt").write_text("gamma=1.4\nmach=.5\nreynolds=1000\nprandtl=.72\nt_amb=300\n")
    (case/"turbulence.txt").write_text("model=RANS\nrans_eq=SA\nnut_nu_amb=.21\n")
    (case/"output.txt").write_text("[field]\nfile=state.fields\nformat=binary\nlayout=global\n"
                                   "every_steps=1\nvariables=nu_tilde\n")
    speed, dt = -.65, .00001
    seed_line(case,args,x,speed,"moving")
    seed_line(case,args,x,0.,"still")
    init = case/"init.txt"
    init.write_text("initial_type=recovery\npath=still.bin\n")
    rest = solve(case,args,"diffusion")["nu_tilde"]
    init.write_text("initial_type=recovery\npath=moving.bin\n")
    errors = {}
    schemes = ("mpweno5",) if mpweno_only else ("off","ebr3","ebr5","weno3","weno5","mp5","mpweno5")
    for scheme in schemes:
        options = [] if scheme == "off" else [f"main:reconstruction={scheme}","turbulence:sa_high_order=on"]
        field = solve(case,args,scheme,options=options)["nu_tilde"]
        assert binary_snapshot(case/scheme/"state.fields")[0][3:] == (1,dt)
        # На одном шаге Эйлера из одинакового поля диффузия и источники совпадают.
        # Вычитание выделяет пространственную производную без ошибки интегрирования по времени.
        unique = [i for i,pos in enumerate(x) if pos < 1.]
        error = [(field[i]-rest[i])/dt + speed*derivative(x[i]) for i in unique]
        errors[scheme] = math.sqrt(sum(e*e for e in error)/cells)
        assert abs(sum(field[i]-rest[i] for i in unique)) < 1e-13
    if cells == 63:
        # Здесь газ постоянный, а поле SA неоднородно: проверяем собственные
        # нелинейные веса MPWENO5–MP6 и передачу одной поправки между процессами.
        options = ["main:reconstruction=mpweno5","scheme:mp6_hybrid=.6","turbulence:sa_high_order=on"]
        reference = solve(case,args,"hybrid",options=options)
        if args.gpu:
            (case/"cuda.txt").write_text("devices=0\n")
            gpu = solve(case,args,"hybrid-gpu",gpu=True,options=options)
            print(f"Scalar MPWENO5-MP6, GPU: {max_difference(reference,gpu):.3g}",flush=True)
            assert max_difference(reference,gpu) < 1e-13
        if args.mpi:
            run([args.meshdec.resolve(),mesh,case/"grid.mdec","--target-ranks=2","--compress"])
            mpi = solve(case,args,"hybrid-mpi",distributed=True,options=options)
            print(f"Scalar MPWENO5-MP6, MPI: {max_difference(reference,mpi):.3g}",flush=True)
            assert max_difference(reference,mpi) < 1e-13
        first = solve(case,args,"startup",options=options+["main:t_fo=.01"])
        assert max_difference(first,binary_snapshot(case/"off/state.fields")[1]) < 1e-15
    if not mpweno_only:
        errors["SSPRK3+EBR3"] = transport_error(case,args,x,speed)
    return errors


def spatial_order(work,args):
    sizes = (31,63,127)
    rows = [spatial_errors(work/f"grid-{n}",args,n) for n in sizes]
    for scheme in rows[0]:
        errors = [row[scheme] for row in rows]
        orders = [math.log(errors[i]/errors[i+1])/math.log(sizes[i+1]/sizes[i]) for i in range(2)]
        print(f"{scheme:8} L2 {', '.join(f'{v:.4g}' for v in errors)}; orders {orders[0]:.2f}, {orders[1]:.2f}",flush=True)
        # WENO-JS и ограничитель MPWENO чувствительны к гладким экстремумам профиля.
        minimum = {"off":.95, "weno3":1.4, "weno5":3.6, "mpweno5":4.0,
                   "ebr5":4.7, "mp5":4.7}.get(scheme,2.8)
        assert orders[-1] > minimum, (scheme,orders)
    # На грубых сетках меняется число граней, для которых датчик выбирает WENO.
    # Два дальнейших измельчения отдельно проверяют асимптотический порядок.
    fine = (255,511)
    errors = [spatial_errors(work/f"grid-{n}",args,n,mpweno_only=True)["mpweno5"] for n in fine]
    order = math.log(errors[0]/errors[1])/math.log(fine[1]/fine[0])
    print(f"MPWENO5, {fine[0]} -> {fine[1]} cells: L2 {errors[0]:.4g}, {errors[1]:.4g}, order {order:.2f}",flush=True)
    assert 4.7 < order < 5.3, (errors, order)


def compare_steps(reference, actual):
    """Сверяет по исходным номерам узлов каждое поле на каждом принятом шаге."""
    files = sorted(p.name for p in reference.glob("step_*.fields"))
    assert files and files == sorted(p.name for p in actual.glob("step_*.fields"))
    maximum = 0.
    for step,name in enumerate(files, start=1):
        a,x = binary_snapshot(reference/name)
        b,y = binary_snapshot(actual/name)
        assert a[3] == step and a[:4] == b[:4] and abs(a[4]-b[4]) < 1e-13, (name,a,b)
        difference = max_difference(x,y)
        assert difference < 2e-10, (actual.name,name,difference)
        maximum = max(maximum,difference)
    return maximum


# Проверяет согласованный перенос газа и рабочей вязкости SA.
def coupled(case,args):
    case.mkdir()
    configure(case,args,dimension=3)
    # Неоднородный RANS формирует поле SA на штатной периодической сетке.
    (case/"init.txt").write_text("initial_type=formula\nrho=1\np=2\nux=.4+.2*sin(2*pi*y)\nuy=.1*sin(2*pi*x)\nuz=0\n")
    seed = solve(case,args,"seed",options=["main:final_time=.01"])
    assert max(seed["nu_tilde"]) > min(seed["nu_tilde"])
    (case/"init.txt").write_text("initial_type=recovery\npath=seed/recovery-0.bin\n")
    output = case/"output.txt"
    schedule = output.read_text()
    # every_time ограничивает длину шагов и должен остаться тем же. Отдельная
    # зона записывает поля каждого шага для обоих вариантов гибрида.
    each_step = ("\n[each]\nfile=step.fields\nformat=binary\nlayout=global\nevery_steps=1\n"
                 "multi=1\nvariables=primary,turbulence\n")
    for model,scheme,integrator,hybrid in (("RANS","weno5","ssprk3",0),("DDES","mpweno5","bdf2",0),
                                          ("DDES","mpweno5","bdf2",.6),("DDES","mp5","bdf2",.6)):
        name = f"{model}-{scheme}-{integrator}-h{hybrid}"
        output.write_text(schedule + (each_step if hybrid else ""))
        settings = "model="+model+"\nrans_eq=SA\nnut_nu_amb=.21\n"
        if model=="DDES": settings += "subgridscale=vol\n"
        (case/"turbulence.txt").write_text(settings)
        options = [f"main:time_scheme={integrator}",f"main:reconstruction={scheme}","main:final_time=.01",
                   "main:upw_coeff=1","main:reconstruction_variables=characteristic","main:euler_solver=roe",
                   "scheme:stencil_geometry=ijk",f"scheme:mp6_hybrid={hybrid}"]
        low = solve(case,args,name+"-low",options=options)
        (case/"turbulence.txt").write_text(settings+"sa_high_order=on\n")
        high = solve(case,args,name,options=options)
        delta = max(abs(a-b) for a,b in zip(low["nu_tilde"],high["nu_tilde"]))
        assert delta > 1e-10, (name,delta)
        for gpu,mpi in ((True,False),(False,True),(True,True)):
            if (gpu and not args.gpu) or (mpi and not args.mpi): continue
            if gpu and mpi and len(args.devices.split(",")) < 2: continue
            label = name + ("-gpu" if gpu else "-cpu") + ("-mpi" if mpi else "")
            result = solve(case,args,label,gpu=gpu,distributed=mpi,options=options)
            diff = compare_steps(case/name,case/label) if hybrid else max_difference(high,result)
            print(f"{label}: max field difference {diff:.3g}, nu_tilde high/low {delta:.3g}",flush=True)
            assert diff < 2e-10


if __name__=="__main__":
    parser = backend_parser(__doc__)
    args = parser.parse_args()
    with mpi_workspace("sa-convection-") as work:
        spatial_order(Path(work),args)
        coupled(Path(work)/"coupled",args)
