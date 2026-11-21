"""Сверяет перенос SA, влияние турбулентной вязкости на газ и восстановление BDF2."""
import math
import shutil
from pathlib import Path

from mpi_common import mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run
from viscosity import setup, solve, compare_backends, max_difference
from run import nodes


def configure(case, args, dimension=2, decompose=True):
    coords = setup(case, args.meshdec.resolve() if args.mpi and decompose else None, dim=dimension)
    with (case / "main.txt").open("a") as main:
        main.write("Turbulence=On\n")
    (case / "turbulence.txt").write_text("model=RANS\nrans_eq=SA\nnut_nu_amb=.21\npr_t=.9\n")
    velocity = "ux,uy" if dimension == 2 else "ux,uy,uz"
    (case / "output.txt").write_text("[flow]\nfile=state.fields\nformat=binary\nlayout=global\n"
        f"every_time=.005\nvariables=rho,{velocity},p,E,nu_tilde,nut\n")
    return coords


def jet_from_rest(case, args):
    """Проверяет, что профиль free запускает струю и переносит SA в первый внутренний узел."""
    case.mkdir()
    coords = configure(case, args, dimension=3, decompose=False)
    shutil.copyfile(Path(__file__).parent / "fixtures" / "jet_profile.dat", case / "jet.dat")
    (case / "init.txt").write_text("initial_type=formula\nrho=1\np=.286301\nux=0\nuy=0\nuz=0\n")
    (case / "bc.txt").write_text(
        "[xmin]\nbc_type free\nprofile jet.dat\nprofile_center 0 .5 .5\nprofile_axis 1 0 0\n"
        "superin profile\ninflow profile\noutflow base_p profile\n"
        "[xmax,ymin,ymax,zmin,zmax]\nbc_type novisc_wall\n"
    )
    # Начальные скорость и рабочая вязкость нулевые. Давление в профиле совпадает
    # с начальным по всему срезу, поэтому скачок давления сам по себе струю не создаёт.
    options = ["main:final_time=.005", "main:upw_coeff=1", "gas:reynolds=1262900",
               "turbulence:nut_nu_amb=0"]
    axis = sorted((x, i) for i, (x, y, z) in enumerate(coords) if y == z == .5)
    (_, inlet), (_, inside) = axis[:2]
    reference = None
    for backend in (("cpu", "gpu") if args.gpu else ("cpu",)):
        field = solve(case, args, backend, gpu=backend == "gpu", options=options)
        for name in ("ux", "nu_tilde"):
            values = field[name]
            assert values[inlet] > values[inside] > 0., (backend, name, values[inlet], values[inside])
        if reference is not None:
            delta = max_difference(reference, field)
            assert delta < 1e-11, delta
            for i in (inlet, inside):
                assert math.isclose(reference["nu_tilde"][i], field["nu_tilde"][i], rel_tol=1e-9)
        reference = field
        print(f"OK jet from rest ({backend}), first interior node: "
              f"ux={field['ux'][inside]:.8g}, nu_tilde={field['nu_tilde'][inside]:.8g}", flush=True)


def shear(case, args):
    coords = configure(case, args)
    amplitude, reynolds, ratio = .1, 100., .21
    (case / "init.txt").write_text("initial_type=formula\nrho=1\nux=.1*sin(2*pi*y)\nuy=0\np=1/(1.4*.5^2)\n")
    common = ["gas:reynolds=100", "main:time_scheme=euler"]
    initial = solve(case,args,"initial",options=[*common,"main:final_time=0"])
    w0 = initial["nu_tilde"][0]
    assert max(abs(x-ratio/reynolds) for x in initial["nut"]) < 1.e-14
    # На одном коротком шаге Эйлера правая часть берётся в начальном поле.
    # Завихренность равна центральной разности, а вязкий поток ux — второй разности.
    dt = 1.e-5
    fields = solve(case,args,"one-step",options=[*common,f"main:final_time={dt}"])
    ny = len({y for _,y,_ in coords})-1
    frequency = ny*math.sin(2*math.pi/ny)
    laplace = (2*ny*math.sin(math.pi/ny))**2
    for i,(_,y,_) in enumerate(coords):
        omega = abs(amplitude*frequency*math.cos(2*math.pi*y))
        rate = .1355*(1.-1.2*math.exp(-.5*(w0*reynolds)**2))*omega*w0
        assert abs(fields["nu_tilde"][i]-(w0+dt*rate)) < 2.e-13
        expected = initial["ux"][i]*(1.-dt*laplace*(1.+ratio)/reynolds)
        assert abs(fields["ux"][i]-expected) < 2.e-13
    print("OK discrete SA production and turbulent shear stress",flush=True)

    for scheme in ("euler","ssprk3","bdf1","bdf2","steady"):
        opts=["gas:reynolds=100",f"main:time_scheme={scheme}","main:final_time=.01","implicit:newton_tol=1e-8"]
        if scheme in ("bdf1","bdf2","steady"):
            opts += ["implicit:simplified_newton=on"]
        if scheme=="steady":
            # Относительная невязка чуть ниже исходной после нескольких псевдошагов.
            opts += ["implicit:steady_tol=.9999"]
        ref=solve(case,args,"shear-"+scheme,options=opts)
        assert max(ref["nu_tilde"])>w0
        compare_backends(case,args,"shear-"+scheme,ref,opts)
        if scheme=="steady":
            local = opts + ["implicit:local_time_step=on", "main:report_every=1"]
            reference = solve(case, args, "shear-local", options=local)
            compare_backends(case, args, "shear-local", reference, local)
            # На вытянутых ячейках сравниваем предиктор CSGS с независимой
            # сборкой Ньютона при том же шаге, допуске и переносе SA.
            csgs = local + ["implicit:preconditioner=CSGS", "implicit:simplified_jacobian=on"]
            reference = solve(case, args, "steady-full-newton", options=csgs+["implicit:simplified_newton=off"])
            accelerated = solve(case, args, "steady-csgs", options=csgs)
            assert max_difference(reference, accelerated) < 2.e-10
            compare_backends(case, args, "steady-csgs", accelerated, csgs)
        if scheme=="bdf2":
            solve(case,args,"split",options=[*opts,"main:final_time=.005"])
            run([args.aero.resolve(),f"--input={case}",f"--output={case/'split'}","--recover",*opts])
            files=list((case/"split").glob("resumed-*/state.fields"))
            assert len(files)==1
            delta=max_difference(ref,binary_snapshot(files[0])[1])
            assert delta<2.e-12,delta
            print(f"OK BDF2 restores SA and its previous layer: maximum difference {delta:.3g}",flush=True)


def sigma(case, args):
    """Проверяет алгебраическую вязкость и её вклад в течение на трёхмерной сетке."""
    case.mkdir()
    coords = configure(case, args, 3)
    (case/"output.txt").write_text("[flow]\nfile=state.fields\nformat=binary\nlayout=global\n"
                                  "every_time=.0015\nvariables=primary,turbulence\n")
    (case/"turbulence.txt").write_text("model=LES\nles_eq=Sigma\nsubgridscale=vol\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1\np=1\n"
                                 "ux=.3*sin(2*pi*y)\nuy=.15*sin(2*pi*x)\nuz=.05*sin(2*pi*z)\n")
    # У начала координат у восьми соседних кубов одинаковый градиент.
    # Сингулярные числа в нём равны 0.3, 0.15, 0.05, умноженным на k.
    h = min(x for x,y,z in coords if x>0.)
    k = math.sin(2*math.pi*h)/h
    rate = k*.05*(.3-.15)*(.15-.05)/(.3**2)
    origin = coords.index((0.,0.,0.))
    initial = solve(case,args,"sigma-initial",options=["main:final_time=0"])
    assert abs(initial["nut"][origin]-(1.35*h)**2*rate) < 1e-14
    # Во всём этом поле ось вихря параллельна z, поэтому VTM=0. Проверяем,
    # что масштаб с нижним пределом 0.2 даёт именно множитель 0.2² у вязкости.
    zero_step = ["main:final_time=0"]
    base = solve(case,args,"tilde-initial",options=zero_step+["turbulence:subgridscale=Delta_omega^tilde"])
    limited = solve(case,args,"sla-initial",options=zero_step+["turbulence:subgridscale=sla","turbulence:F_KH_lim=.2"])
    assert max(abs(.04*a-b) for a,b in zip(base["nut"],limited["nut"])) < 1e-14
    implicit = ["main:time_scheme=bdf2", "main:final_time=.003"]
    fields = {}
    for scale in ("max", "min", "vol", "Delta_omega", "Delta_omega_tilde", "sla"):
        opts = [*implicit, f"turbulence:subgridscale={scale}"]
        name = "sigma-" + scale
        field = solve(case,args,name,options=opts)
        assert "nu_tilde" not in field and min(field["nut"]) >= 0. and max(field["nut"]) > 0.
        compare_backends(case,args,name,field,opts)
        assert binary_snapshot(case/name/"state.fields")[0][3] >= 2
        fields[scale] = field
    opts = [*implicit, "turbulence:subgridscale=sla"]
    # При нижнем пределе 1 или отключённом VTM поле обязано совпасть с базовой Δ̃ω.
    reference = fields["Delta_omega_tilde"]
    for option in ("switch_on_vtm=off", "F_KH_lim=1"):
        field = solve(case,args,"sigma-"+option.split("=")[0],options=opts+["turbulence:"+option])
        assert max_difference(reference,field) == 0.
    assert max_difference(reference,fields["sla"]) > 1.e-8
    # В recovery LES достаточно газовых полей и истории BDF2. После чтения
    # проверяем полное поле и заново вычисленную подсеточную вязкость.
    solve(case,args,"sigma-resume",options=opts+["main:final_time=.0015"])
    dest=case/"sigma-resume"
    run([args.aero.resolve(),f"--input={case}",f"--output={dest}","--recover",*opts])
    resumed=binary_snapshot(next(dest.glob("resumed-*/state.fields")))[1]
    assert max_difference(fields["sla"],resumed)<1.e-12
    # Проверяем физическое действие на импульс, отдельно от файла с самой nut.
    (case/"output.txt").write_text((case/"output.txt").read_text().replace("primary,turbulence","primary"))
    options = ["main:final_time=.003", "main:time_scheme=ssprk3"]
    laminar = solve(case,args,"laminar",options=options+["main:Turbulence=Off"])
    les = solve(case,args,"sigma-explicit",options=options)
    assert max(abs(a-b) for a,b in zip(laminar["ux"],les["ux"])) > 1.e-10
    compare_backends(case,args,"sigma-explicit",les,options)
    (case/"bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
                               "[zmin]\nbc_type periodic\npair zmax\ntranslation 0 0 1\n"
                               "[ymin]\nbc_type adiabatic_wall\n[ymax]\nbc_type adiabatic_wall\n")
    schedule = case/"output.txt"
    schedule.write_text(schedule.read_text().replace("variables=primary","variables=primary,nut"))
    wall = solve(case,args,"sigma-wall",options=options)
    assert max(wall["nut"]) > 0. and all(wall["nut"][i]==0. for i,(_,y,_) in enumerate(coords) if y in (0.,1.))
    compare_backends(case,args,"sigma-wall",wall,options)
    print("OK LES Sigma, six scales, SLA switch, wall and viscous flux",flush=True)


def ddes(case,args):
    """Независимо проверяет сеточные масштабы по источнику SA, затем перенос и recovery."""
    case.mkdir()
    configure(case,args,3,decompose=False)
    # Вытягиваем куб по y и сжимаем по z, чтобы min, max, vol и проекции различались.
    mesh=case/"grid.mesh"
    lines=mesh.read_text().splitlines()
    for axis,multiplier in (("Y",2.),("Z",.5)):
        row=lines.index(axis)+1
        lines[row]=" ".join(str(float(x)*multiplier) for x in lines[row].split())
    mesh.write_text("\n".join(lines)+"\n")
    bc=case/"bc.txt"
    bc.write_text(bc.read_text().replace("translation 0 1 0","translation 0 2 0")
                                .replace("translation 0 0 1","translation 0 0 .5"))
    if args.mpi:
        run([args.meshdec.resolve(),mesh,case/"grid.mdec","--target-ranks=2","--compress"])
    (case/"turbulence.txt").write_text("model=DDES\nrans_eq=SA\nnut_nu_amb=.21\nsubgridscale=max\n")
    (case/"output.txt").write_text("[flow]\nfile=state.fields\nformat=binary\nlayout=global\n"
                                  "every_time=.0015\nvariables=primary,turbulence\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1\np=1/(1.4*.5^2)\n"
                                 "ux=.3*sin(pi*y)\nuy=.14*sin(2*pi*x)\nuz=.05*sin(4*pi*z)\n")
    coords=nodes(mesh,3)
    origin=coords.index((0.,0.,0.))
    h=min(x for x,y,z in coords if x>0.)
    # На начале координат ось вихря вертикальна, VTM нулевой. Начальная рабочая
    # вязкость однородна: в первом шаге Эйлера остаётся только известный источник.
    dt=1.e-5
    common=["main:time_scheme=euler","gas:reynolds=1000"]
    initial=solve(case,args,"ddes-start",options=common+["main:final_time=0"])
    w=initial["nu_tilde"][origin]
    nu=.001
    ft2=1.2*math.exp(-.5*(w/nu)**2)
    fv2=1.-(w/nu)/(1.+(w/nu)*(.21*nu/w))
    omega=abs(.14-.3/2.)*math.sin(2*math.pi*h)/h
    def one_step(model_length,rotation):
        s=w/(.41*model_length)**2
        shear=max(.3*rotation,rotation+s*fv2)
        r=min(10.,s/shear) if shear>0. else 10.
        g=.7*r+.3*r**6
        fw=g*(65./(64.+g**6))**(1./6.)
        cw1=.1355/.41**2+(1.+.622)/(2./3.)
        source=.1355*(1.-ft2)*shear-(cw1*fw*.41**2-.1355*ft2)*s
        return w*(1.+dt*source)

    projected=h*math.sqrt(5./3.)
    for scale,length in (("min",.5*h),("max",2.*h),("vol",h),("Delta_omega",math.sqrt(2.)*h),
                         ("Delta_omega_tilde",projected),("sla",.1*projected)):
        opts=common+[f"turbulence:subgridscale={scale}",f"main:final_time={dt}"]
        result=solve(case,args,"ddes-"+scale,options=opts)
        assert abs(result["nu_tilde"][origin]-one_step(.65*length,omega))<3.e-14,scale
        compare_backends(case,args,"ddes-"+scale,result,opts)

    options=["main:time_scheme=bdf2","gas:reynolds=1000","turbulence:subgridscale=sla","main:final_time=.003"]
    full=solve(case,args,"ddes-bdf2",options=options)
    compare_backends(case,args,"ddes-bdf2",full,options)
    dest=case/"ddes-restart"
    solve(case,args,dest.name,options=options+["main:final_time=.0015"])
    run([args.aero.resolve(),f"--input={case}",f"--output={dest}","--recover",*options])
    assert max_difference(full,binary_snapshot(next(dest.glob("resumed-*/state.fields")))[1])<2.e-12

    # При F_KH=0 в однородном потоке без стенок сеточная длина равна нулю.
    initial_file=case/"init.txt"
    initial_text=initial_file.read_text()
    initial_file.write_text("initial_type=formula\nrho=1\np=1/(1.4*.5^2)\nux=0\nuy=0\nuz=0\n")
    opts=common+[f"main:final_time={dt}","turbulence:subgridscale=sla","turbulence:F_KH_lim=0"]
    zero=solve(case,args,"ddes-zero-scale",options=opts)
    assert max(zero["nu_tilde"])==0. and max(zero["nut"])==0.
    initial_file.write_text(initial_text)

    # При достаточно большом C_DES физическое расстояние короче сеточного у всех узлов.
    # Должны получить ровно исходную SA, включая нулевое значение на стенках.
    bc.write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
                  "[zmin]\nbc_type periodic\npair zmax\ntranslation 0 0 .5\n"
                  "[ymin]\nbc_type adiabatic_wall\n[ymax]\nbc_type adiabatic_wall\n")
    options=["main:time_scheme=ssprk3","main:final_time=.003","gas:reynolds=1000"]
    original=solve(case,args,"ddes-wall-sa",options=options+["turbulence:model=RANS"])
    opts=options+["turbulence:c_des=1000"]
    shielded=solve(case,args,"ddes-wall",options=opts)
    assert max_difference(original,shielded)<1.e-14
    assert all(shielded["nu_tilde"][i]==0. for i,(_,y,_) in enumerate(coords) if y in (0.,2.))
    compare_backends(case,args,"ddes-wall",shielded,opts)
    # Чистое растяжение посередине канала: ротор равен нулю, полный градиент конечен.
    # Подбираем амплитуду для промежуточной защиты, где обе длины влияют на источник.
    initial_file.write_text("initial_type=formula\nrho=1\np=1/(1.4*.5^2)\nux=0\nuy=0\nuz=.005*sin(4*pi*z)\n")
    opts=common+[f"main:final_time={dt}","turbulence:subgridscale=vol"]
    strain=solve(case,args,"ddes-strain",options=opts)
    center=coords.index((.5,1.,0.))
    gradient=.005*math.sin(2*math.pi*h)/(.5*h)
    rd=(1.+.21)*nu/(.41**2*gradient)
    protected=math.tanh((8.*rd)**3)
    assert .2<protected<.9
    length=.65*h+protected*(1.-.65*h)
    assert abs(strain["nu_tilde"][center]-one_step(length,0.))<3.e-14
    compare_backends(case,args,"ddes-strain",strain,opts)
    print("OK DDES six filter lengths, SA source, BDF2 restart, wall and partial shielding",flush=True)


def ddes_warped(case,args):
    """Сверяет перенос на косых гексаэдрах с ненулевыми диагональными связями AES."""
    case.mkdir()
    coords=configure(case,args,3,decompose=False)
    mesh=case/"grid.mesh"
    lines=mesh.read_text().splitlines()
    # Смещение равно нулю на парных плоскостях соответствующей координаты.
    # Поэтому трансляции периодичности и твёрдые стенки остаются исходными.
    warped=[]
    for x,y,z in coords:
        sx,sy,sz=(0. if v in (0.,1.) else math.sin(2*math.pi*v) for v in (x,y,z))
        warped.append((x+.018*sx*sy*sz,y+.014*sy*sz,z+.016*sz*sx))
    for d,axis in enumerate("XYZ"):
        lines[lines.index(axis)+1]=" ".join(format(p[d],".17g") for p in warped)
    mesh.write_text("\n".join(lines)+"\n")
    if args.mpi:
        run([args.meshdec.resolve(),mesh,case/"grid.mdec","--target-ranks=2","--compress"])
    (case/"bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
        "[zmin]\nbc_type periodic\npair zmax\ntranslation 0 0 1\n"
        "[ymin]\nbc_type adiabatic_wall\n[ymax]\nbc_type adiabatic_wall\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1+.12*sin(2*pi*x)\n"
        "ux=.07*sin(2*pi*y)\nuy=.05*cos(2*pi*z)\nuz=.06*cos(2*pi*x)\np=1.6+.17*cos(2*pi*(x+y+z))\n")
    (case/"turbulence.txt").write_text("model=DDES\nrans_eq=SA\nnut_nu_amb=.21\nsubgridscale=sla\n")
    for scheme in ("ssprk3","bdf2"):
        name="warped-"+scheme
        opts=["gas:reynolds=1000","main:final_time=.015","main:time_scheme="+scheme]
        result=solve(case,args,name,options=opts)
        assert min(result["nu_tilde"])==0. and max(result["nut"])>0.
        compare_backends(case,args,name,result,opts)
    print("OK DDES on warped hexahedra with adiabatic walls",flush=True)


def ddes_steady(case,args):
    """На плоском сдвиге проверяем BDF1 и локальный псевдошаг с SA-DDES."""
    case.mkdir()
    configure(case,args,2)
    (case/"turbulence.txt").write_text("model=DDES\nrans_eq=SA\nnut_nu_amb=.21\nsubgridscale=Delta_omega\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1\nux=.1*sin(2*pi*y)\nuy=0\np=1/(1.4*.5^2)\n")
    common=["gas:reynolds=100","main:final_time=.01","implicit:simplified_newton=on"]
    for scheme in ("bdf1","steady"):
        opts=common+[f"main:time_scheme={scheme}"]
        if scheme=="steady":
            opts+=["implicit:local_time_step=on","implicit:steady_tol=.9999"]
        name="ddes-"+scheme
        ref=solve(case,args,name,options=opts)
        assert binary_snapshot(case/name/"state.fields")[0][3]>0
        assert min(ref["nu_tilde"])>0.
        compare_backends(case,args,name,ref,opts)


def planar_sigma(case,args):
    """В двумерном течении третье сингулярное число равно нулю на всей сетке."""
    case.mkdir()
    configure(case,args,2)
    (case/"turbulence.txt").write_text("model=LES\nles_eq=Sigma\nsubgridscale=sla\n")
    (case/"output.txt").write_text("[flow]\nfile=state.fields\nformat=binary\nlayout=global\n"
                                  "every_steps=1\nvariables=primary,nut\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1\np=1\nux=.2*sin(2*pi*y)\nuy=.1*sin(2*pi*x)\n")
    ref=solve(case,args,"planar",options=["main:final_time=.003"])
    assert all(value==0. for value in ref["nut"])
    compare_backends(case,args,"planar",ref,["main:final_time=.003"])
    # При σ3=0 все газовые поля должны в точности совпасть с ламинарным расчётом.
    schedule=case/"output.txt"
    schedule.write_text(schedule.read_text().replace("primary,nut","primary"))
    laminar=solve(case,args,"planar-laminar",options=["main:final_time=.003","main:Turbulence=Off"])
    assert all(laminar[k]==ref[k] for k in laminar)


def walls(case,args):
    case.mkdir()
    coords=configure(case,args,3)
    (case/"bc.txt").write_text("[xmin]\nbc_type periodic\npair xmax\ntranslation 1 0 0\n"
        "[zmin]\nbc_type periodic\npair zmax\ntranslation 0 0 1\n[ymin]\nbc_type adiabatic_wall\n[ymax]\nbc_type adiabatic_wall\n")
    (case/"init.txt").write_text("initial_type=formula\nrho=1\nux=0\nuy=0\nuz=0\np=2\n")
    opts=["main:time_scheme=bdf2","gas:reynolds=100","main:final_time=.01"]
    initial=solve(case,args,"initial",options=[*opts,"main:final_time=0"])
    ref=solve(case,args,"channel",options=opts)
    inside=[]
    for i,(_,y,_) in enumerate(coords):
        if y in (0.,1.):
            assert ref["nu_tilde"][i]==0. and ref["nut"][i]==0.
        else:
            inside.append(i)
            assert 0.<ref["nu_tilde"][i]<=initial["nu_tilde"][i]
    assert any(ref["nu_tilde"][i]<initial["nu_tilde"][i] for i in inside)
    compare_backends(case,args,"channel",ref,opts)
    print("OK 3D SA wall condition and interior decay",flush=True)
    # Большой шаг в покоящемся газе исчерпывает лимит внутренних поправок SA.
    # Для проверки нужен целый шаг dt=1: промежуточный вывод по времени делил бы его на части.
    output = case / "output.txt"
    schedule = output.read_text()
    output.write_text(schedule.replace("every_time=.005", "every_steps=1"))
    capped = ["main:time_scheme=bdf1", "main:final_time=1", "main:cfl=1e6",
              "gas:reynolds=1", "implicit:newton_tol=1e-12"]
    field = solve(case,args,"capped-sa",options=capped)
    meta, _ = binary_snapshot(case / "capped-sa/state.fields")
    assert meta[3:] == (1, 1.) and all(math.isfinite(w) and w >= 0. for w in field["nu_tilde"])
    assert max(field["nu_tilde"]) > 0.
    compare_backends(case,args,"capped-sa",field,capped)
    output.write_text(schedule)
    # Та же сетка со скольжением должна сохранять касательную скорость и фоновую SA.
    bc=case/"bc.txt"
    bc.write_text(bc.read_text().replace("adiabatic_wall","novisc_wall"))
    (case/"init.txt").write_text("initial_type=freestream\nphi=0\ntheta=0\n")
    slip_opts=[*opts,"main:time_scheme=ssprk3"]
    slip=solve(case,args,"slip",options=slip_opts)
    assert max(abs(v-1.) for v in slip["ux"])<2.e-13
    assert max(abs(v-.21/100.) for v in slip["nut"])<2.e-14
    compare_backends(case,args,"slip",slip,slip_opts)
    uniform=solve(case,args,"slip-uniform-bdf2",options=opts)
    assert max_difference(slip,uniform)<2.e-13
    compare_backends(case,args,"slip-uniform-bdf2",uniform,opts)
    # У неявного теста есть физическая невязка: касательная сдвиговая мода затухает.
    (case/"init.txt").write_text("initial_type=formula\nrho=1\nux=1+.1*cos(2*pi*y)\nuy=0\nuz=0\np=1/(1.4*.5^2)\n")
    implicit=solve(case,args,"slip-bdf2",options=opts)
    assert 1.<max(implicit["ux"])<1.1
    assert min(implicit["ux"])>.89
    compare_backends(case,args,"slip-bdf2",implicit,opts)
    print("OK insulated slip wall preserves tangential velocity and ambient SA",flush=True)


if __name__=="__main__":
    parser=mpi_parser(__doc__)
    parser.add_argument("--mpi",action="store_true")
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--devices",default="0")
    parser.add_argument("--jet-only", action="store_true", help="Run only the jet startup check")
    args=parser.parse_args()
    with mpi_workspace("turbulence-") as work:
        jet_from_rest(Path(work)/"jet", args)
        if args.jet_only:
            raise SystemExit(0)
        ddes(Path(work)/"ddes",args)
        ddes_steady(Path(work)/"ddes-steady",args)
        ddes_warped(Path(work)/"ddes-warped",args)
        sigma(Path(work)/"sigma",args)
        planar_sigma(Path(work)/"planar",args)
        shear(Path(work),args)
        walls(Path(work)/"walls",args)
