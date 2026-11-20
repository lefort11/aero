"""Сверяет перенос SA, влияние турбулентной вязкости на газ и восстановление BDF2."""
import math
from pathlib import Path

from mpi_common import mpi_parser, mpi_workspace
from mpi_fields import binary_snapshot
from mpi_geometry import run
from viscosity import setup, solve, compare_backends, max_difference


def configure(case, args, dimension=2):
    coords = setup(case, args.meshdec.resolve() if args.mpi else None, dim=dimension)
    with (case / "main.txt").open("a") as main:
        main.write("Turbulence=On\n")
    (case / "turbulence.txt").write_text("model=RANS\nrans_eq=SA\nnut_nu_amb=.21\npr_t=.9\n")
    velocity = "ux,uy" if dimension == 2 else "ux,uy,uz"
    (case / "output.txt").write_text("[flow]\nfile=state.fields\nformat=binary\nlayout=global\n"
        f"every_time=.005\nvariables=rho,{velocity},p,E,nu_tilde,nut\n")
    return coords


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
        if scheme=="bdf2":
            solve(case,args,"split",options=[*opts,"main:final_time=.005"])
            run([args.aero.resolve(),f"--input={case}",f"--output={case/'split'}","--recover",*opts])
            files=list((case/"split").glob("resumed-*/state.fields"))
            assert len(files)==1
            delta=max_difference(ref,binary_snapshot(files[0])[1])
            assert delta<2.e-12,delta
            print(f"OK BDF2 restores SA and its previous layer: maximum difference {delta:.3g}",flush=True)


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
    args=parser.parse_args()
    with mpi_workspace("turbulence-") as work:
        shear(Path(work),args)
        walls(Path(work)/"walls",args)
