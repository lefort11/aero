"""Сравнение реконструкций на кольцевой сетке с точным решением уравнений Эйлера."""

import math
from pathlib import Path

from mpi_common import mpi_parser, mpi_workspace
from mpi_geometry import run
from mpi_fields import binary_snapshot


def annulus(path, angles, layers, planes=1):
    """Кольцо из четырёхугольников в 2D или призматические слои от цилиндра в 3D."""
    dimension = 2 if planes == 1 else 3
    stride = angles * planes
    coordinates = [[], [], []]
    faces, cells, signs, boundary, patches = [], [], [], [], []
    face_offsets, cell_offsets = [0], [0]
    ids = {}
    for layer in range(layers + 1):
        radius = 1 + .2 * layer / layers
        for plane in range(planes):
            for i in range(angles):
                theta = 2 * math.pi * i / angles
                z = .06 * plane
                if 0 < plane < planes - 1:
                    z += .007 * math.sin(2 * theta)
                for axis, value in zip(coordinates, (radius * math.cos(theta), radius * math.sin(theta), z)):
                    axis.append(value)

    def cell(shell):
        for face in shell:
            key = tuple(sorted(face))
            if key in ids:
                cells.append(ids[key])
                signs.append(-1)
                continue
            index = len(ids)
            ids[key] = index
            cells.append(index)
            signs.append(1)
            faces.extend(face)
            face_offsets.append(len(faces))
            inner = max(face) < stride
            outer = min(face) >= layers * stride
            end = planes > 1 and any(all((n % stride) // angles == k for n in face) for k in (0, planes-1))
            if inner or outer or end:
                boundary.append(index)
                patches.append(0 if inner else 1 if outer else 2)
        cell_offsets.append(len(cells))

    for layer in range(layers):
        for k in range(max(1, planes - 1)):
            for i in range(angles):
                a = layer * stride + k * angles + i
                b = layer * stride + k * angles + (i + 1) % angles
                if planes == 1:
                    corners = (a, a+stride, b+stride, b)
                    cell(zip(corners, corners[1:]+corners[:1]))
                else:
                    for l, m, n in ((a, b, b+angles), (a, b+angles, a+angles)):
                        cell(((l,n,m), (l+stride,m+stride,n+stride), (l,m,m+stride,l+stride),
                              (m,n,n+stride,m+stride), (n,l,l+stride,n+stride)))

    def row(name, values):
        return name + " " + " ".join(map(str, values)) + "\n"
    labels = ['"inner"', '"outer"'] + (['"ends"'] if planes > 1 else [])
    path.write_text("AEROMESH 2 TEXT\n" + row("DIMENSION", [dimension])
                    + row("COUNTS", (len(coordinates[0]), len(ids), len(faces), len(cell_offsets)-1,
                                     len(cells), len(boundary), len(labels)))
                    + "LENGTH_SCALE_TO_METERS 1\n" + "".join(row(k,v) for k,v in zip("XYZ",coordinates))
                    + row("FACE_OFFSETS",face_offsets) + row("FACE_NODES",faces)
                    + row("CELL_OFFSETS",cell_offsets) + row("CELL_FACES",cells)
                    + row("CELL_ORIENTATIONS",signs) + row("BOUNDARY_FACES",boundary)
                    + row("BOUNDARY_PATCH",patches) + row("PATCH_NAMES",labels) + "END\n")


def setup(case, angles, layers, end, planes=1):
    annulus(case/"grid.mesh", angles, layers, planes)
    (case/"main.txt").write_text(f"mesh=grid.mesh\nreconstruction=weno5\nreconstruction_variables=conservative\n"
                                 f"time_scheme=ssprk3\neuler_solver=roe\ncfl=.45\nbackend=cpu\nomp_threads=2\n"
                                 f"mpi_timing=1\nfinal_time={end}\nexact_solution=formula\n")
    # Точное стационарное вращение: u_theta = 0.7 r, dp/dr = rho * 0.49 r.
    fields = ("rho=5*sqrt(x*x+y*y)-4\nux=-.7*y\nuy=.7*x\n"
              "p=1+.49*(5*((x*x+y*y)^1.5-1)/3-2*(x*x+y*y-1))\n")
    if planes > 1:
        fields += "uz=0\n"
    (case/"init.txt").write_text("initial_type formula\n"+fields)
    (case/"exact.txt").write_text(fields)
    (case/"gas.txt").write_text("gamma=1.4\n")
    walls = "inner, outer, ends" if planes > 1 else "inner, outer"
    (case/"bc.txt").write_text(f"[{walls}]\nbc_type novisc_wall\n")
    (case/"cuda.txt").write_text("devices=0\n")
    (case/"output.txt").write_text("[final]\nfile=final.fields\nformat=binary\nlayout=global\n"
                                   "every_steps=1000000\nvariables=rho,ux,uy,p\n")


def compare(ref, actual):
    a, fields = binary_snapshot(ref/"final.fields")
    b, result = binary_snapshot(actual/"final.fields")
    if a != b or fields.keys() != result.keys():
        raise AssertionError("Different snapshot metadata")
    for name, expected in fields.items():
        for x,y in zip(expected,result[name]):
            if not math.isclose(x,y,rel_tol=2.e-11,abs_tol=2.e-12):
                raise AssertionError(f"{name}: {x} != {y}")


def main():
    parser = mpi_parser(__doc__)
    parser.add_argument("--gpu", action="store_true", help="also test the selected visible GPU")
    args = parser.parse_args()
    with mpi_workspace("curved-") as work:
        for planes in (1, 5):
            case = Path(work)/f"planes-{planes}"
            case.mkdir()
            setup(case, 48 if planes == 1 else 24, 6, .015, planes)
            run([args.meshdec,case/"grid.mesh",case/"grid.mdec","--target-ranks=2"])
            for mode in ("straight","ijk","pl","auto"):
                (case/"scheme.txt").write_text(f"stencil_geometry={mode}\nstencil_max_angle=36\n")
                reference = case/(mode+"-cpu")
                solver = [args.aero,f"--input={case}"]
                run([*solver,f"--output={reference}"])
                distributed = case/(mode+"-mpi")
                run([args.mpiexec,"-n",2,*solver,f"--output={distributed}","main:mesh=grid.mdec"])
                compare(reference, distributed)
                if args.gpu:
                    device = case/(mode+"-gpu")
                    run([*solver,f"--output={device}","main:backend=cuda"])
                    compare(reference, device)
                print(f"PASS annulus {mode}, planes={planes}", flush=True)


if __name__ == "__main__":
    main()
