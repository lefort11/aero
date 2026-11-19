"""Сверяет выбор разбиения по ресурсам CPU с геометрией и полями обычного расчёта."""

from mpi_common import ROOT, mpi_parser, mpi_workspace
import shutil
import struct
from pathlib import Path

from mpi_euler import check_output, check_report, launch
from mpi_geometry import compare, geometry, legacy_archive, run
from mpi_periodic import report_rows
from run import nodes, settings


def check_estimates(archive, plan, factor):
    """Суммирует записи каталога листьев и сверяет оценку памяти с планом распределения."""
    # Читаем каталог листьев малой сетки напрямую, чтобы сверить оценки дерева независимым способом.
    data = archive.read_bytes()

    def word(offset):
        return struct.unpack_from("<Q", data, offset)[0]

    dim, cells, leaves = word(24), word(56), word(80)
    index, tree = word(112), word(120)
    raw = [word(index + 48 * i + 16) for i in range(leaves)]
    nodes_per_leaf = [word(index + 48 * i + 24) for i in range(leaves)]
    header = tree + 112 * word(88)
    near = header + 16 + 8 * (word(40) + 1) + 16 * word(header)
    far = near + 8 * (2 * leaves + 1) + 16 * word(header + 8) + 8
    cell_bytes = (sum(raw) + cells - 1) // cells
    for row in plan:
        a, b = int(row["first_leaf"]), int(row["end_leaf"])
        links = word(near + 16 * b) - word(near + 16 * a) + word(far + 8 * b) - word(far + 8 * a)
        base = sum(raw[a:b]) + sum(nodes_per_leaf[a:b]) * 8 * (dim + 2) * 3 + links * cell_bytes
        expected = base * factor / 1048576
        if abs(float(row["estimated_mesh_MiB"]) - expected) > max(expected * 6e-6, 1e-12):
            raise AssertionError(f"Incorrect estimate for leaf range [{a},{b}): {row}")


def verify(case, aero, meshdec, launcher):
    """Сверяет выбор разбиения по ресурсам CPU с геометрией и полями обычного расчёта."""
    source = ROOT / "qa/cases/pulse_1d"
    shutil.copytree(source, case)
    main = case / "main.txt"
    options = settings(main)
    names = ["rho", "ux", "p", "E"]
    count = len(nodes(case / "grid.mesh", 1))
    options.update(reconstruction="weno5", time_scheme="ssprk3", final_time=0.008, flowreport_every=1)
    (case / "output.txt").write_text(
        "[history]\nfile = state.fields\nevery_steps = 1\nmulti = 1\n"
        "format = text\nvariables = " + ", ".join(names) + "\n"
    )

    def configure(**extra):
        main.write_text("".join(f"{k} = {v}\n" for k, v in dict(options, **extra).items()))

    configure()
    reference = case / "native"
    serial = run([aero, f"--input={case}", f"--output={reference}"])
    run([aero, f"--input={case}", f"--output={reference}", "--geometry-only"])
    expected, _ = geometry(reference, 1)
    archive = case / "grid.mdec"
    run([meshdec, case / "grid.mesh", archive, "--target-ranks=16", "--compress"])
    run([meshdec, case / "grid.mesh", case / "one.mdec"])
    legacy_archive(archive, case / "legacy.mdec")
    (case / "grid.mesh").unlink()
    options["mesh"] = "grid.mdec"
    for title, weights in (("fast-right", [1, 3]), ("fast-left", [3, 1]), ("four", [1, 3, 2, 2])):
        ranks = len(weights)
        configure(
            mpi_rank_weights=",".join(map(str, weights)),
            mpi_node_memory_mib="auto",
            mpi_memory_factor=7.5,
        )
        out = case / title
        geom = launch(aero, case, out, launcher, ranks, "--geometry-only")
        actual, owners = geometry(out, ranks)
        compare(expected, actual)
        log = launch(aero, case, out, launcher, ranks)
        check_report(serial, log)
        error = check_output(reference, out, ranks, count, 1, names)
        plan = report_rows(log, "mpi_rank\tweight")
        check_estimates(archive, plan, 7.5)
        if plan != report_rows(geom, "mpi_rank\tweight"):
            raise AssertionError("Geometry and time integration chose different cuts")
        for row in plan:
            rank = int(row["mpi_rank"])
            if (
                float(row["weight"]) != weights[rank]
                or abs(owners[rank] - count * weights[rank] / sum(weights)) > 4
            ):
                raise AssertionError(f"Incorrect work share {owners} for weights {weights}")
        print(f"OK {title} weights={weights} owned={owners} max field difference={error:.3g}")
    configure(mesh="one.mdec", mpi_rank_weights="1,8,1,1")
    out = case / "one"
    log = launch(aero, case, out, launcher, 4)
    check_report(serial, log)
    check_output(reference, out, 4, count, 1, names, empty=True)
    ranks = report_rows(log, "rank\tnode")
    if [int(r["leaves"]) for r in ranks] != [0, 1, 0, 0]:
        raise AssertionError("The fastest CPU did not receive the single leaf")
    configure(mesh="legacy.mdec", reconstruction="first_order", mpi_rank_weights="1,3")
    launch(aero, case, out, launcher, 2, "--geometry-only")
    actual, _ = geometry(out, 2)
    compare(expected, actual)
    launch(aero, case, out, launcher, 2)
    print("OK empty ranges and legacy archive with weighted cuts")

    cmd = [launcher, "-n", 2, aero, f"--input={case}", f"--output={case/'invalid'}", "--geometry-only"]
    for extra, message in (
        (dict(mpi_rank_weights="1,2,3"), "one weight per MPI rank"),
        (dict(mpi_rank_weights="1,0"), "expected finite value > 0"),
        (dict(mpi_node_memory_mib=".001"), "initial process RSS"),
        (dict(mpi_memory_factor="1e15", mpi_node_memory_mib="4096"), "first unplaced leaf"),
    ):
        configure(**extra)
        log = run(cmd, success=False)
        if message not in log or (case / "invalid").exists():
            raise AssertionError(f"Failure before mesh loading expected: {log}")
    print("OK collective diagnostics for invalid weights and insufficient node RAM")


def main():
    parser = mpi_parser(__doc__, require_programs=True)
    args = parser.parse_args()
    with mpi_workspace("mpi-balance-") as directory:
        verify(Path(directory) / "case", args.aero.resolve(), args.meshdec.resolve(), args.mpiexec)


if __name__ == "__main__":
    main()
