"""Run the aero verification matrix. Requires Python 3.8+ and the two project binaries."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from exact import CASES

QA = Path(__file__).resolve().parent
ROOT = QA.parent
RECONSTRUCTIONS = ("first_order", "ebr3", "ebr5", "weno3", "weno5")
VARIABLES = ("conservative", "characteristic", "flux")
FLUXES = ("roe", "lf")
TIME_SCHEMES = ("euler", "ssprk3")
SCHEME_KEYS = ("reconstruction", "reconstruction_variables", "euler_solver", "time_scheme")


def binary(name, override):
    if override:
        paths = [Path(override)]
    elif os.name == "nt":
        release = ROOT / "build/msvs/x64/Release"
        paths = [p / (name + ".exe") for p in (ROOT / "bin", release, release / "cuda-sm86")]
    else:
        paths = [ROOT / "bin" / name]
    for path in paths:
        if path.is_file():
            return path.resolve()
    raise ValueError(f"Missing {name}. Build the project with 'make programs' or supply --{name.replace('_', '-')}.")


def settings(path):
    pairs = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    return dict(tuple(s.strip() for s in line.split("=", 1)) for line in pairs if line)


def nodes(mesh, dim):
    words = mesh.read_text(encoding="ascii").split()
    count = int(words[words.index("COUNTS") + 1])
    if count < 1 or words[:5] != ["AEROMESH", "2", "TEXT", "DIMENSION", str(dim)]:
        raise ValueError(f"Unexpected mesh header in {mesh}")
    coords = []
    for axis in "XYZ":
        start = words.index(axis) + 1
        coords.append([float(v) for v in words[start:start + count]])
    return list(zip(*coords))


def reference(coords, formula, dim, time, gamma):
    names = ["rho", *("u" + a for a in "xyz"[:dim]), "p", "E"]
    exact = {key: [] for key in names}
    weights = []
    for xyz in coords:
        state = formula(*xyz, time, gamma)
        energy = state[-1]/(gamma - 1) + 0.5*state[0]*sum(u*u for u in state[1:-1])
        for key, value in zip(names, (*state, energy)):
            exact[key].append(value)
        # Relative nodal dual volumes on the uniform unit Cartesian meshes.
        # Opposite periodic boundary copies together have one interior weight.
        weights.append(0.5 ** sum(c in (0.0, 1.0) for c in xyz[:dim]))
    return exact, weights


def read_snapshot(path, count, dim, time, expected):
    words = iter(path.read_text(encoding="ascii").split())

    def expect(*tokens):
        for token in tokens:
            actual = next(words, None)
            if actual != token:
                raise ValueError(f"Malformed {path.name}: expected {token!r}, got {actual!r}")

    expect("AEROFIELDS", "1", "TEXT", "DIMENSION", str(dim), "MESH_SHA256")
    next(words)  # The Tecplot exporter verifies the digest against the actual mesh.
    expect("COUNTS", str(count))
    for _ in range(3):
        next(words)
    expect("STEP")
    step = int(next(words))
    expect("TIME")
    actual_time = float(next(words))
    if step < 1 or not math.isclose(time, actual_time, rel_tol=1e-13, abs_tol=0):
        raise ValueError(f"Expected an evolved solution at t={time}, got step={step}, t={actual_time}")
    expect("FIELDS", str(len(expected)))
    fields = {}
    for _ in expected:
        expect("FIELD")
        key = next(words).strip('"')
        if key not in expected or key in fields:
            raise ValueError(f"Unexpected or duplicate field {key!r}")
        expect("LOCATION", "NODE", "COMPONENTS", "1", '"value"', "UNITS")
        unit = next(words)
        expect("VALUES", str(count), "DATA")
        values = [float(next(words)) for _ in range(count)]
        if not all(map(math.isfinite, values)):
            raise ValueError(f"Non-finite {key} at step {step}")
        expect("END_FIELD")
        fields[key] = (unit, values)
    expect("END")
    if next(words, None) is not None:
        raise ValueError("Trailing data in snapshot")
    return fields


def with_reference(snapshot, target, fields, exact):
    # Keep the solver-produced mesh identity, time, step and numerical fields.
    original = snapshot.read_text(encoding="ascii").rstrip()
    header, body = original.split("\nFIELDS ", 1)
    _, body = body.split("\n", 1)
    with target.open("w", encoding="ascii") as out:
        field_data = body.rsplit("END", 1)[0]
        out.write(f"{header}\nFIELDS {3*len(fields)}\n" + field_data)
        for key, (units, actual) in fields.items():
            errors = (a - b for a, b in zip(actual, exact[key]))
            for prefix, values in (("exact", exact[key]), ("error", errors)):
                out.write(f'FIELD "{prefix}_{key}"\nLOCATION NODE\nCOMPONENTS 1 "value"\n')
                out.write(f"UNITS {units}\nVALUES {len(actual)}\nDATA ")
                out.write(" ".join(map(repr, values)) + "\nEND_FIELD\n")
        out.write("END\n")


def compare(fields, exact, coords, weights, limits, tol):
    failures = []
    summary = []
    total = sum(weights)
    for key, (_, values) in fields.items():
        errors = [abs(a-b) for a, b in zip(values, exact[key])]
        worst = max(range(len(errors)), key=errors.__getitem__)
        peak = errors[worst]
        mean = sum(w*e for w, e in zip(weights, errors))/total
        peak_limit, mean_limit = limits[key]
        if tol is not None:
            peak_limit = tol
        if peak > peak_limit:
            failures.append(f"{key}: max={peak:.6g} > {peak_limit:g} at {coords[worst]}")
        if mean > mean_limit:
            failures.append(f"{key}: mean={mean:.6g} > {mean_limit:g}")
        summary.append(f"{key}={peak:.3g}/{mean:.3g}")
    return failures, " ".join(summary)


def load_tolerances(path, names):
    groups = json.loads(path.read_text(encoding="ascii"))
    if "default" not in groups or set(groups) - {"default", *RECONSTRUCTIONS}:
        raise ValueError(f"Invalid scheme groups in {path}")
    for group, limits in groups.items():
        if set(limits) != set(names):
            raise ValueError(f"Missing or unexpected fields in {path} ({group})")
        for key, pair in limits.items():
            if (not isinstance(pair, list) or len(pair) != 2 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in pair)):
                raise ValueError(f"Invalid [max, mean] tolerances for {group}/{key} in {path}")
    return groups


def execute(command):
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    if proc.returncode:
        raise ValueError(f"{Path(command[0]).name} exit {proc.returncode}\n{proc.stdout.strip()}")
    return proc.stdout


def run_case(name, options, solver, exporter, progress, total):
    formula, dim, smooth = CASES[name]
    folder = QA / "cases" / name
    main = settings(folder / "main.txt")
    mesh = (folder / main["mesh"]).resolve()
    time = float(main["final_time"])
    gamma = float(settings(folder / "gas.txt")["gamma"])
    coords = nodes(mesh, dim)
    exact, weights = reference(coords, formula, dim, time, gamma)
    tolerances = load_tolerances(folder / "tolerances.json", exact)
    outdir = options.output / name
    outdir.mkdir(parents=True, exist_ok=True)
    active = f"CUDA device {options.device}" if options.backend == "cuda" else "CPU"
    failures = 0
    recs = RECONSTRUCTIONS if smooth else ("weno3", "weno5")
    for variant in itertools.product(recs, VARIABLES, FLUXES, TIME_SCHEMES):
        label = "_".join(variant)
        progress += 1
        print(f"[{progress}/{total}] {name} / {' / '.join(variant)}", flush=True)
        try:
            # A fresh directory prevents a missing write from reusing an old snapshot.
            with tempfile.TemporaryDirectory(dir=outdir, prefix=".run-") as work:
                work = Path(work)
                for file in ("init.txt", "bc.txt", "output.txt", "gas.txt"):
                    shutil.copyfile(folder / file, work / file)
                run = dict(main, **dict(zip(SCHEME_KEYS, variant)))
                run["mesh"] = f'"{mesh.as_posix()}"'
                run["backend"] = options.backend
                if options.backend == "cuda":
                    (work / "cuda.txt").write_text(f"devices = {options.device}\n", encoding="ascii")
                (work / "main.txt").write_text("".join(f"{k} = {v}\n" for k, v in run.items()))
                log = execute([solver, f"--input={work}", f"--output={work}"])
                if f"Active backend = {active}" not in log.splitlines():
                    raise RuntimeError(f"{name}/{label}: solver did not confirm {active}.\n{log.strip()}")
                snap = work / "final.fields"
                fields = read_snapshot(snap, len(coords), dim, time, exact)
                limits = tolerances.get(variant[0], tolerances["default"])
                problems, summary = compare(fields, exact, coords, weights, limits, options.tol)
                with_reference(snap, work / "reference.fields", fields, exact)
                plot = work / "solution.plt"
                execute([exporter, mesh, work / "reference.fields", plot, "--threads", "1"])
                plot.replace(outdir / (label + ".plt"))
                if problems:
                    raise ValueError(" | ".join(problems))
                print(f"  OK  {active}  max/mean  {summary}", flush=True)
        except (OSError, ValueError, StopIteration, KeyError) as exc:
            failures += 1
            print(f"  FAIL  {name}/{label}: {exc}", flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", help="only run named case (repeatable)")
    parser.add_argument("--tol", type=float, help="absolute maximum nodal error for every field")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu", help="solver backend (default: cpu)")
    parser.add_argument("--device", type=int, help="CUDA device index (default: 0)")
    parser.add_argument("--output", type=Path, default=QA / "output",
                        help="results root (default: qa/output); CUDA runs go in its cuda subdirectory")
    parser.add_argument("--aero", help="path to the solver executable")
    parser.add_argument("--fields-to-tecplot", help="path to the Tecplot exporter")
    options = parser.parse_args()
    if options.tol is not None and (not math.isfinite(options.tol) or options.tol < 0):
        parser.error("--tol must be finite and non-negative")
    if options.device is not None and (options.device < 0 or options.backend != "cuda"):
        parser.error("--device requires --backend cuda and a non-negative index")
    options.output = options.output.resolve()
    if options.backend == "cuda":
        if options.device is None:
            options.device = 0
        options.output = options.output / "cuda"
    solver = binary("aero", options.aero)
    exporter = binary("fields_to_tecplot", options.fields_to_tecplot)
    print(f"Solver: {solver}\nBackend: {options.backend}\nTecplot: {exporter}\nOutput: {options.output}", flush=True)
    cases = list(dict.fromkeys(options.case or CASES))
    counts = {name: (len(RECONSTRUCTIONS) if CASES[name][2] else 2)
              * len(VARIABLES)*len(FLUXES)*len(TIME_SCHEMES) for name in cases}
    total = sum(counts.values())
    failures = progress = 0
    for name in cases:
        try:
            failures += run_case(name, options, solver, exporter, progress, total)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FAIL  {name}: {exc}", flush=True)
            failures += counts[name]
        progress += counts[name]
    if failures:
        print(f"FAIL: {failures}/{total} runs failed. See FAIL lines above.")
        return 1
    print(f"OK: all {total}/{total} runs passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"FAIL: {exc}")
