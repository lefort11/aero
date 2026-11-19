"""Общие параметры запуска MPI-проверок и их временные каталоги."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def mpi_parser(description, *, require_programs=False):
    """Добавляет пути к солверу, декомпозеру и программе запуска MPI."""
    parser = argparse.ArgumentParser(description=description)
    suffix = ".exe" if os.name == "nt" else ""
    for name in ("aero", "meshdec"):
        # Часть сценариев требует явно выбрать сборку, например при проверке нескольких GPU.
        default = None if require_programs else ROOT / "bin" / (name + suffix)
        parser.add_argument("--" + name, type=Path, required=require_programs, default=default)
    parser.add_argument("--mpiexec", default="mpiexec")
    return parser


def mpi_workspace(prefix):
    """Выделяет отдельный каталог для входов и результатов одного сценария."""
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    return TemporaryDirectory(prefix=prefix, dir=build)
