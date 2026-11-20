"""Общие параметры запуска MPI-проверок и их временные каталоги."""

import argparse
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def report_rows(log, pattern):
    """Сопоставляет строку внутри каждого блока с номером принятого шага."""
    headers = list(re.finditer(r"(?m)^STEP (\d+)  \|", log))
    for i, header in enumerate(headers):
        end = headers[i+1].start() if i+1 < len(headers) else len(log)
        row = re.search(pattern, log[header.end():end], re.M)
        if row:
            yield (header[1], *row.groups())


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
