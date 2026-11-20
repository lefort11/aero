"""Распаковывает проверенный выпуск Kokkos с поддержкой C++11 для текущей сборки."""
import hashlib
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

def prepare(destination):
    """Читает локальный архив проекта и применяет поправки для современных компиляторов."""
    third_party = Path(__file__).resolve().parents[2] / "third_party"
    archive = third_party / "kokkos-3.2.01.zip"
    checksum = "4e88d9cfab0520fd183e3dd76b1c327447f47b44d4dae24b360979e518fc5d75"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError("Повреждён архив Kokkos 3.2.01")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # ZipFile распаковывает и на Windows без права создавать символические ссылки.
    with ZipFile(archive) as source:
        for entry in source.infolist():
            relative = Path(entry.filename).relative_to("kokkos-3.2.01")
            if not entry.is_dir():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(entry))
    # Поправки учитывают ABI Windows и карту, которой принадлежит переданный поток CUDA.
    with (third_party / "kokkos-3.2.01-compat.patch").open("rb") as patch:
        subprocess.run(["patch", "--binary", "-p1", "--batch"], cwd=destination, stdin=patch, check=True)

if __name__ == "__main__":
    prepare(Path(sys.argv[1]))

