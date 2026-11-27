"""Заголовок текущего формата AERODEC1, который meshdec создаёт в QA."""

import measurements
import struct
from typing import NamedTuple


_HEADER = struct.Struct("<8s15Q")


# Именованные поля заголовка текущей версии архива разбиения сетки.
class MdecHeader(NamedTuple):
    # Флаги формата и пространственная размерность исходной сетки.
    flags: int
    dimension: int
    # Число исходных сущностей каждого вида и число именованных патчей.
    nodes: int
    faces: int
    cells: int
    boundary_faces: int
    patches: int
    # Размер дерева разбиения и индекс его корневой записи.
    leaves: int
    branches: int
    root: int
    # Байтовые смещения имён патчей, индекса листьев, дерева и содержимого листьев.
    names: int
    index: int
    tree: int
    payload: int

    @classmethod
    # Распаковывает заголовок AERODEC1 и проверяет совместимость его версии.
    def unpack(cls, data):
        # В v5 слово единиц длины исключено. Запрещаем применять эти смещения
        # к старому архиву, чтобы не принять, например, число ветвей за число листьев.
        signature, version, *fields = _HEADER.unpack_from(data)
        if signature != b"AERODEC1" or measurements.compare(version, "!=", 5):
            raise AssertionError(f"Expected AERODEC1 v5, got {signature!r}, version {version}")
        return cls(*fields)


# Читает фиксированный заголовок архива разбиения из файла.
def read_mdec_header(path):
    with path.open("rb") as file:
        return MdecHeader.unpack(file.read(_HEADER.size))
