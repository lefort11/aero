"""Численная диагностика встроенных проверок QA."""
import atexit
import math
import operator
import sys
from pathlib import Path


_operations = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
               "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_measurements = {}


def _difference(left, right):
    """Возвращает модуль разности чисел или максимальную разность одноимённых компонент."""
    if type(left) in (float, int) and type(right) in (float, int):
        return 0 if left == right else abs(left - right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        pairs = zip(left, right)
        if len(left) != len(right):
            return None
    elif isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        pairs = ((left[k], right[k]) for k in left)
    else:
        return None
    return max((d for a, b in pairs if (d := _difference(a, b)) is not None), default=None)


def _record(caller, difference, measured, comparison, reference, margin):
    """Для цикла по узлам сохраняет худший случай с данного места сравнения."""
    site = caller.f_code, caller.f_lasti, comparison
    previous = _measurements.get(site)
    if previous is None or math.isnan(margin) or margin > previous[0]:
        _measurements[site] = (margin, caller.f_lineno, difference, measured, reference)


def compare(left, relation, right):
    """Сравнивает исходные операнды и регистрирует числа для итоговой сводки."""
    return _compare(sys._getframe(1), left, relation, right)


def chain(first, *bounds):
    """Проверяет последовательные границы, например нижнюю, значение и верхнюю."""
    caller = sys._getframe(1)
    for index in range(0, len(bounds), 2):
        relation, right = bounds[index:index + 2]
        if not _compare(caller, first, relation, right):
            return False
        first = right
    return True


def _compare(caller, left, relation, right):
    """Регистрирует измерение за вызывающей строкой QA, включая границы диапазона."""
    result = _operations[relation](left, right)
    difference = _difference(left, right)
    if difference is None:
        return result
    # Для массивов печатаем норму разности. Для скаляров также видны оба операнда:
    # в проверке вида error < tolerance именно левый операнд является ошибкой поля.
    scalar = type(left) in (int, float) and type(right) in (int, float)
    margin = difference
    if scalar and relation in ("<", "<="):
        margin = left - right
    elif scalar and relation in (">", ">="):
        margin = right - left
    _record(caller, difference, left if scalar else None, relation,
            right if scalar else None, margin)
    return result


def error_below(error, limit):
    """Проверяет строгий допуск и сохраняет саму ошибку в единицах поля."""
    _record(sys._getframe(1), error, error, "< tolerance", limit, error - limit)
    if not error < limit:
        print(f"QA error={error:.17g}, tolerance={limit:.17g}", flush=True)
    return error < limit


def error_exceeds(error, limit):
    """Проверяет превышение включительной границы и выводит ошибку в единицах поля."""
    _record(sys._getframe(1), error, error, "tolerance", limit, error - limit)
    if error > limit:
        print(f"QA error={error:.17g}, tolerance={limit:.17g}", flush=True)
    return error > limit


def error_at_most(error, limit):
    """Проверяет включительную границу, сохраняя правила сравнения с NaN."""
    _record(sys._getframe(1), error, error, "<= tolerance", limit, error - limit)
    if not error <= limit:
        print(f"QA error={error:.17g}, tolerance={limit:.17g}", flush=True)
    return error <= limit


def isclose(left, right, *, rel_tol=1e-9, abs_tol=0.0):
    """Печатает абсолютную ошибку и фактический допуск стандартной math.isclose."""
    result = math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
    limit = max(abs_tol, rel_tol * max(abs(left), abs(right)))
    error = _difference(left, right)
    _record(sys._getframe(1), error, error, "isclose tolerance", limit, error - limit)
    return result


@atexit.register
def report():
    """Выводит результаты и при успешном завершении, и при исключении в тесте."""
    for (code, _, comparison), (_, line, delta, actual, expected) in _measurements.items():
        detail = "" if actual is None else f", operands: {actual!r} {comparison} {expected!r}"
        print(f"QA {Path(code.co_filename).name}:{line} {code.co_name}: |delta|={delta:.17g}{detail}")
