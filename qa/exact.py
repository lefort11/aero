"""Аналитические решения уравнений Эйлера для независимой проверки полей в узлах."""

from functools import lru_cache
from math import cos, pi, sin, sqrt


def pulse_1d(x, y, z, t, gamma):
    """Переносит гладкий импульс плотности вдоль X с единичной скоростью."""
    return 1 + 0.25 * cos(pi * (x - t - 0.25)) ** 4, 1.0, 1.0


def momentum_2d(x, y, z, t, gamma):
    """Переносит синусоидальный профиль продольной скорости поперёк сетки."""
    return 1.0, 0.3 + 0.2 * sin(2 * pi * (y - 0.75 * t)), 0.75, 1.0


def shear_jump_2d(x, y, z, t, gamma):
    """Восстанавливает переносимый разрыв касательной скорости по периодической координате."""
    upstream = (y - 0.85 * t) % 1
    return 1.0, (0.3 if 0.625 <= upstream < 0.875 else -0.1), 0.85, 1.0


def wave_3d(x, y, z, t, gamma):
    """Вычисляет гладкую волну плотности при постоянных скорости и давлении."""
    rho = (
        1
        + 0.1 * sin(2 * pi * (x - 0.8 * t))
        + 0.06 * cos(2 * pi * (y + 0.6 * t))
        + 0.04 * sin(2 * pi * (z - 0.45 * t))
    )
    return rho, 0.8, -0.6, 0.45, 1.0


def contact_3d(x, y, z, t, gamma):
    """Определяет плотность по положению переносимого контактного разрыва."""
    upstream = (z - 0.8 * t) % 1
    return (1.4 if 0.3125 <= upstream < 0.6125 else 1.0), 0.3, -0.2, 0.8, 1.0


@lru_cache(maxsize=8)
def sod_contact(gamma, rho_l, p_l, rho_r, p_r):
    """Находит одно давление и одну скорость на контакте для всех узлов данной задачи Сода."""
    # Левая волна разрежения и правая ударная волна должны давать одинаковую скорость контакта.
    # Давление заключено между p_r и p_l. Уточняем его делением этого интервала пополам.
    c_l, c_r = sqrt(gamma * p_l / rho_l), sqrt(gamma * p_r / rho_r)
    g = (gamma - 1) / (gamma + 1)
    exponent = (gamma - 1) / (2 * gamma)

    def fan_velocity(p):
        return 2 * c_l / (gamma - 1) * (1 - (p / p_l) ** exponent)

    def shock_velocity(p):
        return (p - p_r) * sqrt(2 / ((gamma + 1) * rho_r * (p + g * p_r)))

    low, high = p_r, p_l
    for _ in range(55):
        p = (low + high) / 2
        if fan_velocity(p) > shock_velocity(p):
            low = p
        else:
            high = p
    p_star = (low + high) / 2
    u_star = shock_velocity(p_star)
    return c_l, c_r, g, exponent, p_star, u_star


def sod_1d(x, y, z, t, gamma):
    """Выбирает область точного решения по положению узла относительно фронтов волн."""
    rho_l, p_l, rho_r, p_r = 1.0, 1.0, 0.125, 0.1
    if t == 0:
        return (rho_l, 0.0, p_l) if x < 0.5 else (rho_r, 0.0, p_r)
    c_l, c_r, g, exponent, p_star, u_star = sod_contact(gamma, rho_l, p_l, rho_r, p_r)
    # Отношение (x−x₀)/t сравнивается со скоростями головы и хвоста веера, контакта и скачка.
    xi = (x - 0.5) / t
    if xi <= -c_l:
        return rho_l, 0.0, p_l
    if xi < u_star - c_l * (p_star / p_l) ** exponent:
        u = 2 * (c_l + xi) / (gamma + 1)
        c = c_l - (gamma - 1) * u / 2
        return rho_l * (c / c_l) ** (2 / (gamma - 1)), u, p_l * (c / c_l) ** (1 / exponent)
    if xi < u_star:
        return rho_l * (p_star / p_l) ** (1 / gamma), u_star, p_star
    ratio = p_star / p_r
    if xi < c_r * sqrt((ratio + g) / (1 + g)):
        return rho_r * (ratio + g) / (g * ratio + 1), u_star, p_star
    return rho_r, 0.0, p_r


# Имя постановки связывает аналитическую формулу, размерность и признак гладкого решения.
CASES = {
    "pulse_1d": (pulse_1d, 1, True),
    "sod_1d": (sod_1d, 1, False),
    "momentum_2d": (momentum_2d, 2, True),
    "shear_jump_2d": (shear_jump_2d, 2, False),
    "wave_3d": (wave_3d, 3, True),
    "contact_3d": (contact_3d, 3, False),
}


def linear_acoustic(dim):
    """Плоская звуковая волна вдоль диагонали. Фон покоится, ρ₀ = c₀ = 1."""
    def solution(x, y, z, t, gamma):
        phase = sum((x, y, z)[:dim]) - sqrt(dim) * t
        p = 0.01 * sin(2 * pi * phase)
        velocity = (p / sqrt(dim),) * dim
        return (p, *velocity, p)
    return solution


for dimension in (1, 2, 3):
    CASES[f"linear_acoustic_{dimension}d"] = (linear_acoustic(dimension), dimension, True)
