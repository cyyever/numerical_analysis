import math
from typing import Callable

from polynomial import Polynomial


def lagrange_interpolating(points: list):
    def f(x):
        y = 0
        for i, (x_1, y_1) in enumerate(points):
            numerator = 1
            denominator = 1
            for j, (x_2, _) in enumerate(points):
                if i != j:
                    numerator *= x - x_2
                    denominator *= x_1 - x_2
            y += y_1 * numerator / denominator
        return y

    return f


def chebyshev_base_points(a: float, b: float, n: int, f: Callable):
    """Get n base points from interval [a,b] for interpolation of f"""
    base_points = []
    for i in range(1, n + 1, 2):
        x = (b - a) * math.cos(i * math.pi / (2 * n)) / 2 + (b + a) / 2
        base_points.append((x, f(x)))
    return base_points


def newton_divided_difference(points: list):
    n = len(points)
    divided_differences = {}
    for (x, y) in points:
        divided_differences[(x,)] = y
    x = tuple(x for x, _ in points)
    coefficients = [divided_differences[(points[0][0],)]]
    for i in range(2, n + 1):
        for j in range(n - i + 1):
            divided_differences[x[j: j + i]] = (
                divided_differences[x[j + 1: j + i]]
                - divided_differences[x[j: j + i - 1]]
            ) / (x[j + i - 1] - x[j])
            if j == 0:
                coefficients.append(divided_differences[x[j: j + i]])

    return Polynomial(coefficients, x[:-1])
