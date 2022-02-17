import math
from typing import Callable

import numpy

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


def natural_cubic_spline(points: list):
    x_delta = []
    y_delta = []
    n = len(points)
    for i in range(n - 1):
        x_delta.append(points[i + 1][0] - points[i][0])
        y_delta.append(points[i + 1][1] - points[i][1])

    A = numpy.zeros((n - 1, n - 1))
    assert A[0, 0] == 0
    B = numpy.zeros(n - 1)
    A[0, 0] = 1
    for i in range(n - 2):
        A[i + 1, i] = x_delta[i]
        A[i + 1, i + 1] = 2 * (x_delta[i] + x_delta[i + 1])
        if i + 2 < n - 2:
            A[i + 1, i + 2] = x_delta[i + 1]
        B[i + 1] = 3 * (y_delta[i + 1] / x_delta[i + 1] - y_delta[i] / x_delta[i])
    c = numpy.linalg.solve(A, B).tolist()
    c.append(0)
    polynomials = []
    for i in range(n - 1):
        a = points[i][1]
        b = y_delta[i] / x_delta[i] - x_delta[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * x_delta[i])

        polynomials.append(
            Polynomial(
                coefficients=(a, b, c[i], d),
                base_points=[points[i][0]] * 3,
            )
        )
    return polynomials
