import math

from interpolation import (
    chebyshev_base_points,
    lagrange_interpolating,
    newton_divided_difference,
)


def test_lagrange_interpolating():
    f = lagrange_interpolating([(0, 1), (2, 2), (3, 4)])
    assert f(0) == 1
    assert f(2) == 2
    assert f(3) == 4


def test_newton_divided_difference():
    f = newton_divided_difference([(0, 1), (2, 2), (3, 4)])
    assert f(0) == 1
    assert f(2) == 2
    assert f(3) == 4


def test_chebyshev_base_points():
    a = 0
    b = math.pi / 2
    n = 3
    base_points = chebyshev_base_points(a, b, n, f=math.sin)
    P = newton_divided_difference(base_points)
    assert math.fabs(math.sin(b / 2) - P(b / 2)) <= (
        (((b - a) / 2) ** n) / (math.factorial(n) * (2 ** (n - 1)))
    )
