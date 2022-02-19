from polynomial import Polynomial
from spline import bezier_curve, natural_cubic_spline


def test_natural_cubic_spline():
    polynomials = natural_cubic_spline([(0, 3), (1, -2), (2, 1)])
    assert polynomials[0] == Polynomial([3.0, -7.0, 0.0, 2.0], [0.0] * 3)


def test_bezier_curve():
    f = bezier_curve([(1, 1), (1, 3), (3, 3), (2, 2)])
    assert f(0) == 1
