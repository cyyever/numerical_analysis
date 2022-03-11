import numpy
import pytest
from polynomial import PolynomialWithBasePoint
from spline import BezierCurve, natural_cubic_spline


def test_natural_cubic_spline():
    polynomials = natural_cubic_spline([(0, 3), (1, -2), (2, 1)])
    assert polynomials[0] == PolynomialWithBasePoint([3.0, -7.0, 0.0, 2.0], [0.0] * 3)


def test_bezier_curve():
    f = BezierCurve([(1, 1), (1, 3), (3, 3), (2, 2)])
    assert numpy.all(f(0) == numpy.array([1, 1]))
    assert numpy.all(f(1) == numpy.array([2, 2]))
    t = 0
    while t <= 1:
        assert numpy.linalg.norm(
            f(t)
            - numpy.array(
                [1 + 6 * t**2 - 5 * t**3, 1 + 6 * t - 6 * t**2 + t**3]
            )
        ) == pytest.approx(0, abs=0.000000001)
        t += 0.1
