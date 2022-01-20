import math

import pytest
from equation import (bisection, fixed_point_iteration, n_th_root,
                      newton_method, secant_method, sqrt)
from polynomial import Polynomial


def test_bisection():
    f = Polynomial([-1, 1, 0, 1])
    x = bisection(f, 0, 1)
    assert pytest.approx(x, 0.6821)


def test_fixed_point_iteration():
    x = fixed_point_iteration(math.cos, 1)
    assert pytest.approx(x, 0.739085133)


def test_n_th_root():
    x = sqrt(2)
    assert pytest.approx(x, 1.414)
    x = n_th_root(2, 3)
    assert pytest.approx(x, 1.2599)


def test_newton_method():
    x = newton_method(lambda x: x ** 3 + x - 1, lambda x: 3 * x ** 2 + 1, -0.7)
    assert pytest.approx(x, 0.6823)


def test_secant_method():
    x = secant_method(lambda x: x ** 3 + x - 1, 0, 1)
    assert pytest.approx(x, 0.6823)
