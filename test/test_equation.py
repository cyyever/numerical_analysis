import math

import pytest
from equation import (bisection_method, brend_method, false_position_method,
                      fixed_point_iteration,
                      inverse_quadratic_interpolation_method, n_th_root,
                      newton_method, secant_method, sqrt)
from polynomial import Polynomial


def test_bisection_method():
    f = Polynomial([-1, 1, 0, 1])
    x = bisection_method(f, 0, 1)
    assert x is not None
    assert pytest.approx(x, abs=0.001) == 0.6821


def test_fixed_point_iteration():
    x = fixed_point_iteration(math.cos, 1)
    assert pytest.approx(x) == 0.739085133


def test_n_th_root():
    x = sqrt(2)
    assert pytest.approx(x, abs=0.001) == 1.414
    x = n_th_root(2, 3)
    assert pytest.approx(x, abs=0.0001) == 1.2599


def test_newton_method():
    f = Polynomial([-1, 1, 0, 1])
    x = newton_method(f, f.derivative(), -0.7)
    assert pytest.approx(x, abs=0.0001) == 0.6823


def test_secant_method():
    f = Polynomial([-1, 1, 0, 1])
    x = secant_method(f, 0, 1)
    assert pytest.approx(x, abs=0.0001) == 0.6823


def test_false_position_method():
    f = Polynomial([0, 3 / 2, -2, 1])
    x = false_position_method(f, -1, 1)
    assert x is not None
    assert pytest.approx(x) == 0


def test_inverse_quadratic_interpolation_method():
    f = Polynomial([0, 3 / 2, -2, 1])
    x = inverse_quadratic_interpolation_method(f, -1, 0.5, 1)
    assert x is not None
    assert pytest.approx(x) == 0


def test_brend_method():
    f = Polynomial([-1, 1, 0, 1])
    x = brend_method(f, -1, 1)
    assert x is not None
    assert pytest.approx(x, abs=0.001) == 0.6821
