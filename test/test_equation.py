import math

import pytest
from equation import bisection, fixed_point_iteration
from polynomial import Polynomial


def test_bisection():
    f = Polynomial([-1, 1, 0, 1])
    x = bisection(f, 0, 1)
    assert pytest.approx(x, 0.6821)


def test_fixed_point_iteration():
    x = fixed_point_iteration(math.cos, 1, 20)
    assert pytest.approx(x, 0.739085133)
