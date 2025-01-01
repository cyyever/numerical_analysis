import math

import pytest
from integration import (
    adaptive_quadrature,
    composite_midpoint_rule,
    composite_simpson_rule,
    composite_trapezoid_rule,
    romberg_integration,
    simpson_rule,
    three_point_rule,
    trapezoid_rule,
)


def test_trapezoid_rule():
    assert trapezoid_rule(math.log, 1, 2) == pytest.approx(0.3466, abs=0.001)


def test_composite_trapezoid_rule():
    assert composite_trapezoid_rule(math.log, 1, 2, m=4) == pytest.approx(
        0.3837, abs=0.001
    )


def test_simpson_rule():
    assert simpson_rule(math.log, 1, 2) == pytest.approx(0.3858, abs=0.001)


def test_composite_simpson_rule():
    assert composite_simpson_rule(math.log, 1, 2, m=4) == pytest.approx(
        0.3863, abs=0.001
    )


def test_three_point_rule():
    assert three_point_rule(lambda x: math.sin(x) / x, 0, 1) == pytest.approx(
        0.9462, abs=0.001
    )


def test_composite_midpoint_rule():
    assert composite_midpoint_rule(
        lambda x: math.sin(x) / x, 0, 1, m=10
    ) == pytest.approx(0.9462, abs=0.001)


def test_romberg_integration():
    assert romberg_integration(math.log, 1, 2, step=4) == pytest.approx(
        0.3863, abs=0.001
    )


def test_adaptive_quadrature():
    assert adaptive_quadrature(
        lambda x: (1 + math.sin(math.e ** (3 * x))), -1, 1, 0.005
    ) == pytest.approx(2.502, abs=0.001)
