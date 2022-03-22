import functools

import pytest
from differentiation import (
    richardson_extrapolation, three_point_centered_difference,
    three_point_centered_difference_for_second_derivative,
    two_point_forward_difference)


def test_two_point_forward_difference():
    assert two_point_forward_difference(
        f=lambda x: 1 / x, x=2, h=0.00000001
    ) == pytest.approx(-1 / 4, abs=0.001)


def test_three_point_centered_difference():
    assert three_point_centered_difference(
        f=lambda x: 1 / x, x=2, h=0.00000001
    ) == pytest.approx(-1 / 4, abs=0.001)


def test_three_point_centered_difference_for_second_derivative():
    assert three_point_centered_difference_for_second_derivative(
        f=lambda x: 1 / x, x=2, h=0.01
    ) == pytest.approx(1 / 4, abs=0.001)


def test_richardson_extrapolation():
    F = functools.partial(three_point_centered_difference, f=lambda x: 1 / x, x=2)
    assert richardson_extrapolation(f=F, error_order=2, h=0.1) == pytest.approx(
        -1 / 4, abs=0.001
    )
    F = functools.partial(
        three_point_centered_difference_for_second_derivative, f=lambda x: 1 / x, x=2
    )
    assert richardson_extrapolation(f=F, error_order=2, h=0.1) == pytest.approx(
        1 / 4, abs=0.001
    )
