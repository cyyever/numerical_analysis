from typing import Callable


def two_point_forward_difference(f: Callable, x: float, h=None) -> float:
    if h is None:
        h = 0.0001
    return (f(x + h) - f(x)) / h


def three_point_centered_difference(f: Callable, x: float, h=None) -> float:
    if h is None:
        h = 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def three_point_centered_difference_for_second_derivative(
    f: Callable, x: float, h=None
) -> float:
    if h is None:
        h = 0.0001
    return (f(x - h) - 2 * f(x) + f(x + h)) / (h**2)


def richardson_extrapolation(f: Callable, error_order: int, h=None) -> float:
    if h is None:
        h = 0.0001
    return ((2**error_order) * f(h=h / 2) - f(h=h)) / (2**error_order - 1)
