from typing import Callable


def trapezoid_rule(f: Callable, a: float, b: float):
    h = b - a
    return h * (f(a) + f(b)) / 2


def simpson_rule(f: Callable, a: float, b: float):
    h = (b - a) / 2
    return h * (f(a) + 4 * f(a + h) + f(b)) / 3
