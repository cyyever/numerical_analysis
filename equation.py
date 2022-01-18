from numbers import Real
from typing import Callable

epsilon = 1e-8


def bisection(f: Callable, a: Real, b: Real) -> Real | None:
    """find a root of equation f in the interval [a,b]. Return None when no root is found."""
    assert a <= b
    if f(a) * f(b) > 0:
        return None
    while b - a > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return a


def fixed_point_iteration(f: Callable, x: Real, step_number: int) -> Real:
    for _ in range(step_number):
        x = f(x)
    return x
