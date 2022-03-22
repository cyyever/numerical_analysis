from typing import Callable


def trapezoid_rule(f: Callable, a: float, b: float) -> float:
    h = b - a
    return h * (f(a) + f(b)) / 2


def simpson_rule(f: Callable, a: float, b: float) -> float:
    h = (b - a) / 2
    return h * (f(a) + 4 * f(a + h) + f(b)) / 3


def composite_trapezoid_rule(f: Callable, a: float, b: float, m: int) -> float:
    h = (b - a) / m
    res = f(a) + f(b)
    for i in range(1, m):
        res += 2 * f(a + h * i)
    return res * h / 2
