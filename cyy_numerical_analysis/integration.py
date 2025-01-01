import functools
from collections.abc import Callable


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


def composite_simpson_rule(f: Callable, a: float, b: float, m: int) -> float:
    h = (b - a) / (2 * m)
    res = f(a) + f(b)
    for i in range(1, m + 1):
        res += 4 * f(a + h * (2 * i - 1))
    for i in range(1, m):
        res += 2 * f(a + h * 2 * i)
    return res * h / 3


def midpoint_rule(f: Callable, a: float, b: float) -> float:
    h = b - a
    return h * f(a + h / 2)


def three_point_rule(f: Callable, a: float, b: float) -> float:
    h = (b - a) / 4
    return (2 * f(a + h * 1) - f(a + h * 2) + 2 * f(a + h * 3)) * 4 * h / 3


def composite_integration(
    f: Callable, a: float, b: float, m: int, integration_method: Callable
) -> float:
    h = (b - a) / m
    res: float = 0
    for i in range(m):
        res += integration_method(f, a + h * i, a + h * (i + 1))
    return res


composite_midpoint_rule = functools.partial(
    composite_integration, integration_method=midpoint_rule
)


def romberg_integration(f: Callable, a: float, b: float, step: int) -> float:
    R: list[float] = [(b - a) * (f(a) + f(b)) / 2]
    assert step >= 1
    for j in range(2, step + 1):
        next_R: list[float] = [0] * j
        h = (b - a) / (2 ** (j - 1))
        next_R[0] = R[0] / 2
        partial_sum = 0
        for i in range(1, 2 ** (j - 2) + 1):
            partial_sum += f(a + (2 * i - 1) * h)
        partial_sum *= h
        next_R[0] += partial_sum
        for k in range(2, j + 1):
            next_R[k - 1] = ((4 ** (k - 1)) * next_R[k - 2] - R[k - 2]) / (
                4 ** (k - 1) - 1
            )
        R = next_R
    return R[step - 1]


def adaptive_quadrature(f: Callable, a: float, b: float, tol: float):
    intervals = [(a, b)]
    res: float = 0
    while intervals:
        a_n, b_n = intervals.pop(-1)
        tmp = trapezoid_rule(f, a_n, (a_n + b_n) / 2) + trapezoid_rule(
            f, (a_n + b_n) / 2, b_n
        )
        if abs(trapezoid_rule(f, a_n, b_n) - tmp) < 3 * tol * (b_n - a_n) / (b - a):
            res += tmp
        else:
            intervals += [(a_n, (a_n + b_n) / 2), ((a_n + b_n) / 2, b_n)]
    return res
