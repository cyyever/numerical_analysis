from typing import Callable

epsilon = 1e-8


def bisection_method(f: Callable, a: float, b: float) -> float | None:
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


def fixed_point_iteration(f: Callable, x: float, step_number: int = 10) -> float:
    for _ in range(step_number):
        x = f(x)
    return x


def n_th_root(x: float, n: int, **kwargs: dict) -> float:
    assert n > 0
    if x == 0:
        return x
    assert x > 0
    return fixed_point_iteration(
        lambda a: (n - 1) * a / n + x / (n * a ** (n - 1)), 1.0, **kwargs
    )


def sqrt(x: float, **kwargs: dict) -> float:
    return n_th_root(x, 2, **kwargs)


def newton_method(f: Callable, derivative: Callable, x: float, **kwargs) -> float:
    return fixed_point_iteration(lambda a: a - f(a) / derivative(a), x, **kwargs)


def __two_guess_iteration(
    f: Callable, x_0: float, x_1: float, step_number: int = 10
) -> float:
    for _ in range(step_number):
        tmp = x_1
        x_1 = f(x_0, x_1)
        x_1 = tmp
    return x_1


def secant_method(f: Callable, x_0: float, x_1: float, **kwargs: dict) -> float:
    return __two_guess_iteration(
        lambda a, b: b - f(b) * (b - a) / (f(b) - f(a)), x_0, x_1, **kwargs
    )


def false_position_method(
    f: Callable, a: float, b: float, step_number: int
) -> float | None:
    """find a root of equation f in the interval [a,b]. Combine bisection method and secant method."""
    """Return None when no root is found."""
    assert a <= b
    if f(a) * f(b) > 0:
        return None
    for _ in range(step_number):
        c = (b * f(a) - a * f(b)) / (f(a) - f(b))
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return a
