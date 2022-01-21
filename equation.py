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


def __n_guess_iteration(f: Callable, guesses: tuple, step_number: int = 10) -> float:
    guess_num = len(guesses)
    for _ in range(step_number):
        new_point = f(*guesses)
        if new_point == guesses[-1]:
            return new_point
        if guess_num == 1:
            guesses = (new_point,)
        else:
            guesses = (*guesses[1:], new_point)
    return guesses[-1]


def fixed_point_iteration(f: Callable, x: float, **kwargs) -> float:
    return __n_guess_iteration(f, guesses=(x,), **kwargs)


def n_th_root(x: float, n: int, **kwargs) -> float:
    assert n > 0
    if x == 0:
        return x
    assert x > 0
    return fixed_point_iteration(
        lambda a: (n - 1) * a / n + x / (n * a ** (n - 1)), 1.0, **kwargs
    )


def sqrt(x: float, **kwargs) -> float:
    return n_th_root(x, 2, **kwargs)


def newton_method(f: Callable, derivative: Callable, x: float, **kwargs) -> float:
    return fixed_point_iteration(lambda a: a - f(a) / derivative(a), x, **kwargs)


def secant_method(f: Callable, x_0: float, x_1: float, **kwargs) -> float:
    return __n_guess_iteration(
        lambda a, b: b - f(b) * (b - a) / (f(b) - f(a)), guesses=(x_0, x_1), **kwargs
    )


def false_position_method(
    f: Callable, a: float, b: float, step_number: int
) -> float | None:
    """find a root of equation f in the interval [a,b]. Combine bisection method and secant method. Return None when no root is found."""
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


def inverse_quadratic_interpolation_method(
    f: Callable, a: float, b: float, c: float, **kwargs
) -> float | None:
    """A similar generalization of the Secant
    Method to parabolas. However, the parabola is of form x = p(y) instead of y = p(x), as in Muller’s Method."""

    def __formula(x_i, x_i_1, x_i_2):
        q = f(x_i) / f(x_i_1)
        r = f(x_i_2) / f(x_i_1)
        s = f(x_i_2) / f(x_i)
        return x_i_2 - (r * (r - q) * (x_i_2 - x_i_1) + (1 - r) * s * (x_i_2 - x_i)) / (
            (q - 1) * (r - 1) * (s - 1)
        )

    return __n_guess_iteration(__formula, guesses=(a, b, c), **kwargs)
