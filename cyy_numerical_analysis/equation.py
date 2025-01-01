import math
from collections.abc import Callable

from iterative_method import fixed_point_iteration, n_guess_iteration


def __interval_method(
    f: Callable,
    next_point_formula: Callable,
    a: float,
    b: float,
    epsilon=1e-8,
    step_number=1000,
) -> float | None:
    """find a root of equation f in the interval [a,b]. Return None when no root is found."""
    assert a <= b
    if f(a) * f(b) > 0:
        return None
    c = a
    for _ in range(step_number):
        if b - a < epsilon:
            break
        res = next_point_formula(a, b)
        match res:
            case list():
                a, b, c = res
            case _:
                c = res

        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return next_point_formula(a, b)


def bisection_method(f: Callable, a: float, b: float, **kwargs) -> float | None:
    """find a root of equation f in the interval [a,b]. Return None when no root is found."""
    return __interval_method(f, lambda a, b: (a + b) / 2, a, b, **kwargs)


def false_position_method(f: Callable, a: float, b: float, **kwargs) -> float | None:
    """find a root of equation f in the interval [a,b]. Combine bisection method and secant method. Return None when no root is found."""
    return __interval_method(
        f, lambda a, b: (b * f(a) - a * f(b)) / (f(a) - f(b)), a, b, **kwargs
    )


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
    return n_guess_iteration(
        lambda a, b: b - f(b) * (b - a) / (f(b) - f(a)), guesses=(x_0, x_1), **kwargs
    )


def inverse_quadratic_interpolation_method(
    f: Callable, a: float, b: float, c: float, **kwargs
) -> float:
    """A similar generalization of the Secant
    Method to parabolas. However, the parabola is of form x = p(y) instead of y = p(x), as in Muller’s Method."""

    def __formula(x_i, x_i_1, x_i_2):
        q = f(x_i) / f(x_i_1)
        r = f(x_i_2) / f(x_i_1)
        s = f(x_i_2) / f(x_i)
        return x_i_2 - (r * (r - q) * (x_i_2 - x_i_1) + (1 - r) * s * (x_i_2 - x_i)) / (
            (q - 1) * (r - 1) * (s - 1)
        )

    return n_guess_iteration(__formula, guesses=(a, b, c), **kwargs)


def brend_method(
    f: Callable, a: float, b: float, debug: bool = False, **kwargs
) -> float | None:
    min_backward_error = min(math.fabs(f(a)), math.fabs(f(b)))
    last_three_guess: None | tuple[float, float, float] = None

    def criterion(next_point):
        nonlocal min_backward_error
        nonlocal last_three_guess
        if math.fabs(f(next_point)) < min_backward_error:
            assert last_three_guess is not None
            four_guesses = sorted(list(last_three_guess) + [next_point])
            idx = four_guesses.index(next_point)
            if (
                idx in (1, 2)
                and four_guesses[idx + 1] - four_guesses[idx - 1] <= (b - a) / 2
            ):
                return (
                    four_guesses[idx - 1],
                    four_guesses[idx],
                    four_guesses[idx + 1],
                )
        return None

    def next_point_formula(a: float, b: float):
        nonlocal min_backward_error
        nonlocal last_three_guess
        if last_three_guess is None:
            c = (a + b) / 2
            last_three_guess = (a, c, b)
            min_backward_error = min(min_backward_error, math.fabs(f(c)))
            return c
        c = inverse_quadratic_interpolation_method(f, *last_three_guess, step_number=1)
        res = criterion(c)
        if res is not None:
            if debug:
                print("use inverse_quadratic_interpolation_method")
            last_three_guess = res
            min_backward_error = min(min_backward_error, math.fabs(f(c)))
            return c
        c = secant_method(f, a, b, step_number=1)
        res = criterion(c)
        if res is not None:
            if debug:
                print("use secant_method")
            last_three_guess = res
            min_backward_error = min(min_backward_error, math.fabs(f(c)))
            return c
        c = bisection_method(f, a, b, step_number=1)
        if debug:
            print("use bisection_method")
        last_three_guess = (a, c, b)
        min_backward_error = min(min_backward_error, math.fabs(f(c)))
        return c

    return __interval_method(f, next_point_formula, a, b, **kwargs)
