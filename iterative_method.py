from typing import Callable

import numpy


def n_guess_iteration(f: Callable, guesses: tuple, step_number: int = 100) -> float:
    guess_num = len(guesses)
    for _ in range(step_number):
        new_point = f(*guesses)
        res = new_point == guesses[-1]
        match res:
            case numpy.ndarray():
                if numpy.all(res):
                    return new_point
            case _:
                if res:
                    return new_point

        if guess_num == 1:
            guesses = (new_point,)
        else:
            guesses = (*guesses[1:], new_point)
    return guesses[-1]


def fixed_point_iteration(f: Callable, x: float, **kwargs) -> float:
    return n_guess_iteration(f, guesses=(x,), **kwargs)
