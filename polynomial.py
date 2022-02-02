from functools import lru_cache
from typing import Sequence


class Polynomial:
    def __init__(self, coefficients: Sequence, base_points: None | Sequence = None):
        assert coefficients
        if len(coefficients) == 1 and coefficients[0] == 0:
            raise RuntimeError("polynomial is 0")
        self.__coefficients: tuple = coefficients
        self.__base_points: tuple | None = base_points

    def derivative(self):
        assert self.__base_points is None
        return Polynomial(
            coefficients=[idx * coef for idx, coef in enumerate(self.__coefficients)][
                1:
            ]
        )

    @lru_cache
    def __call__(self, x):
        # Nested multiplication
        coefficients: tuple = tuple(reversed(self.__coefficients))
        y = coefficients[0]
        if self.__base_points is None:
            for c in coefficients[1:]:
                y = y * x + c
            return y
        base_points = reversed(self.__base_points)
        for c, base_point in zip(coefficients[1:], base_points):
            y = y * (x - base_point) + c
        return y
