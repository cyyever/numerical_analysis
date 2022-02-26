from functools import lru_cache
from typing import Sequence


class Polynomial:
    def __init__(self, coefficients: Sequence):
        """the coefficients are from the lowest power to the highest power"""
        assert coefficients
        while coefficients and coefficients[-1] == 0:
            coefficients.pop()
        if not coefficients:
            coefficients = [0]
        self.__coefficients: tuple = tuple(coefficients)

    @property
    def coefficients(self):
        return self.__coefficients

    def __add__(self, other):
        coef_size = max(len(self.coefficients), len(other.coefficients))
        result_coefficient = list(self.coefficients) + [0] * (
            coef_size - len(self.coefficients)
        )
        for i, cof in enumerate(other.coefficients):
            result_coefficient[i] += cof
        return Polynomial(result_coefficient)

    def __sub__(self, other):
        coef_size = max(len(self.coefficients), len(other.coefficients))
        result_coefficient = list(self.coefficients) + [0] * (
            coef_size - len(self.coefficients)
        )
        for i, cof in enumerate(other.coefficients):
            result_coefficient[i] -= cof
        return Polynomial(result_coefficient)

    def __mul__(self, other):
        other_coefficients = None
        match other:
            case Polynomial():
                other_coefficients = other.coefficients
            case _:
                other_coefficients = [other]

        coef_size = len(self.coefficients) + len(other.coefficients)
        result_coefficient = [0] * coef_size
        for i, coef in enumerate(self.coefficients):
            for j, coef2 in enumerate(other_coefficients):
                result_coefficient[i + j] += coef * coef2

        return Polynomial(result_coefficient)

    def __str__(self):
        return f"coefficients:{self.coefficients}"

    def __hash__(self):
        return hash(self.coefficients)

    def __eq__(self, other):
        return self.coefficients == other.coefficients

    def derivative(self):
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
        for c in coefficients[1:]:
            y = y * x + c
        return y


class PolynomialWithBasePoint:
    def __init__(self, coefficients: Sequence, base_points: Sequence):
        assert coefficients
        while coefficients and coefficients[-1] == 0:
            coefficients.pop()
        if not coefficients:
            coefficients = [0]
        self.__coefficients: tuple = tuple(coefficients)
        self.__base_points: tuple = tuple(base_points)

    @property
    def coefficients(self):
        return self.__coefficients

    @property
    def base_points(self):
        return self.__base_points

    def __str__(self):
        return f"coefficients:{self.coefficients} base_points:{self.base_points}"

    def __hash__(self):
        return hash(self.coefficients) ^ hash(self.base_points)

    def __eq__(self, other):
        return (
            self.coefficients == other.coefficients
            and self.base_points == other.base_points
        )

    @lru_cache
    def __call__(self, x):
        # Nested multiplication
        coefficients: tuple = tuple(reversed(self.__coefficients))
        y = coefficients[0]
        base_points = reversed(self.__base_points)
        for c, base_point in zip(coefficients[1:], base_points):
            y = y * (x - base_point) + c
        return y
