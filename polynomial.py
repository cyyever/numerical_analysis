from functools import lru_cache


class Polynomial:
    def __init__(self, coefficients: list):
        assert coefficients
        if len(coefficients) == 1 and coefficients[0] == 0:
            raise RuntimeError("polynomial is 0")
        self.__coefficients = coefficients

    def derivative(self):
        return Polynomial(
            coefficients=[idx * coef for idx, coef in enumerate(self.__coefficients)][
                1:
            ]
        )

    @lru_cache
    def __call__(self, x):
        # Nested multiplication
        y = self.__coefficients[-1]
        for c in reversed(self.__coefficients[:-1]):
            y = y * x + c
        return y
