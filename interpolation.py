import numpy


def lagrange_interpolating(points: list):
    def f(x):
        y = 0
        for i, (x_1, y_1) in enumerate(points):
            numerator = 1
            denominator = 1
            for j, (x_2, y_2) in enumerate(points):
                if i != j:
                    numerator *= x - x_2
                    denominator *= x_1 - x_2
            y += y_1 * numerator / denominator
        return y

    return f
