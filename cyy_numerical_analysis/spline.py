import numpy
from polynomial import PolynomialWithBasePoint


def natural_cubic_spline(points: list) -> list:
    x_delta = []
    y_delta = []
    n = len(points)
    for i in range(n - 1):
        x_delta.append(points[i + 1][0] - points[i][0])
        y_delta.append(points[i + 1][1] - points[i][1])

    A = numpy.zeros((n - 1, n - 1))
    assert A[0, 0] == 0
    B = numpy.zeros(n - 1)
    A[0, 0] = 1
    for i in range(n - 2):
        A[i + 1, i] = x_delta[i]
        A[i + 1, i + 1] = 2 * (x_delta[i] + x_delta[i + 1])
        if i + 2 < n - 2:
            A[i + 1, i + 2] = x_delta[i + 1]
        B[i + 1] = 3 * (y_delta[i + 1] / x_delta[i + 1] - y_delta[i] / x_delta[i])
    c = numpy.linalg.solve(A, B).tolist()
    c.append(0)
    polynomials = []
    for i in range(n - 1):
        a = points[i][1]
        b = y_delta[i] / x_delta[i] - x_delta[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * x_delta[i])

        polynomials.append(
            PolynomialWithBasePoint(
                coefficients=(a, b, c[i], d),
                base_points=[points[i][0]] * 3,
            )
        )
    return polynomials


class BezierCurve:
    def __init__(self, control_points: list):
        self.__control_points = tuple(numpy.array(p) for p in control_points)

    def __call__(self, t: float):
        assert 0 <= t <= 1
        tmp = list(self.__control_points)
        while len(tmp) >= 2:
            for i in range(len(tmp) - 1):
                tmp[i] = (1 - t) * tmp[i] + t * tmp[i + 1]
            tmp.pop()
        return tmp[0]

    def degree_elevation(self):
        n = len(self.__control_points)
        new_control_points = list(self.__control_points)
        new_control_points.append(new_control_points[-1])
        for i in range(1, n):
            new_control_points[i] = (
                i * self.__control_points[i - 1] + (n - i) * self.__control_points[i]
            ) / n
        return BezierCurve(new_control_points)
