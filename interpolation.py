from polynomial import Polynomial


def lagrange_interpolating(points: list):
    def f(x):
        y = 0
        for i, (x_1, y_1) in enumerate(points):
            numerator = 1
            denominator = 1
            for j, (x_2, _) in enumerate(points):
                if i != j:
                    numerator *= x - x_2
                    denominator *= x_1 - x_2
            y += y_1 * numerator / denominator
        return y

    return f


def newton_divided_difference(points: list):
    n = len(points)
    divided_differences = {}
    for (x, y) in points:
        divided_differences[(x,)] = y
    x = tuple(x for x, _ in points)
    coefficients = [divided_differences[(points[0][0],)]]
    for i in range(2, n + 1):
        for j in range(n - i + 1):
            divided_differences[x[j: j + i]] = (
                divided_differences[x[j + 1: j + i]]
                - divided_differences[x[j: j + i - 1]]
            ) / (x[j + i - 1] - x[j])
            if j == 0:
                coefficients.append(divided_differences[x[j: j + i]])

    return Polynomial(coefficients, x[:-1])

    # def f(z):
    #     nonlocal x
    #     y = 0
    #     for x_point in reversed(x[:-1]):
    #         print("x is ", x, "dif is", divided_differences[x], "x_point is", x_point)
    #         y = (y + divided_differences[x]) * (z - x_point)
    #         print("y=", y)
    #         x = x[:-1]
    #     assert len(x) == 1
    #     print("x is ", x, "dif is", divided_differences[x])
    #     y += divided_differences[x]
    #     print("y=", y)
    #     return y

    # return f
