import numpy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from cyy_numerical_analysis.b_spline import BSpline, KnotVector


def chord_length_parameterization(points: list, degree: int) -> list:
    total_norm = 0
    for i in range(0, len(points) - 1):
        total_norm += numpy.linalg.norm(points[i] - points[i + 1], ord=2)
    knot_vector = [0]
    partial_norm = 0
    for i in range(1, len(points)):
        partial_norm += numpy.linalg.norm(points[i] - points[i - 1], ord=2)
        knot_vector.append(partial_norm / total_norm)
    assert knot_vector[-1] == 1
    knot_vector = [0] * degree + knot_vector + [1] * degree
    return knot_vector


def B_spline_interpolation(
    points: list, degree: int = 3, print_matrices: bool = False
) -> BSpline:
    knot_vector = KnotVector(
        chord_length_parameterization(points=points, degree=degree), degree=degree
    )
    n = len(points) - 1
    A = numpy.zeros((n + 3, n + 3), dtype=numpy.float64)
    b1 = numpy.zeros(n + 3, dtype=numpy.float64)
    b2 = numpy.zeros(n + 3, dtype=numpy.float64)
    row_num = 0
    for i, point in enumerate(points):
        if row_num in (1, n + 1):
            row_num += 1
        knot = knot_vector.get_knot(i + degree)
        A[row_num][i] = knot_vector.evaluate_base_function(
            t=knot, index=i, degree=degree
        )
        A[row_num][i + 1] = knot_vector.evaluate_base_function(
            t=knot, index=i + 1, degree=degree
        )
        A[row_num][i + 2] = knot_vector.evaluate_base_function(
            t=knot, index=i + 2, degree=degree
        )
        b1[row_num] = point[0]
        b2[row_num] = point[1]
        row_num += 1

    # endpoint conditions
    knot = knot_vector.get_knot(degree)
    A[1][0] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=0, degree=degree, derivative_degree=2
    )
    A[1][1] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=1, degree=degree, derivative_degree=2
    )
    A[1][2] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=2, degree=degree, derivative_degree=2
    )
    knot = knot_vector.get_knot(degree + n)
    A[n + 1][n] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=n, degree=degree, derivative_degree=2
    )
    A[n + 1][n + 1] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=n + 1, degree=degree, derivative_degree=2
    )
    A[n + 1][n + 2] = knot_vector.evaluate_base_function_derivative(
        t=knot, index=n + 2, degree=degree, derivative_degree=2
    )
    if print_matrices:
        print(A)
        print(b1)
        print(b2)
    sparse_A = csc_matrix(A, dtype=float)
    X = spsolve(sparse_A, b1).tolist()
    Y = spsolve(sparse_A, b2).tolist()

    return BSpline(
        points=list(zip(X, Y)), degree=3, knot_vector=knot_vector.get_raw_kot_vector()
    )
