import copy

import numpy

from iterative_method import fixed_point_iteration


def jacobi_method(A, b, **kwargs):
    diagonal = A.diagonal()
    L_U = copy.deepcopy(A)
    numpy.fill_diagonal(L_U, 0)

    def f(x):
        return numpy.divide(b - L_U @ x, diagonal)

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)


def gauss_seidel_method(A, b, **kwargs):
    """Just like jabobi method, but use the most recently updated values"""
    diagonal = A.diagonal()
    L_U = copy.deepcopy(A)
    numpy.fill_diagonal(L_U, 0)

    def f(x):
        for idx in range(len(x)):
            x[idx] = (b[idx] - L_U[idx] @ x) / diagonal[idx]
        return x

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)
