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


def successive_over_relaxation_method(A, b, w, **kwargs):
    """Let ω be a real number, and define each component of the new guess xk+1 as a weighted average of ω times the Gauss–Seidel formula and 1 − ω times the current guess."""
    diagonal = A.diagonal()
    L_U = copy.deepcopy(A)
    numpy.fill_diagonal(L_U, 0)

    def f(x):
        for idx, x_idx in enumerate(x):
            x[idx] = (1 - w) * x_idx + w * ((b[idx] - L_U[idx] @ x) / diagonal[idx])
        return x

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)


def gauss_seidel_method(A, b, **kwargs):
    """Just like jabobi method, but use the most recently updated values"""
    return successive_over_relaxation_method(A, b, w=1, **kwargs)
