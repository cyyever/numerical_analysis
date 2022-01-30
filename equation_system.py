import numpy

from iterative_method import fixed_point_iteration


def jacobi_method(A, b, **kwargs):
    diagonal = A.diagonal()

    def f(x):
        return numpy.divide(b - A @ x + numpy.multiply(diagonal, x), diagonal)

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)
