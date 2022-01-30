import copy

import numpy

from iterative_method import fixed_point_iteration


def jacobi_method(A, b, **kwargs):
    def f(x):
        tmp = copy.deepcopy(x)
        for idx, a in enumerate(x):
            print(b[idx] - A[idx] @ x + A[idx][idx] * a)
            tmp[idx] = (b[idx] - A[idx] @ x + A[idx][idx] * a) / A[idx][idx]
        return tmp

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)
