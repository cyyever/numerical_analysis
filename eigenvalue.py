import numpy as np

from iterative_method import fixed_point_iteration


def power_method(A: np.ndarray, **kwargs):
    def f(x):
        _, v = x
        v = A @ v
        eigenvalue = v[np.argmax(np.absolute(v))]
        v = v / eigenvalue
        return (eigenvalue, v)

    v = np.random.rand(A.shape[0], 1)
    return fixed_point_iteration(f=f, x=(None, v))
