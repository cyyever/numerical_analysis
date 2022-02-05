import numpy as np

from iterative_method import fixed_point_iteration


def power_method(A: np.ndarray, **kwargs) -> tuple:
    def f(v):
        v = A @ v
        eigenvalue = np.amax(np.absolute(v))
        v = v / eigenvalue
        return v

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1))
    eigenvalue = np.amax(np.absolute(v))
    return (eigenvalue, v)
