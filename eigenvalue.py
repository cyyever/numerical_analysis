import numpy as np
import scipy.linalg

from iterative_method import fixed_point_iteration


def power_iteration_method(A: np.ndarray, **kwargs) -> tuple:
    def f(v):
        v = A @ v
        eigenvalue = np.amax(np.absolute(v))
        v = v / eigenvalue
        return v

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1), **kwargs)
    # use Rayleigh quotient
    u = A @ v
    eigenvalue = v.reshape(-1).dot(u.reshape(-1)) / (v.reshape(-1).dot(v.reshape(-1)))
    return (eigenvalue, u / eigenvalue)


def inverse_power_iteration_method(A: np.ndarray, shift: float = 0, **kwargs) -> tuple:
    A = A - np.eye(A.shape[0]) * shift
    lu, piv = scipy.linalg.lu_factor(A)

    def f(v):
        v = scipy.linalg.lu_solve((lu, piv), v)
        eigenvalue = np.amax(np.absolute(v))
        v = v / eigenvalue
        return v

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1), **kwargs)
    # use Rayleigh quotient
    u = scipy.linalg.lu_solve((lu, piv), v)
    eigenvalue = v.reshape(-1).dot(u.reshape(-1)) / (v.reshape(-1).dot(v.reshape(-1)))
    eigenvalue = 1 / eigenvalue + shift
    return (eigenvalue, u / eigenvalue)
