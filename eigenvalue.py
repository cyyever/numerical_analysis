import numpy as np
import scipy.linalg

from iterative_method import fixed_point_iteration


def rayleigh_quotient(v, u):
    return v.reshape(-1).dot(u.reshape(-1)) / (v.reshape(-1).dot(v.reshape(-1)))


def power_iteration_method(A: np.ndarray, **kwargs) -> tuple:
    def f(v):
        v = A @ v
        eigenvalue = np.amax(np.absolute(v))
        # normalization
        return v / eigenvalue

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1), **kwargs)
    # use Rayleigh quotient
    u = A @ v
    eigenvalue = rayleigh_quotient(v, u)
    return (eigenvalue, u / eigenvalue)


def inverse_power_iteration_method(A: np.ndarray, shift: float = 0, **kwargs) -> tuple:
    A = A - np.eye(A.shape[0]) * shift
    lu, piv = scipy.linalg.lu_factor(A)

    def f(v):
        v = scipy.linalg.lu_solve((lu, piv), v)
        eigenvalue = np.amax(np.absolute(v))
        # normalization
        v = v / eigenvalue
        return v

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1), **kwargs)
    # use Rayleigh quotient
    u = scipy.linalg.lu_solve((lu, piv), v)
    eigenvalue = rayleigh_quotient(v, u)
    eigenvalue = 1 / eigenvalue + shift
    return (eigenvalue, u / eigenvalue)


def rayleigh_quotient_iteration_method(
    A: np.ndarray, shift: float = 0, **kwargs
) -> tuple:
    """
    The Rayleigh quotient can be used in conjunction with Inverse Power Iteration. We know that it converges to the eigenvector associated to the eigenvalue with the smallest distance to the shift s, and that convergence is fast if this distance is small. If at any step along the way an approximate eigenvalue were known, it could be used as the shift s, to speed convergence.
    """

    def f(v):
        nonlocal shift
        u = scipy.linalg.solve(A - np.eye(A.shape[0]) * shift, v)
        eigenvalue = np.amax(np.absolute(v))
        shift = eigenvalue
        # normalization
        return u / eigenvalue

    v = fixed_point_iteration(f=f, x=np.random.rand(A.shape[0], 1), **kwargs)
    # use Rayleigh quotient
    u = A @ v
    eigenvalue = rayleigh_quotient(v, u)
    return (eigenvalue, u / eigenvalue)
