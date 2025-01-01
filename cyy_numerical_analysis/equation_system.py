import copy

import numpy
import numpy.linalg
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
        x = copy.deepcopy(x)
        for idx, x_idx in enumerate(x):
            x[idx] = (1 - w) * x_idx + w * ((b[idx] - L_U[idx] @ x) / diagonal[idx])
        return x

    return fixed_point_iteration(f=f, x=numpy.zeros_like(b), **kwargs)


def gauss_seidel_method(A, b, **kwargs):
    """Just like jabobi method, but use the most recently updated values"""
    return successive_over_relaxation_method(A, b, w=1, **kwargs)


def cholesky_factorization(A):
    """If A is a symmetric positive-definite n × n matrix, then there exists an upper triangular n × n matrix R such that A = (R^T)R."""
    if not numpy.all(A == A.T):
        raise RuntimeError("A is not symmetric")
    R = numpy.zeros_like(A)
    for i in range(A.shape[0]):
        if A[i][i] <= 0:
            raise RuntimeError("A is not positive definite")
        R[i][i] = numpy.sqrt(A[i][i])
        b = A[i, i + 1 :]
        R[i, i + 1 :] = b / R[i][i]
        b = b.reshape(A.shape[0] - i - 1, 1)
        A[i + 1 :, i + 1 :] -= (b.T * b) / A[i][i]
    return R


def gram_schmidt_orthogonalization(A, use_classical_version: bool = False):
    """Perform Gram-Schemidt orthogonalization"""
    n = A.shape[1]
    q = numpy.zeros_like(A)
    r = numpy.zeros((n, n))

    for j in range(n):
        y = A[:, j]
        for i in range(j):
            if use_classical_version:
                r[i][j] = q[:, i] @ A[:, j]
            else:
                r[i][j] = q[:, i] @ y
            y -= r[i][j] * q[:, i]
            i += 1
        r[j][j] = numpy.linalg.norm(y, 2)
        q[:, j] = y / r[j][j]
    return q, r


def householder_reflector_QR(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = numpy.identity(m)
    for i in range(n):
        x = A[i:, i]
        w = numpy.zeros_like(x)
        w[0] = -numpy.sign(x[0]) * numpy.linalg.norm(x, 2)
        v = w - x
        H = numpy.identity(m)
        H[i:, i:] = numpy.identity(m - i) - 2 * (
            v.reshape(-1, 1) @ v.reshape(1, -1) / (v.dot(v))
        )
        Q = Q @ H
        A = H @ A
    return Q, A


def broyden_method_1(F: callable, n: int, x=None, A=None, **kwargs):
    if x is None:
        x = numpy.random.rand(n, 1)
    if A is None:
        A = numpy.identity(n)

    def f(x):
        nonlocal A
        new_x = x - numpy.linalg.inv(A) @ F(x)
        x_delta = new_x - x
        if numpy.all(x_delta == 0):
            return new_x
        F_delta = F(new_x) - F(x)
        A = A + (F_delta - A @ x_delta) @ (x_delta.T) / x_delta.T.dot(x_delta)
        return new_x

    return fixed_point_iteration(f=f, x=x, **kwargs)


def broyden_method_2(F: callable, n: int, x=None, B=None, **kwargs):
    if x is None:
        x = numpy.random.rand(n, 1)
    if B is None:
        B = numpy.identity(n)

    def f(x):
        nonlocal B
        new_x = x - B @ F(x)
        x_delta = new_x - x
        if numpy.all(x_delta == 0):
            return new_x
        F_delta = F(new_x) - F(x)
        B = B + ((x_delta - B @ F_delta) @ (x_delta.T) @ B) / (x_delta.T @ B @ F_delta)
        return new_x

    return fixed_point_iteration(f=f, x=x, **kwargs)
