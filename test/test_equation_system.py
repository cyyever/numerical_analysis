import numpy
import pytest
from equation_system import (cholesky_factorization,
                             classical_gram_schmidt_orthogonalization,
                             gauss_seidel_method, jacobi_method,
                             successive_over_relaxation_method)


def test_jacobi_method():
    A = numpy.array([[3, 1], [1, 2]])
    b = numpy.array([5, 5])
    x = jacobi_method(A, b)
    assert x is not None
    assert numpy.linalg.norm(x - numpy.array([1, 2])) == 0
    x = gauss_seidel_method(A, b)
    assert x is not None
    assert numpy.linalg.norm(x - numpy.array([1, 2])) == 0


def test_gauss_seidel_method():
    A = numpy.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]]).astype(float)
    b = numpy.array([4, 1, 1]).astype(float)
    x = gauss_seidel_method(A, b)
    assert x is not None
    assert numpy.linalg.norm(x - numpy.array([2, -1, 1])) == 0


def test_successive_over_relaxation_method():
    A = numpy.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]]).astype(float)
    b = numpy.array([4, 1, 1]).astype(float)
    x = successive_over_relaxation_method(A, b, w=1.1)
    assert x is not None
    assert numpy.linalg.norm(x - numpy.array([2, -1, 1])) == 0


def test_cholesky_factorization():
    A = numpy.array([[4, -2, 2], [-2, 2, -4], [2, -4, 11]]).astype(float)
    R = cholesky_factorization(A)
    assert numpy.linalg.norm(R - numpy.array([[2, -1, 1], [0, 1, -3], [0, 0, 1]])) == 0


def test_classical_gram_schmidt_orthogonalization():
    A = numpy.array([[1, -4], [2, 3], [2, 2]]).astype(float)
    q, r = classical_gram_schmidt_orthogonalization(A)
    assert numpy.linalg.norm(
        q - numpy.array([[1 / 3, -14 / 15], [2 / 3, 1 / 3], [2 / 3, 2 / 15]])
    ) == pytest.approx(0, abs=1e-6)
