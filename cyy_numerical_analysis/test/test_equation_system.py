import numpy
import pytest
from equation_system import (
    broyden_method_1,
    broyden_method_2,
    cholesky_factorization,
    gauss_seidel_method,
    gram_schmidt_orthogonalization,
    householder_reflector_QR,
    jacobi_method,
    successive_over_relaxation_method,
)


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


def test_gram_schmidt_orthogonalization():
    A = numpy.array([[1, -4], [2, 3], [2, 2]]).astype(float)
    Q, _ = gram_schmidt_orthogonalization(A, use_classical_version=False)
    assert numpy.linalg.norm(
        Q - numpy.array([[1 / 3, -14 / 15], [2 / 3, 1 / 3], [2 / 3, 2 / 15]])
    ) == pytest.approx(0, abs=1e-6)
    Q, _ = gram_schmidt_orthogonalization(A, use_classical_version=True)
    assert numpy.linalg.norm(
        Q - numpy.array([[1 / 3, -14 / 15], [2 / 3, 1 / 3], [2 / 3, 2 / 15]])
    ) == pytest.approx(0, abs=1e-6)


def test_householder_reflector_QR():
    A = numpy.array([[3, 1], [4, 3]]).astype(float)
    Q, _ = householder_reflector_QR(A)
    assert numpy.linalg.norm(
        Q - numpy.array([[-0.6, 0.8], [-0.8, -0.6]])
    ) == pytest.approx(0, abs=1e-6)


def test_broyden_method_1():
    def F(x):
        u = x[0][0]
        v = x[1][0]
        return numpy.array([[v - u**3], [u**2 + v**2 - 1]])

    x = broyden_method_1(F=F, x=numpy.array([[1.0], [2.0]]), n=2)
    assert numpy.linalg.norm(x - numpy.array([[0.8260], [0.5636]])) == pytest.approx(
        0, abs=1e-4
    )


def test_broyden_method_2():
    def F(x):
        u = x[0][0]
        v = x[1][0]
        return numpy.array([[v - u**3], [u**2 + v**2 - 1]])

    x = broyden_method_2(F=F, x=numpy.array([[1.0], [2.0]]), n=2)
    assert numpy.linalg.norm(x - numpy.array([[0.8260], [0.5636]])) == pytest.approx(
        0, abs=1e-4
    )
