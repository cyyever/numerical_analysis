import numpy
import pytest
from equation_system import (gauss_seidel_method, jacobi_method,
                             successive_over_relaxation_method)


def test_jacobi_method():
    A = numpy.array([[3, 1], [1, 2]])
    b = numpy.array([5, 5])
    x = jacobi_method(A, b)
    assert x is not None
    assert pytest.approx(x, numpy.array([1, 2]))


def test_gauss_seidel_method():
    A = numpy.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]])
    b = numpy.array([4, 1, 1])
    x = gauss_seidel_method(A, b)
    assert x is not None
    assert pytest.approx(x, numpy.array([2, -1, 1]))


def test_successive_over_relaxation_method():
    A = numpy.array([[3, 1, -1], [2, 4, 1], [-1, 2, 5]])
    b = numpy.array([4, 1, 1])
    x = successive_over_relaxation_method(A, b, w=1.1)
    assert x is not None
    assert pytest.approx(x, numpy.array([2, -1, 1]))