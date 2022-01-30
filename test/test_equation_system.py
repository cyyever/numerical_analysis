import numpy
import pytest
from equation_system import jacobi_method


def test_jacobi_method():
    A = numpy.array([[3, 1], [1, 2]])
    b = numpy.array([5, 5])
    x = jacobi_method(A, b)
    assert x is not None
    assert pytest.approx(x, numpy.array([1, 2]))
