import numpy
import pytest
from conjugate_gradient import conjugate_gradient


def test_conjugate_gradient():
    A = numpy.array([[2.0, 2.0], [2.0, 5.0]])
    b = numpy.array([6.0, 3.0])
    x = conjugate_gradient(A, b)
    assert numpy.linalg.norm(A @ x - b) == pytest.approx(0, abs=0.0000001)
    A = -A
    b = -b
    x = conjugate_gradient(A, b)
    assert numpy.linalg.norm(A @ x - b) == pytest.approx(0, abs=0.0000001)
