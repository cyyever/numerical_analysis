import numpy as np
import pytest
from eigenvalue import (
    inverse_power_iteration_method,
    power_iteration_method,
    rayleigh_quotient_iteration_method,
)


def test_power_iteration_method():
    A = np.asarray([[6, 5], [1, 2]])
    max_eigenvalue, _ = power_iteration_method(A)
    assert pytest.approx(max_eigenvalue) == 7


def test_inverse_power_iteration_method():
    A = np.asarray([[6, 5], [1, 2]])
    min_eigenvalue, _ = inverse_power_iteration_method(A)
    assert pytest.approx(min_eigenvalue) == 1


def test_rayleigh_quotient_iteration_method():
    A = np.asarray([[6, 5], [1, 2]])
    min_eigenvalue, _ = rayleigh_quotient_iteration_method(A)
    assert pytest.approx(min_eigenvalue) == 1
