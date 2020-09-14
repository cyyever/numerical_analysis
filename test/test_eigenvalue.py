import numpy as np
import pytest

from eigenvalue import power_method


def test_power_method():
    A = np.asarray([[6, 5], [1, 2]])
    max_eigenvalue, eigenvector = power_method(A)
    assert pytest.approx(max_eigenvalue, 7)
