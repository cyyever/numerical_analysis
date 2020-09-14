import numpy as np
import pytest


def power_method(A: np.matrix, epsilon=0.00001, max_iteration=1000):
    last_eigenvalue = None
    cur_eigenvalue = None
    v = np.random.rand(A.shape[0], 1)
    for _ in range(max_iteration):
        last_eigenvalue = cur_eigenvalue
        v = A @ v
        cur_eigenvalue = v[np.argmax(np.absolute(v))]
        v = v / cur_eigenvalue
        if last_eigenvalue is not None and cur_eigenvalue is not None:
            if abs(cur_eigenvalue - last_eigenvalue) < epsilon:
                break
    return (cur_eigenvalue, v)


A = np.matrix([[6, 5], [1, 2]])
max_eigenvalue, eigenvector = power_method(A)
assert pytest.approx(max_eigenvalue, 7)
