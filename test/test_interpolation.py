from interpolation import lagrange_interpolating


def test_lagrange_interpolating():
    f = lagrange_interpolating([(0, 1), (2, 2), (3, 4)])
    assert f(0) == 1
    assert f(2) == 2
    assert f(3) == 4
