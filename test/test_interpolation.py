from interpolation import lagrange_interpolating, newton_divided_difference


def test_lagrange_interpolating():
    f = lagrange_interpolating([(0, 1), (2, 2), (3, 4)])
    assert f(0) == 1
    assert f(2) == 2
    assert f(3) == 4


def test_newton_divided_difference():
    f = newton_divided_difference([(0, 1), (2, 2), (3, 4)])
    assert f(0) == 1
    assert f(2) == 2
    assert f(3) == 4
