from polynomial import Polynomial


def test_polynomial():
    f = Polynomial([1, 2, 3])
    assert f(1) == 6
    assert f(2) == 17
