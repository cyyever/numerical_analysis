from polynomial import Polynomial


def test_polynomial():
    f = Polynomial([1, 2, 3])
    assert f(1) == 6
    assert f(2) == 17
    g = f
    h = f + g
    assert h(1) == f(1) * 2
    assert h(2) == f(2) * 2
    h = f - g
    assert h(1) == 0
    assert h(2) == 0

    h = f * g
    assert h(1) == f(1) * g(1)
    assert h(2) == f(2) * g(2)
