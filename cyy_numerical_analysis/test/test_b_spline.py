from b_spline import BSpline, KnotVector


def test_eval():
    spline = BSpline(
        points=[0, 1, 2, 3, 4, 5], knot_vector=[0, 0, 0, 0, 1, 4, 5, 5, 5, 5], degree=3
    )
    assert spline(0) == 0
    assert spline(5) == 5


def test_basis_function():
    knot_vector = KnotVector([-1, 0, 0, 1, 1, 2, 3, 4], degree=2)
    assert knot_vector.evaluate_base_function(t=1, index=2, degree=2) == 1
