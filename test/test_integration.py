import math

import pytest
from integration import simpson_rule, trapezoid_rule


def test_trapezoid_rule():
    assert trapezoid_rule(math.log, 1, 2) == pytest.approx(0.3466, abs=0.001)


def test_simpson_rule():
    assert simpson_rule(math.log, 1, 2) == pytest.approx(0.3858, abs=0.001)
