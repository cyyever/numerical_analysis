import math

import pytest
from integration import (composite_simpson_rule, composite_trapezoid_rule,
                         simpson_rule, trapezoid_rule)


def test_trapezoid_rule():
    assert trapezoid_rule(math.log, 1, 2) == pytest.approx(0.3466, abs=0.001)


def test_composite_trapezoid_rule():
    assert composite_trapezoid_rule(math.log, 1, 2, m=4) == pytest.approx(
        0.3837, abs=0.001
    )


def test_simpson_rule():
    assert simpson_rule(math.log, 1, 2) == pytest.approx(0.3858, abs=0.001)


def test_composite_simpson_rule():
    assert composite_simpson_rule(math.log, 1, 2,m=4) == pytest.approx(0.3863, abs=0.001)
