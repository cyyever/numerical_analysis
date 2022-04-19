import math
from typing import Callable

import numpy as np


def DFF_interpolation(f: Callable, n: int, c: float, d: float) -> Callable:
    dff_y = np.fft.fft(f(np.array([i / n for i in range(n)])), norm=None)
    a = np.real(dff_y)
    b = np.imag(dff_y)

    def P(t: float):
        u = (t - c) / (d - c)
        r = a[0] + a[n // 2] * np.cos(n * np.pi * u)
        for k in range(1, n // 2):
            r += 2 * (
                a[k] * np.cos(2 * k * np.pi * u) - b[k] * np.sin(2 * k * math.pi * u)
            )
        return r / n

    return P
