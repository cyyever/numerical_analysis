#!/usr/bin/env python3

import numpy

from iterative_method import fixed_point_iteration


def conjugate_gradient(A, b, **kwargs):
    """A must be positive definite"""
    assert numpy.all(A == A.T)

    x = numpy.zeros_like(b)
    r = b - A @ x
    d = r
    delta = r @ r

    def conjugate_gradient_impl(x):
        r"""
        Implements Conjugate Gradient illustrated by
        An Introduction to the Conjugate Gradient Method Without the Agonizing Pain
        """
        nonlocal r, d, delta
        q = A @ d
        alpha = delta / (d @ q)
        x = x + alpha * d
        r = r - alpha * q
        old_delta = delta
        delta = r @ r
        d = r + (delta / old_delta) * d
        return x

    return fixed_point_iteration(conjugate_gradient_impl, x=x, **kwargs)


def jacobi_preconditioner(A):
    return numpy.diagflat(numpy.diagonal(A))


def preconditioned_conjugate_gradient(A, b, M, **kwargs):
    """A must be positive definite"""
    assert numpy.all(A == A.T)

    x = numpy.zeros_like(b)
    r = b - A @ x
    z = numpy.linalg.solve(M, r)
    d = z
    delta = r @ z

    def conjugate_gradient_impl(x):
        r"""
        Implements Conjugate Gradient illustrated by
        An Introduction to the Conjugate Gradient Method Without the Agonizing Pain
        """
        nonlocal r, d, delta
        q = A @ d
        alpha = delta / (d @ q)
        x = x + alpha * d
        r = r - alpha * q
        z = numpy.linalg.solve(M, r)
        old_delta = delta
        delta = r @ z
        d = z + (delta / old_delta) * d
        return x

    return fixed_point_iteration(conjugate_gradient_impl, x=x, **kwargs)
