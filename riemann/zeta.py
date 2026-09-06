"""Analytic machinery around the Riemann zeta function.

Definitions used throughout the framework:

* zeta(s): the Riemann zeta function, analytically continued to C \\ {1}.
* xi(s) = (1/2) s (s-1) pi^(-s/2) Gamma(s/2) zeta(s), the completed zeta
  function. It is entire, satisfies xi(s) = xi(1-s), and its zeros are
  exactly the nontrivial zeros of zeta. RH says: all zeros of xi have
  Re(s) = 1/2.
* Z(t) = e^{i theta(t)} zeta(1/2 + it), the Riemann-Siegel Z-function.
  Z is real-valued for real t, and its sign changes locate zeros of zeta
  on the critical line.
* theta(t): the Riemann-Siegel theta function.
* N(T): the number of zeros rho = beta + i*gamma of zeta with
  0 < gamma <= T, counted with multiplicity. Unconditionally,
  N(T) = theta(T)/pi + 1 + S(T) with S(T) = (1/pi) arg zeta(1/2 + iT).

All numerics use mpmath at a caller-configurable working precision.
"""

from __future__ import annotations

import mpmath as mp


def zeta(s) -> mp.mpc:
    """Riemann zeta function at s (analytic continuation)."""
    return mp.zeta(s)


def xi(s) -> mp.mpc:
    """Completed zeta function xi(s), entire with xi(s) = xi(1-s)."""
    s = mp.mpmathify(s)
    return 0.5 * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def Z(t) -> mp.mpf:
    """Riemann-Siegel Z-function; real for real t, sign changes = zeros on the line."""
    return mp.siegelz(t)


def theta(t) -> mp.mpf:
    """Riemann-Siegel theta function."""
    return mp.siegeltheta(t)


def zero(n: int) -> mp.mpc:
    """The n-th nontrivial zero of zeta in the upper half-plane (n >= 1)."""
    return mp.zetazero(n)


def S(T) -> mp.mpf:
    """S(T) = (1/pi) arg zeta(1/2 + iT), the argument along the standard path.

    Computed by numerically continuing arg zeta from 2 + iT (where
    |zeta - 1| < 1 guarantees the principal branch) leftward to 1/2 + iT,
    tracking the branch across small steps. T must not be the ordinate of
    a zero.
    """
    T = mp.mpf(T)
    steps = 64
    prev = mp.arg(mp.zeta(2 + 1j * T))
    total = prev
    for k in range(1, steps + 1):
        sigma = 2 - (mp.mpf(3) / 2) * k / steps  # 2 -> 1/2
        cur = mp.arg(mp.zeta(sigma + 1j * T))
        delta = cur - prev
        # unwrap: choose the branch nearest the previous point
        while delta > mp.pi:
            delta -= 2 * mp.pi
        while delta < -mp.pi:
            delta += 2 * mp.pi
        total += delta
        prev = cur
    return total / mp.pi


def N(T) -> int:
    """N(T): number of zeros with 0 < Im(rho) <= T, via the argument principle.

    Uses the unconditional Riemann-von Mangoldt exact formula
    N(T) = theta(T)/pi + 1 + S(T). The result is an integer whenever T is
    not the ordinate of a zero; we round the (near-integer) value and check
    it is close, so a grossly wrong branch count raises instead of lying.
    """
    T = mp.mpf(T)
    value = theta(T) / mp.pi + 1 + S(T)
    nearest = int(mp.nint(value))
    if abs(value - nearest) > mp.mpf("0.25"):
        raise ArithmeticError(
            f"N({T}) = {value} is not near an integer; "
            "T may be too close to a zero ordinate, or precision is insufficient"
        )
    return nearest
