"""Proven-equivalent reformulations of the Riemann Hypothesis.

Each criterion here is a THEOREM of the form "RH is true if and only if X".
Proving X (for all n, an infinite family) would prove RH. The functions
below evaluate X for finitely many n, which yields evidence, never proof.
A single counterexample, however, would disprove RH outright — so these
checks are genuine falsification attempts.

Criteria implemented:

* Li's criterion (Li 1997): RH  <=>  lambda_n >= 0 for all n >= 1, where
  lambda_n = sum over nontrivial zeros rho of [1 - (1 - 1/rho)^n], the sum
  taken over rho and 1-rho paired (equivalently rho and conjugate(rho)).

* Robin's criterion (Robin 1984): RH  <=>  sigma(n) < e^gamma * n * ln(ln n)
  for all n > 5040 (sigma = sum-of-divisors function).

* Lagarias's criterion (Lagarias 2002): RH  <=>
  sigma(n) <= H_n + exp(H_n) * ln(H_n) for all n >= 1, where H_n is the
  n-th harmonic number. (Equality only at n = 1.)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import mpmath as mp
from sympy import divisor_sigma


@lru_cache(maxsize=None)
def _cached_zero(k: int, dps: int) -> mp.mpc:
    with mp.workdps(dps):
        return mp.zetazero(k)

# Colossally abundant numbers are where Robin's inequality is tightest;
# checking only these would be the smart search for a counterexample, but
# for a framework demo we scan all n in range.
ROBIN_THRESHOLD = 5040


def li_coefficient(n: int, num_zeros: int = 200) -> mp.mpf:
    """Approximate Li's lambda_n from the first num_zeros zero pairs.

    lambda_n = sum_rho [1 - (1 - 1/rho)^n], zeros paired with conjugates so
    each pair contributes 2*Re[1 - (1 - 1/rho)^n]. Truncation to num_zeros
    pairs is accurate for n small relative to num_zeros; this is a numerical
    probe, not a certified evaluation.
    """
    if n < 1:
        raise ValueError("Li coefficients are defined for n >= 1")
    with mp.workdps(30):
        total = mp.mpf(0)
        for k in range(1, num_zeros + 1):
            rho = _cached_zero(k, 30)
            term = 1 - (1 - 1 / rho) ** n
            total += 2 * term.real  # rho paired with its conjugate
        return total


def check_li_positivity(max_n: int, num_zeros: int = 200) -> dict[int, float]:
    """Evaluate lambda_1..lambda_max_n; returns {n: lambda_n} for inspection.

    Raises AssertionError if any computed lambda_n is negative beyond
    truncation noise, since that would be evidence AGAINST RH and must not
    pass silently.
    """
    values: dict[int, float] = {}
    for n in range(1, max_n + 1):
        lam = li_coefficient(n, num_zeros)
        values[n] = float(lam)
        if lam < -mp.mpf("1e-6"):
            raise AssertionError(
                f"lambda_{n} = {lam} < 0: potential RH counterexample — "
                "recompute with more zeros and higher precision immediately"
            )
    return values


def robin_inequality_holds(n: int) -> bool:
    """Robin: sigma(n) < e^gamma * n * ln(ln n), meaningful for n > 5040."""
    if n <= ROBIN_THRESHOLD:
        raise ValueError("Robin's criterion applies only for n > 5040")
    with mp.workdps(30):
        lhs = mp.mpf(int(divisor_sigma(n)))
        rhs = mp.exp(mp.euler) * n * mp.log(mp.log(n))
        return lhs < rhs


def lagarias_inequality_holds(n: int) -> bool:
    """Lagarias: sigma(n) <= H_n + exp(H_n) * ln(H_n) for n >= 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return True  # sigma(1) = 1 = H_1 + e^{H_1} ln(H_1) = 1 + e*0
    with mp.workdps(30):
        h = mp.harmonic(n)
        lhs = mp.mpf(int(divisor_sigma(n)))
        rhs = h + mp.exp(h) * mp.log(h)
        return lhs <= rhs


def scan_robin(ns: Iterable[int]) -> list[int]:
    """Return the list of n (each > 5040) violating Robin's inequality.

    A nonempty result would disprove RH; expected result is [].
    """
    return [n for n in ns if not robin_inequality_holds(n)]


def scan_lagarias(max_n: int) -> list[int]:
    """Return the list of 1 <= n <= max_n violating Lagarias's inequality."""
    return [n for n in range(1, max_n + 1) if not lagarias_inequality_holds(n)]


def main() -> None:
    print("Li coefficients lambda_1..lambda_8 (truncated to 200 zero pairs):")
    for n, lam in check_li_positivity(8).items():
        print(f"  lambda_{n} = {lam:.6f}  {'>= 0 OK' if lam >= 0 else 'NEGATIVE'}")

    lagarias_violations = scan_lagarias(2000)
    print(f"Lagarias criterion, n <= 2000: "
          f"{'no violations' if not lagarias_violations else lagarias_violations}")

    robin_sample = range(ROBIN_THRESHOLD + 1, ROBIN_THRESHOLD + 2001)
    robin_violations = scan_robin(robin_sample)
    print(f"Robin criterion, 5041 <= n <= 7040: "
          f"{'no violations' if not robin_violations else robin_violations}")

    print("\nAll checks are finite-range evidence; none of them proves RH.")


if __name__ == "__main__":
    main()
