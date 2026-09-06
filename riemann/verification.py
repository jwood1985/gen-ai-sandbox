"""Turing-style verification that all zeros up to height T lie on the line.

The logic (Turing 1953, as used in every large-scale RH verification):

1. Count sign changes of the real-valued Z(t) on (0, T]. Each sign change
   is a zero of zeta ON the critical line, so the count is a lower bound
   for the number of critical-line zeros up to T.
2. Independently compute N(T), the total number of zeros (anywhere in the
   critical strip) with 0 < Im(rho) <= T, via the unconditional
   Riemann-von Mangoldt formula N(T) = theta(T)/pi + 1 + S(T).
3. If (sign changes found) == N(T), then every zero up to height T is
   simple? -- no: it means every zero up to T lies on the critical line
   (a zero off the line would appear in N(T) but produce no sign change,
   and so would its mirror image, making the counts disagree).

The conclusion "RH holds up to height T" is rigorous for the checked range,
modulo floating-point evaluation of Z and S; we use interval-free mpmath
numerics at elevated precision with sign evaluations far from zero
magnitude, which is the standard practical compromise. It says nothing
about zeros above T.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mpmath as mp

from riemann.zeta import N, Z


@dataclass(frozen=True)
class VerificationResult:
    height: float
    zeros_expected: int          # N(T), all zeros in the strip up to T
    sign_changes_found: int      # zeros located ON the line up to T
    all_on_critical_line: bool

    def summary(self) -> str:
        verdict = (
            f"VERIFIED: all {self.zeros_expected} zeros with 0 < Im(s) <= "
            f"{self.height} lie on the critical line Re(s) = 1/2"
            if self.all_on_critical_line
            else f"INCOMPLETE: expected {self.zeros_expected} zeros up to height "
            f"{self.height} but located only {self.sign_changes_found} on the "
            "line (refine the grid before drawing any conclusion)"
        )
        return verdict + " [this verifies a finite range only; it is not a proof of RH]"


def count_sign_changes(T: float, grid_points_per_unit: int = 8) -> int:
    """Count sign changes of Z(t) on (0, T] using a grid plus local refinement.

    The initial grid density must exceed the local zero density
    (~ log(T/2pi)/(2pi) zeros per unit height); on a miss, the count will
    fall short of N(T) and verify() will refine rather than overclaim.
    """
    T = mp.mpf(T)
    n_points = max(int(T * grid_points_per_unit), 16)
    step = T / n_points
    changes = 0
    prev_t = step
    prev_sign = mp.sign(Z(prev_t))
    for k in range(2, n_points + 1):
        t = k * step
        sign = mp.sign(Z(t))
        if sign == 0:
            # grid point exactly on a zero (measure zero, but be correct)
            changes += 1
            prev_sign = -prev_sign
        elif sign != prev_sign:
            changes += 1
        prev_t, prev_sign = t, sign
    return changes


def verify(T: float, max_refinements: int = 4) -> VerificationResult:
    """Verify that every zeta zero with 0 < Im(s) <= T lies on the critical line.

    Compares N(T) (argument principle, counts zeros anywhere in the strip)
    with the number of sign changes of Z (counts zeros on the line),
    refining the search grid until the counts agree or refinement is
    exhausted. Never overclaims: a shortfall yields all_on_critical_line
    False rather than an error-masking pass.
    """
    with mp.workdps(30):
        expected = N(T)
        density = 8
        found = count_sign_changes(T, density)
        refinements = 0
        while found < expected and refinements < max_refinements:
            density *= 4
            found = count_sign_changes(T, density)
            refinements += 1
        if found > expected:
            raise ArithmeticError(
                f"Found {found} sign changes but N({T}) = {expected}; "
                "argument-principle count and grid disagree upward, "
                "which indicates a numerical error"
            )
        return VerificationResult(
            height=float(T),
            zeros_expected=expected,
            sign_changes_found=found,
            all_on_critical_line=(found == expected),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify all zeta zeros up to a height lie on the critical line"
    )
    parser.add_argument("--height", type=float, default=100.0,
                        help="verify up to Im(s) = HEIGHT (default 100)")
    args = parser.parse_args()
    result = verify(args.height)
    print(result.summary())


if __name__ == "__main__":
    main()
