"""Tests for the Riemann Hypothesis proof framework.

These tests pin down both the mathematics (known zeros, counting formula,
criteria on finite ranges) and the epistemics (the ledger must report RH
as OPEN — a test that fails the day someone actually proves it, which is
the correct behavior for a regression suite against overclaiming).
"""

import mpmath as mp
import pytest

from riemann.criteria import (
    check_li_positivity,
    lagarias_inequality_holds,
    robin_inequality_holds,
    scan_lagarias,
    scan_robin,
)
from riemann.framework import Ledger, Statement, Status, build_rh_ledger
from riemann.verification import verify
from riemann.zeta import N, Z, theta, xi, zero


class TestZetaMachinery:
    def test_first_zero_matches_known_value(self):
        with mp.workdps(25):
            rho = zero(1)
            assert abs(rho.real - mp.mpf("0.5")) < mp.mpf("1e-20")
            assert abs(rho.imag - mp.mpf("14.134725141734693790")) < mp.mpf("1e-15")

    def test_xi_functional_equation(self):
        with mp.workdps(25):
            for s in (mp.mpc(0.3, 7.2), mp.mpc(2.5, -1.0), mp.mpc(0.9, 25.0)):
                assert abs(xi(s) - xi(1 - s)) < mp.mpf("1e-18")

    def test_Z_vanishes_at_zero_ordinate(self):
        with mp.workdps(25):
            gamma1 = zero(1).imag
            assert abs(Z(gamma1)) < mp.mpf("1e-18")

    def test_N_counts_zeros_correctly(self):
        # gamma_1 ~ 14.13, gamma_2 ~ 21.02, gamma_3 ~ 25.01
        with mp.workdps(25):
            assert N(20) == 1
            assert N(30) == 3
            assert N(50) == 10  # gamma_10 ~ 49.77, gamma_11 ~ 52.97

    def test_theta_is_real_and_increasing_for_large_t(self):
        with mp.workdps(25):
            assert theta(50) < theta(60)


class TestVerification:
    def test_all_zeros_up_to_100_on_critical_line(self):
        result = verify(100)
        assert result.zeros_expected == 29  # N(100) = 29, a classical value
        assert result.sign_changes_found == 29
        assert result.all_on_critical_line

    def test_summary_never_claims_full_proof(self):
        result = verify(50)
        assert "not a proof of RH" in result.summary()


class TestCriteria:
    def test_li_coefficients_positive_small_n(self):
        values = check_li_positivity(max_n=6, num_zeros=150)
        assert all(lam > 0 for lam in values.values())
        # lambda_1 = 1 + gamma/2 - (1/2) ln(4 pi) ~ 0.0230957; the truncated
        # zero sum converges like log(T)/T, so 150 pairs carries a tail of
        # a few times 1e-3 — the tolerance reflects that truncation error.
        assert values[1] == pytest.approx(0.0230957, abs=5e-3)

    def test_robin_holds_on_sample(self):
        assert scan_robin(range(5041, 5241)) == []
        # 5040 itself violates the inequality (that is why it is excluded)
        with pytest.raises(ValueError):
            robin_inequality_holds(5040)

    def test_lagarias_holds_up_to_1000(self):
        assert scan_lagarias(1000) == []

    def test_lagarias_tight_at_small_n(self):
        assert lagarias_inequality_holds(1)
        assert lagarias_inequality_holds(2)


class TestLedgerEpistemics:
    def test_rh_is_open(self):
        """RH is an open problem. If this test ever fails, celebrate carefully."""
        ledger = build_rh_ledger()
        status, obligations = ledger.audit("RH")
        assert status is Status.OPEN
        assert set(obligations) == {"LI_ALL", "ROBIN_ALL", "LAGARIAS_ALL"}

    def test_computation_cannot_prove(self):
        ledger = build_rh_ledger()
        ledger.record_numerical_support("RH")
        status, _ = ledger.audit("RH")
        assert status is Status.NUMERICALLY_SUPPORTED
        assert status is not Status.PROVEN
        assert "RH" not in ledger.established()

    def test_proven_requires_citation(self):
        with pytest.raises(ValueError):
            Statement("X", "some claim", Status.PROVEN)

    def test_implication_requires_citation(self):
        ledger = build_rh_ledger()
        with pytest.raises(ValueError):
            ledger.imply(("LI_ALL",), "RH", "")

    def test_closure_propagates_a_real_proof(self):
        """If (hypothetically) an equivalent criterion were proven, the audit flips."""
        ledger = build_rh_ledger()
        ledger.statements["LI_ALL"].status = Status.PROVEN
        ledger.statements["LI_ALL"].reference = "hypothetical future theorem"
        status, obligations = ledger.audit("RH")
        assert status is Status.PROVEN
        assert obligations == []
