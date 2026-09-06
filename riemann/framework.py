"""Proof-obligation ledger for the Riemann Hypothesis.

This module is the epistemic core of the framework. It models a proof
effort as a directed graph of mathematical statements:

* Each Statement has a Status: PROVEN (a published theorem, with citation),
  NUMERICALLY_SUPPORTED (finite computation consistent with truth), or
  OPEN (neither).
* An Implication records a PROVEN theorem of the form "premises => conclusion"
  (with citation). Only proven implications may be added; conjectural
  implications are themselves statements.
* audit(target) computes whether the target is proven, by closure: a
  statement is established iff its status is PROVEN, or some proven
  implication has all its premises established.

Design rule (enforced): computation can never set a statement to PROVEN.
The only way audit() will ever report RH as proven is for someone to add
a genuinely proven statement chain — i.e., to actually prove it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Status(enum.Enum):
    PROVEN = "proven"
    NUMERICALLY_SUPPORTED = "numerically supported"
    OPEN = "open"


@dataclass
class Statement:
    key: str
    text: str
    status: Status
    reference: str = ""

    def __post_init__(self) -> None:
        if self.status is Status.PROVEN and not self.reference:
            raise ValueError(
                f"Statement {self.key!r} marked PROVEN without a citation; "
                "a proof must be attributable"
            )


@dataclass
class Implication:
    premises: tuple[str, ...]
    conclusion: str
    reference: str

    def __post_init__(self) -> None:
        if not self.reference:
            raise ValueError("Implications must cite the theorem proving them")


@dataclass
class Ledger:
    statements: dict[str, Statement] = field(default_factory=dict)
    implications: list[Implication] = field(default_factory=list)

    def add(self, statement: Statement) -> None:
        if statement.key in self.statements:
            raise KeyError(f"Duplicate statement key {statement.key!r}")
        self.statements[statement.key] = statement

    def imply(self, premises: tuple[str, ...], conclusion: str, reference: str) -> None:
        for key in premises + (conclusion,):
            if key not in self.statements:
                raise KeyError(f"Unknown statement key {key!r}")
        self.implications.append(Implication(premises, conclusion, reference))

    def record_numerical_support(self, key: str) -> None:
        """The ONLY mutation computation is allowed to make: OPEN -> NUMERICALLY_SUPPORTED."""
        st = self.statements[key]
        if st.status is Status.OPEN:
            st.status = Status.NUMERICALLY_SUPPORTED

    def established(self) -> set[str]:
        """Fixed-point closure of PROVEN statements under proven implications."""
        proven = {k for k, s in self.statements.items() if s.status is Status.PROVEN}
        changed = True
        while changed:
            changed = False
            for imp in self.implications:
                if imp.conclusion not in proven and all(p in proven for p in imp.premises):
                    proven.add(imp.conclusion)
                    changed = True
        return proven

    def audit(self, target: str = "RH") -> tuple[Status, list[str]]:
        """Return the effective status of `target` and the open obligations.

        The open obligations are the premises of implications into the
        target (transitively) that are not yet established — i.e., what a
        proof still needs.
        """
        established = self.established()
        if target in established:
            return Status.PROVEN, []

        # collect unmet premises transitively feeding the target
        obligations: list[str] = []
        seen: set[str] = set()
        frontier = [target]
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            for imp in self.implications:
                if imp.conclusion == node:
                    for p in imp.premises:
                        if p not in established:
                            obligations.append(p)
                            frontier.append(p)
        status = self.statements[target].status
        # dedupe, preserve order
        unique = list(dict.fromkeys(obligations))
        return status, unique


def build_rh_ledger() -> Ledger:
    """The current, honest state of knowledge around RH."""
    led = Ledger()

    led.add(Statement(
        "RH",
        "Every nontrivial zero of the Riemann zeta function has real part 1/2.",
        Status.OPEN,
    ))
    led.add(Statement(
        "LI_ALL",
        "Li's coefficients satisfy lambda_n >= 0 for all n >= 1.",
        Status.OPEN,
    ))
    led.add(Statement(
        "ROBIN_ALL",
        "sigma(n) < e^gamma * n * ln ln n for all n > 5040.",
        Status.OPEN,
    ))
    led.add(Statement(
        "LAGARIAS_ALL",
        "sigma(n) <= H_n + e^{H_n} ln H_n for all n >= 1.",
        Status.OPEN,
    ))
    led.add(Statement(
        "ZEROS_ON_LINE_3E12",
        "All zeros with 0 < Im(s) <= 3·10^12 lie on the critical line.",
        Status.PROVEN,
        reference="Platt & Trudgian, Bull. LMS 53 (2021)",
    ))
    led.add(Statement(
        "ZERO_FREE_REGION",
        "zeta(s) != 0 in the classical zero-free region "
        "Re(s) >= 1 - c/log|Im(s)| (de la Vallée Poussin type).",
        Status.PROVEN,
        reference="de la Vallée Poussin (1899); Ford (2002) for explicit c",
    ))
    led.add(Statement(
        "POSITIVE_PROPORTION",
        "More than 40% of the zeros lie on the critical line.",
        Status.PROVEN,
        reference="Conrey (1989): at least 2/5 of zeros are on the line",
    ))

    # Proven equivalences: each single premise suffices to conclude RH.
    led.imply(("LI_ALL",), "RH", "Li, J. Number Theory 65 (1997)")
    led.imply(("ROBIN_ALL",), "RH", "Robin, J. Math. Pures Appl. 63 (1984)")
    led.imply(("LAGARIAS_ALL",), "RH", "Lagarias, Amer. Math. Monthly 109 (2002)")

    return led


def main() -> None:
    ledger = build_rh_ledger()
    status, obligations = ledger.audit("RH")
    print("=== Riemann Hypothesis proof audit ===\n")
    for key, st in ledger.statements.items():
        ref = f"  [{st.reference}]" if st.reference else ""
        print(f"  {st.status.value.upper():24s} {key}: {st.text}{ref}")
    print(f"\nEffective status of RH: {status.value.upper()}")
    if status is not Status.PROVEN:
        print("Remaining proof obligations (prove ANY one of these in full):")
        for ob in obligations:
            print(f"  - {ob}: {ledger.statements[ob].text}")
        print("\nNo amount of computation recorded in this ledger can change "
              "the verdict; only a theorem can.")


if __name__ == "__main__":
    main()
