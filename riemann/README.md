# Riemann Hypothesis Proof Framework

A rigorous *framework* for attacking the Riemann Hypothesis (RH) — not a proof of it.

## Honest scope statement

The Riemann Hypothesis is an open problem (one of the Clay Millennium Prize
Problems). No proof is known, and this framework does not contain one. What it
does provide is the honest infrastructure a proof effort needs:

1. **The precise statement** and the analytic machinery around it
   (`zeta.py`): the completed zeta function ξ(s), the Riemann–Siegel
   Z-function, the zero-counting function N(T).
2. **Rigorous partial verification** (`verification.py`): a
   Turing-style check that *all* nontrivial zeros up to a given height T lie
   exactly on the critical line Re(s) = 1/2 — by comparing an unconditional
   count of zeros (the argument-principle formula for N(T)) against the number
   of sign changes of Z(t) actually found on the line. When the two counts
   agree, every zero up to height T is accounted for on the line.
3. **Proven-equivalent criteria** (`criteria.py`): reformulations that are
   *theorems* of the form "RH ⟺ X", so that proving X proves RH:
   - **Li's criterion** (Li, 1997): RH ⟺ λₙ ≥ 0 for all n ≥ 1.
   - **Robin's criterion** (Robin, 1984): RH ⟺ σ(n) < e^γ · n·log log n
     for all n > 5040.
   - **Lagarias's criterion** (Lagarias, 2002): RH ⟺ σ(n) ≤ Hₙ + e^{Hₙ}·log Hₙ
     for all n ≥ 1.
4. **A proof-obligation ledger** (`framework.py`): a directed graph of
   mathematical statements with statuses (`PROVEN`, `NUMERICALLY_SUPPORTED`,
   `OPEN`) and implication edges, each carrying a literature reference. Its
   `audit()` walks the graph and reports — truthfully — whether RH is proven
   and exactly which obligations remain. Today the audit reports **OPEN**.
   The ledger cannot be argued into reporting "proven" by numerical evidence:
   only a chain of `PROVEN` statements implying RH flips it.

## Why a framework and not a proof

Numerical verification (currently >10¹³ zeros on the line, Platt & Trudgian
2021) supports RH but can never prove it: a universally quantified statement
over infinitely many zeros is not settled by finitely many checks. Each module
here is explicit about which side of that line it sits on. The framework's
design rule: **computation may set a statement to `NUMERICALLY_SUPPORTED`,
never to `PROVEN`.** `PROVEN` requires a citation to an actual theorem.

## What a completed proof would look like in this framework

Any one of the following would discharge the root obligation:

- A proof that λₙ ≥ 0 for all n (Li's criterion node → PROVEN),
- A proof of Robin's inequality for all n > 5040,
- A proof of Lagarias's inequality for all n ≥ 1,
- A direct proof that ξ(s) ≠ 0 off the critical line (e.g. via a
  Hilbert–Pólya operator whose self-adjointness forces real spectrum),
- Any other statement added to the ledger with a `PROVEN` status and a
  proven implication edge to RH.

Until then, `framework.audit()` returns `OPEN`, and the test suite asserts
that it does — a regression test against wishful thinking.

## Usage

```bash
pip install -r riemann/requirements.txt

# Verify all zeros up to height T=100 lie on the critical line (rigorous, partial)
python -m riemann.verification --height 100

# Evaluate the equivalent criteria numerically
python -m riemann.criteria

# Print the proof-obligation audit
python -m riemann.framework
```

Run the tests:

```bash
python -m pytest riemann/tests -v
```

## Module map

| Module | Role | Epistemic status of its output |
|---|---|---|
| `zeta.py` | ζ, ξ, Z, θ, N(T) machinery | Exact definitions, high-precision numerics |
| `verification.py` | All zeros ≤ T on the line | Rigorous *for the verified range only* |
| `criteria.py` | Li / Robin / Lagarias checks | Evidence, never proof |
| `framework.py` | Obligation ledger + audit | Ground truth about what is proven |

## Key references

- B. Riemann, *Über die Anzahl der Primzahlen unter einer gegebenen Größe* (1859)
- A. M. Turing, *Some calculations of the Riemann zeta-function* (1953)
- X.-J. Li, *The positivity of a sequence of numbers and the Riemann hypothesis* (1997)
- G. Robin, *Grandes valeurs de la fonction somme des diviseurs et hypothèse de Riemann* (1984)
- J. C. Lagarias, *An elementary problem equivalent to the Riemann hypothesis* (2002)
- D. Platt, T. Trudgian, *The Riemann hypothesis is true up to 3·10¹²* (2021)
