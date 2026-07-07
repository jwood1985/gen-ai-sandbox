# spin-cycle: CCE coherence simulations for V_B⁻ in hBN

## Mission
Predict T2 vs isotopic composition. Output = 2D coherence matrix
(isotope config × DD sequence) + purity-vs-payoff curves.

## Non-negotiables
1. VALIDATION GATE: reproduce literature natural-abundance Hahn T2
   within 2x before any sweep results are reported as findings.
2. First-shell N atoms are NOT bath. Treat explicitly.
3. Quadrupole tensors ON for I≥1 species. Always.
4. Every physics parameter in config/physics.yaml carries a citation
   comment. No uncited numbers.
5. Convergence before production: CCE order (2 vs 3), bath radius
   (sweep until T2 changes <5%), pulse points per decay curve.
6. Ensemble stats: never report single-seed T2. Median of ≥50 seeds,
   report IQR.
7. Units: report T2 in μs, fields in mT, couplings in MHz. Stretch
   exponent n reported alongside every T2 (decay shape is physics,
   not a nuisance parameter).

## Workflow
- Pull jobs from queue/, run, write results/raw/, update queue state.
- On anomaly (non-monotonic convergence, fit failure, T2 outside
  0.1–1000 μs sanity window): halt job, log to AGENT-MEMORY.md,
  notify, do not silently retry.
- Append learnings (converged parameters, PyCCE gotchas, runtime
  scaling data) to AGENT-MEMORY.md as discovered.

## What NOT to do
- No DFT. Hyperfine/ZFS come from literature in Phase 1.
- No spin-lattice (T1) modeling. Nuclear bath T2 only. State this
  limitation in every summary.
- No result interpretation beyond the data. Rankings yes,
  synthesis-route recommendations no (that's Josh's call).

## Project structure
```
config/          Physics parameters, isotope matrix, sweep definitions
src/             Core simulation modules
queue/           Job queue (JSON files, crash-recoverable)
results/raw/     Per-job coherence curves
results/fits/    Extracted T2 + stretch exponents
results/figures/ Proposal-ready plots
notebooks/       Exploration only; nothing load-bearing
```

## Running
```bash
# Phase 0: smoke test
python src/run_cce.py --config config/sweeps.yaml --job smoke_test

# Phase 1: convergence + validation
python src/converge.py --config config/sweeps.yaml

# Phase 2: full sweep
python src/ensemble.py --config config/sweeps.yaml --seeds 50

# Phase 3: analysis
python src/analyze.py --results-dir results/raw --output results/fits
```
