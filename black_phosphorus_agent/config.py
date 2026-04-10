"""Configuration for the Black Phosphorus DFT Agent."""

# ── Supercell ──────────────────────────────────────────────────────────────────
# Black phosphorus primitive cell has 4 P atoms.
# We build the supercell as (nx, ny, nz) repetitions of the primitive cell.
# 3×4×1 = 48 atoms  (closest tiling to nATOM=50 without exceeding it)
N_ATOM = 50           # target atom count (actual count may be nearest supercell)
SUPERCELL = (3, 4, 1) # gives 48 P atoms; change to (4, 4, 1)=64 if you want >50

# ── DFT settings ──────────────────────────────────────────────────────────────
XC_FUNCTIONAL   = "PBE"          # exchange-correlation functional
KPOINTS         = (2, 2, 4)      # Monkhorst-Pack k-point grid
ENERGY_CUTOFF   = 400            # plane-wave cutoff [eV]   (GPAW)
MAX_FORCE       = 0.05           # convergence threshold [eV/Å]
MAX_STEPS       = 200            # max BFGS steps per relaxation

# ── Nitrogen doping ───────────────────────────────────────────────────────────
# N covalent radius 0.71 Å, P covalent radius 1.07 Å → bond threshold ~1.82 Å
COVALENT_RADII  = {"N": 0.71, "P": 1.07}   # Å
BOND_TOLERANCE  = 1.20           # scale factor on sum of covalent radii

# Offset (Å) used when placing N at each site
BRIDGE_OFFSET      = 1.50        # above midpoint of P-P bond
INTERSTITIAL_OFFSET = 2.50       # in the interlayer van-der-Waals gap
ZIGZAG_OFFSET      = 1.50        # along the zigzag chain (b-axis)
ARMCHAIR_OFFSET    = 1.50        # along the armchair direction (a-axis)

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "dft_results"
