"""
Build hBN spin bath for V_B⁻ CCE simulations using PyCCE's BathCell.

Uses PyCCE's native BathCell.from_ase() workflow:
  1. Build hBN unit cell with ASE (P6₃/mmc, #194)
  2. Create BathCell, set isotope concentrations
  3. Generate supercell with vacancy via gen_supercell()
  4. Override first-shell N hyperfine tensors with literature DFT values

PyCCE units: couplings in kHz, distances in Å, time in ms, field in Gauss.

Usage:
    from build_bath import build_bath
    bath, vac_pos = build_bath(
        B10_fraction=0.199, N14_fraction=0.996,
        supercell_size=200, seed=42,
    )
"""

import numpy as np
import yaml
from pathlib import Path

try:
    from ase.spacegroup import crystal
except ImportError:
    raise ImportError("ASE is required: pip install ase")

try:
    import pycce as pc
except ImportError:
    raise ImportError("PyCCE is required: pip install pycce")


CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_physics_config() -> dict:
    with open(CONFIG_DIR / "physics.yaml") as f:
        return yaml.safe_load(f)


def build_hbn_unitcell():
    """
    Build hBN unit cell with ASE.

    Space group P6₃/mmc (#194), a = 2.504 Å, c = 6.661 Å.
    Wyckoff positions: B at 2c (1/3, 2/3, 1/4), N at 2d (1/3, 2/3, 3/4).
    Ref: Pease, Acta Crystallogr. 5, 356 (1952)
    """
    return crystal(
        symbols=["B", "N"],
        basis=[(1 / 3, 2 / 3, 1 / 4), (1 / 3, 2 / 3, 3 / 4)],
        spacegroup=194,
        cellpar=[2.504, 2.504, 6.661, 90, 90, 120],
    )


def build_bath(
    B10_fraction: float = 0.199,
    N14_fraction: float = 0.996,
    supercell_size: float = 200.0,
    seed: int = 42,
) -> tuple:
    """
    Build the nuclear spin bath for V_B⁻ in hBN.

    Uses PyCCE's BathCell to generate a supercell with stochastic isotope
    placement, then removes the boron vacancy site.

    Parameters
    ----------
    B10_fraction : float
        Fraction of B sites that are ¹⁰B (I=3). Rest are ¹¹B (I=3/2).
    N14_fraction : float
        Fraction of N sites that are ¹⁴N (I=1). Rest are ¹⁵N (I=1/2).
    supercell_size : float
        Linear dimension of the supercell in Å. Should be >> r_bath.
    seed : int
        Random seed for reproducible isotope placement.

    Returns
    -------
    bath : BathArray
        PyCCE bath array with isotope names, positions, and spin properties.
    vacancy_position : ndarray
        Cartesian position of the removed B atom (the defect center).
    """
    atoms = build_hbn_unitcell()
    cell = pc.BathCell.from_ase(atoms)

    # Override default isotope concentrations
    # PyCCE's common_isotopes has natural abundances by default;
    # we set custom concentrations for the isotope sweep
    B11_fraction = 1.0 - B10_fraction
    N15_fraction = 1.0 - N14_fraction

    # Set boron isotope concentrations
    cell.add_isotopes(("10B", B10_fraction))
    cell.add_isotopes(("11B", B11_fraction))

    # Set nitrogen isotope concentrations
    cell.add_isotopes(("14N", N14_fraction))
    cell.add_isotopes(("15N", N15_fraction))

    # Generate supercell, removing one B atom to create V_B⁻
    # The B atom at Wyckoff 2c has fractional coords (1/3, 2/3, 1/4)
    bath = cell.gen_supercell(
        supercell_size,
        remove=("B", [1 / 3, 2 / 3, 1 / 4]),
        seed=seed,
    )

    # Vacancy is at the origin after gen_supercell's remove
    vacancy_position = np.zeros(3)

    return bath, vacancy_position


def set_first_shell_hyperfine(
    bath,
    vacancy_position: np.ndarray,
    physics: dict,
    nn_cutoff: float = 2.0,
):
    """
    Override point-dipole hyperfine for nearest-neighbor N atoms with
    literature DFT values.

    The 3 NN nitrogen atoms (~1.45 Å from the vacancy) have strong
    hyperfine coupling that the point-dipole approximation gets wrong.

    Parameters
    ----------
    bath : BathArray
        PyCCE bath array (modified in place).
    vacancy_position : ndarray
        Position of the vacancy.
    physics : dict
        Physics config with first_shell hyperfine tensors (in MHz).
    nn_cutoff : float
        Distance cutoff in Å to identify nearest neighbors.
    """
    distances = np.linalg.norm(bath["xyz"] - vacancy_position, axis=1)
    nn_mask = distances < nn_cutoff

    nn_count = np.sum(nn_mask)
    if nn_count != 3:
        raise ValueError(
            f"Expected 3 nearest-neighbor N atoms within {nn_cutoff} Å, "
            f"found {nn_count}. Check supercell construction."
        )

    fs = physics["first_shell"]

    for idx in np.where(nn_mask)[0]:
        name = bath["N"][idx]
        if "14N" in name or "N14" in name:
            # ¹⁴N hyperfine: [47, 47, 85] MHz → kHz
            hf = fs["N14"]["hyperfine"]
            bath["A"][idx] = np.diag([h * 1000.0 for h in hf])  # MHz → kHz
        elif "15N" in name or "N15" in name:
            # ¹⁵N hyperfine: [65.9, 65.9, 119.2] MHz → kHz
            hf = fs["N15"]["hyperfine"]
            bath["A"][idx] = np.diag([h * 1000.0 for h in hf])


def bath_census(bath) -> dict:
    """Count isotopes in the bath for logging."""
    from collections import Counter
    return dict(Counter(bath["N"]))


if __name__ == "__main__":
    physics = load_physics_config()

    print("=== Building natural-abundance hBN bath ===")
    bath, vac_pos = build_bath(
        B10_fraction=0.199,
        N14_fraction=0.996,
        supercell_size=200,
        seed=42,
    )
    print(f"Total bath spins: {len(bath)}")
    print(f"Isotope census: {bath_census(bath)}")

    # Identify first-shell
    dists = np.linalg.norm(bath["xyz"] - vac_pos, axis=1)
    nn = np.where(dists < 2.0)[0]
    print(f"First-shell N atoms: {len(nn)}")
    for i in nn:
        print(f"  {bath['N'][i]} at {bath['xyz'][i]}, dist={dists[i]:.3f} Å")

    set_first_shell_hyperfine(bath, vac_pos, physics)
    print("First-shell hyperfine tensors set from literature DFT values.")
