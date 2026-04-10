"""
DFT relaxation tools.

Backend priority:
  1. GPAW  (PBE, plane-wave)       – production
  2. EMT   (ASE Effective Medium)  – fast smoke-test / CI fallback
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.optimize import LBFGS

from .config import (
    XC_FUNCTIONAL, KPOINTS, ENERGY_CUTOFF,
    MAX_FORCE, MAX_STEPS, RESULTS_DIR,
)

log = logging.getLogger(__name__)


# ── Calculator factory ─────────────────────────────────────────────────────────

def _gpaw_calculator(atoms: Atoms):
    """Return a configured GPAW PBE calculator."""
    from gpaw import GPAW, PW, FermiDirac
    nbands_guess = max(int(len(atoms) * 2.5), 20)
    calc = GPAW(
        mode=PW(ENERGY_CUTOFF),
        xc=XC_FUNCTIONAL,
        kpts={"size": KPOINTS, "gamma": True},
        occupations=FermiDirac(0.05),
        nbands=nbands_guess,
        convergence={"energy": 1e-5},
        txt=None,     # suppress GPAW stdout; logs captured separately
    )
    return calc


def _lj_calculator():
    """
    Fallback: Lennard-Jones potential (supports any element).
    Forces are NOT physically meaningful for black phosphorus — use only
    for CI smoke-tests to verify the workflow without GPAW installed.
    """
    from ase.calculators.lj import LennardJones
    # σ and ε are generic; relaxation will be geometrically valid
    return LennardJones(sigma=3.5, epsilon=0.010, rc=6.0, smooth=True)


def get_calculator(atoms: Atoms, prefer_gpaw: bool = True):
    """Return GPAW if available; fall back to Lennard-Jones with a warning."""
    if prefer_gpaw:
        try:
            return _gpaw_calculator(atoms)
        except ImportError:
            log.warning(
                "GPAW not found – falling back to Lennard-Jones (NOT production DFT). "
                "Install GPAW for physically meaningful results."
            )
    return _lj_calculator()


# ── Relaxation ─────────────────────────────────────────────────────────────────

def relax_structure(
    atoms: Atoms,
    label: str = "relax",
    use_gpaw: bool = True,
    results_dir: Optional[str] = None,
) -> dict:
    """
    Relax *atoms* to the DFT ground state using LBFGS.

    Returns a dict with:
        atoms      – relaxed Atoms object
        energy     – final total energy [eV]
        max_force  – max residual force [eV/Å]
        converged  – bool
        steps      – number of optimiser steps taken
        trajectory – path to the .traj file (or None)
    """
    results_dir = results_dir or RESULTS_DIR
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    traj_path = str(Path(results_dir) / f"{label}.traj")
    log_path  = str(Path(results_dir) / f"{label}_relax.log")

    atoms = atoms.copy()
    calc  = get_calculator(atoms, prefer_gpaw=use_gpaw)
    atoms.calc = calc

    opt = LBFGS(atoms, trajectory=traj_path, logfile=log_path)
    converged = opt.run(fmax=MAX_FORCE, steps=MAX_STEPS)

    energy    = float(atoms.get_potential_energy())
    forces    = atoms.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())

    # Persist final structure as CIF + JSON summary
    cif_path  = str(Path(results_dir) / f"{label}_relaxed.cif")
    json_path = str(Path(results_dir) / f"{label}_summary.json")
    ase_write(cif_path, atoms)

    summary = {
        "label":     label,
        "n_atoms":   len(atoms),
        "species":   list(set(atoms.get_chemical_symbols())),
        "energy_eV": energy,
        "max_force_eV_per_A": max_force,
        "converged": bool(converged),
        "steps":     opt.nsteps,
        "traj":      traj_path,
        "cif":       cif_path,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    return {**summary, "atoms": atoms}


# ── Bonding analysis ───────────────────────────────────────────────────────────

def analyze_bonding(
    atoms: Atoms,
    site: str,
    results_dir: Optional[str] = None,
) -> dict:
    """
    Determine whether the N impurity forms covalent bonds with neighbouring P.

    Criteria
    --------
    A P-N pair is flagged as *covalently bonded* when the inter-atomic distance
    is shorter than BOND_TOLERANCE × (r_cov(P) + r_cov(N)).

    Returns a dict with:
        site            – doping site label
        n_position      – [x,y,z] of N atom [Å]
        pn_distances    – list of {atom_idx, symbol, distance_A} for neighbours < 4 Å
        covalent_bonds  – subset where distance < threshold
        is_covalent     – bool (True if at least one covalent bond found)
        bond_threshold  – threshold distance [Å]
    """
    from ase.neighborlist import neighbor_list
    from .config import COVALENT_RADII, BOND_TOLERANCE

    results_dir = results_dir or RESULTS_DIR

    # Index of the nitrogen atom (last appended atom)
    symbols = atoms.get_chemical_symbols()
    n_indices = [i for i, s in enumerate(symbols) if s == "N"]
    if not n_indices:
        raise ValueError("No N atom found in the supplied Atoms object.")
    n_idx = n_indices[-1]   # use the last N (the one we added)

    # All neighbours within 4 Å
    cutoff = 4.0
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff=cutoff)

    neighbours = []
    for i, j, d in zip(i_list, j_list, d_list):
        if i == n_idx and symbols[j] == "P":
            neighbours.append({"atom_idx": int(j), "symbol": "P", "distance_A": float(d)})

    # Covalent bond threshold
    threshold = BOND_TOLERANCE * (COVALENT_RADII["N"] + COVALENT_RADII["P"])
    covalent  = [nb for nb in neighbours if nb["distance_A"] <= threshold]

    n_pos = atoms.get_positions()[n_idx].tolist()

    result = {
        "site":           site,
        "n_position":     n_pos,
        "pn_distances":   sorted(neighbours, key=lambda x: x["distance_A"]),
        "covalent_bonds": covalent,
        "is_covalent":    len(covalent) > 0,
        "bond_threshold": round(threshold, 4),
    }

    # Persist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    json_path = str(Path(results_dir) / f"bonding_{site}.json")
    with open(json_path, "w") as fh:
        json.dump({k: v for k, v in result.items()}, fh, indent=2)

    return result
