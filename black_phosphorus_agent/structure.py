"""
Black phosphorus structure builder.

Orthorhombic black phosphorus (space group Cmce, #64).
Experimental lattice parameters (Å):
    a = 3.3136,  b = 4.3763,  c = 10.4788
4 atoms per primitive cell at Wyckoff 8f positions.
"""

import numpy as np
from ase import Atoms
from ase.build import make_supercell

# ── Primitive cell ─────────────────────────────────────────────────────────────
_A = 3.3136   # Å
_B = 4.3763
_C = 10.4788

# Fractional coordinates of the 4-atom primitive cell (Wyckoff 8f, x~0.0, y~0.10, z~0.08)
_FRAC_COORDS = np.array([
    [0.0000,  0.1004,  0.0816],   # P1
    [0.5000,  0.6004,  0.0816],   # P2
    [0.0000,  0.8996,  0.4184],   # P3  (= 1 - P1_y, 0.5 - P1_z shifted)
    [0.5000,  0.3996,  0.4184],   # P4
])


def build_primitive_cell() -> Atoms:
    """Return the 4-atom orthorhombic primitive cell of black phosphorus."""
    cell = np.diag([_A, _B, _C])
    positions = _FRAC_COORDS @ cell
    atoms = Atoms(
        symbols="P4",
        positions=positions,
        cell=cell,
        pbc=True,
    )
    return atoms


def build_supercell(nx: int = 3, ny: int = 4, nz: int = 1) -> Atoms:
    """
    Build an (nx × ny × nz) supercell of black phosphorus.

    Default 3×4×1 gives 48 P atoms — the largest tiling with ≤50 atoms.
    """
    prim = build_primitive_cell()
    P = np.diag([nx, ny, nz])
    sc = make_supercell(prim, P)
    sc.info["supercell"] = (nx, ny, nz)
    return sc


def get_nitrogen_site(atoms: Atoms, site: str) -> np.ndarray:
    """
    Return a Cartesian position (3,) for placing an N impurity.

    Parameters
    ----------
    atoms : Atoms  (relaxed or unrelaxed black phosphorus supercell)
    site  : one of 'bridge' | 'interstitial' | 'zigzag' | 'armchair'

    Returns
    -------
    pos : np.ndarray, shape (3,), Cartesian coordinates in Å
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell()

    # Centre-of-cell reference point
    cx, cy, cz = cell.sum(axis=0) / 2.0

    if site == "bridge":
        # Midpoint of the shortest P-P bond + small perpendicular offset.
        # Use 3.5 Å cutoff: nearest-neighbour P-P in BP is ~2.74 Å (within-layer)
        from ase.neighborlist import neighbor_list
        i_idx, j_idx, d = neighbor_list("ijd", atoms, cutoff=3.5)
        k = int(np.argmin(d))
        midpoint = (pos[i_idx[k]] + pos[j_idx[k]]) / 2.0
        # displace along z (out of plane)
        n_pos = midpoint + np.array([0.0, 0.0, 1.50])

    elif site == "interstitial":
        # Place N in the van-der-Waals gap between layers (mid-z of supercell,
        # shifted by half the interlayer spacing from the puckered-layer centre)
        z_mean = pos[:, 2].mean()
        # interlayer gap centre ≈ z_mean + c/2  (mod cell c)
        c_len = np.linalg.norm(cell[2])
        z_gap = (z_mean + c_len / 2.0) % c_len
        n_pos = np.array([cx, cy, z_gap])

    elif site == "zigzag":
        # Along the zigzag chain: shift one of the top-layer P atoms by ~1.5 Å
        # in the b-axis (armchair ⊥ = y) direction
        top_idx = int(np.argmax(pos[:, 2]))
        n_pos = pos[top_idx] + np.array([0.0, 1.50, 0.0])

    elif site == "armchair":
        # Along the armchair ridge: shift a top-layer P atom by ~1.5 Å in a-axis
        top_idx = int(np.argmax(pos[:, 2]))
        n_pos = pos[top_idx] + np.array([1.50, 0.0, 0.0])

    else:
        raise ValueError(
            f"Unknown site '{site}'. Choose from: bridge, interstitial, zigzag, armchair"
        )

    return n_pos


def add_nitrogen(atoms: Atoms, site: str) -> Atoms:
    """
    Return a new Atoms object with one N added at the requested doping site.
    """
    from ase import Atom
    n_pos = get_nitrogen_site(atoms, site)
    doped = atoms.copy()
    doped.append(Atom("N", position=n_pos))
    doped.info["doping_site"] = site
    return doped
