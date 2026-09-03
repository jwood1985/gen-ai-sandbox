"""
PyCCE driver: run a single CCE coherence calculation for V_B⁻ in hBN.

One job = (isotope config, B field, pulse sequence, spatial seed).

PyCCE units: couplings in kHz, time in ms, magnetic field in Gauss.
User-facing units: T2 in μs, fields in mT, couplings in MHz.

Uses generalized CCE (gCCE) which is essential for hBN due to the
strong, non-secular hyperfine interactions in the dense spin bath.

Usage:
    python src/run_cce.py --smoke-test
    python src/run_cce.py --job <job_id>
    python src/run_cce.py --worker
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import pycce as pc
except ImportError:
    raise ImportError("PyCCE is required: pip install pycce")

from build_bath import build_bath, set_first_shell_hyperfine, load_physics_config
from job_queue import Job, load_job, complete_job, fail_job, claim_next_job


RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
CONFIG_DIR = Path(__file__).parent.parent / "config"


# --- Unit conversions ---
# User config uses mT and μs; PyCCE uses Gauss and ms
def mT_to_gauss(mT: float) -> float:
    return mT * 10.0

def us_to_ms(us: float) -> float:
    return us / 1000.0

def MHz_to_kHz(mhz: float) -> float:
    return mhz * 1000.0


# --- Pulse sequences ---

def make_pulse_sequence(name: str, n_pulses: int):
    """
    Build PyCCE-compatible pulse sequence.

    PyCCE accepts:
      - Integer N for CPMG-N (Hahn echo = CPMG-1)
      - List of tuples ('axis', angle) for custom sequences

    XY8 unit: {X, Y, X, Y, Y, X, Y, X} — each a π-pulse.
    XY8-N repeats the unit N times.
    """
    if name == "hahn":
        return 1

    if name == "cpmg":
        return n_pulses

    if name == "xy8":
        unit = [
            ("x", np.pi), ("y", np.pi), ("x", np.pi), ("y", np.pi),
            ("y", np.pi), ("x", np.pi), ("y", np.pi), ("x", np.pi),
        ]
        return unit * n_pulses

    raise ValueError(f"Unknown pulse sequence: {name}")


def setup_simulator(
    physics: dict,
    bath,
    magnetic_field_mT: float,
    cce_order: int,
    r_bath: float,
    r_dipole: float = 8.0,
    pulses=1,
) -> pc.Simulator:
    """
    Configure PyCCE Simulator for V_B⁻.

    Parameters
    ----------
    physics : dict
        Contents of config/physics.yaml.
    bath : BathArray
        From build_bath(), with first-shell hyperfine already set.
    magnetic_field_mT : float
        External field along c-axis in mT (converted to Gauss internally).
    cce_order : int
        CCE expansion order (2 or 3).
    r_bath : float
        Bath cutoff radius in Å.
    r_dipole : float
        Max bath-bath dipolar coupling radius in Å.
    pulses : int or list
        Pulse sequence specification.

    Returns
    -------
    calc : pc.Simulator
    """
    cs = physics["central_spin"]

    # ZFS in kHz (PyCCE units)
    D_kHz = MHz_to_kHz(cs["D"])  # 3470 MHz → 3470000 kHz
    E_kHz = MHz_to_kHz(cs["E"])  # 0

    # Magnetic field along c-axis (z), in Gauss
    B_gauss = mT_to_gauss(magnetic_field_mT)

    # V_B⁻: S=1, qubit subspace ms=0 ↔ ms=-1
    # Eigenstate ordering depends on field; alpha=0, beta=1 selects
    # the two lowest-energy eigenstates of the central spin Hamiltonian
    calc = pc.Simulator(
        spin=cs["S"],
        bath=bath,
        r_bath=r_bath,
        r_dipole=r_dipole,
        order=cce_order,
        D=D_kHz,
        E=E_kHz,
        position=np.zeros(3),
        magnetic_field=np.array([0, 0, B_gauss]),
        pulses=pulses,
        alpha=0,
        beta=1,
    )

    return calc


def run_single_job(job: Job) -> dict:
    """
    Execute a single CCE job.

    Returns dict with time_us, coherence, and metadata.
    """
    physics = load_physics_config()
    t_start = time.time()

    # Build bath with stochastic isotope placement
    bath, vac_pos = build_bath(
        B10_fraction=job.B10_fraction,
        N14_fraction=job.N14_fraction,
        supercell_size=200.0,
        seed=job.spatial_seed,
    )

    # Override first-shell N hyperfine with literature values
    set_first_shell_hyperfine(bath, vac_pos, physics)

    # Pulse sequence
    pulses = make_pulse_sequence(job.pulse_sequence, job.n_pulses)

    # Set up simulator
    calc = setup_simulator(
        physics=physics,
        bath=bath,
        magnetic_field_mT=job.magnetic_field_mT,
        cce_order=job.cce_order,
        r_bath=job.bath_radius_A,
        pulses=pulses,
    )

    # Time array: config is in μs, PyCCE wants ms
    time_us = np.linspace(
        job.time_range_us[0], job.time_range_us[1], job.n_time_points
    )
    time_ms = us_to_ms(time_us)

    # Run gCCE (generalized CCE — essential for hBN's dense, strongly-coupled bath)
    coherence = calc.compute(
        time_ms,
        method="gcce",
        nbstates=30,
        quantity="coherence",
        interlaced=True,
    )

    runtime_s = time.time() - t_start

    return {
        "time_us": time_us.tolist(),
        "coherence": np.abs(coherence).tolist(),
        "job_id": job.job_id,
        "isotope_config": job.isotope_config,
        "B10_fraction": job.B10_fraction,
        "N14_fraction": job.N14_fraction,
        "magnetic_field_mT": job.magnetic_field_mT,
        "pulse_sequence": job.pulse_sequence,
        "n_pulses": job.n_pulses,
        "cce_order": job.cce_order,
        "bath_radius_A": job.bath_radius_A,
        "spatial_seed": job.spatial_seed,
        "n_bath_spins": len(bath),
        "runtime_s": runtime_s,
    }


def save_result(result: dict) -> str:
    """Save result to results/raw/ as .npz."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result['job_id']}.npz"
    np.savez(
        path,
        time_us=np.array(result["time_us"]),
        coherence=np.array(result["coherence"]),
        metadata=json.dumps({
            k: v for k, v in result.items()
            if k not in ("time_us", "coherence")
        }),
    )
    return str(path)


def sanity_check(result: dict) -> str | None:
    """
    Check for anomalies. Returns error message or None if OK.

    Sanity window: T2 between 0.01 and 1000 μs.
    (Natural-abundance Hahn T2 is ~50-100 ns for V_B⁻)
    """
    coherence = np.array(result["coherence"])
    time_us = np.array(result["time_us"])

    if np.any(np.isnan(coherence)) or np.any(np.isinf(coherence)):
        return "NaN or Inf in coherence values"

    if len(coherence) > 0 and coherence[0] < 0.8:
        return f"Coherence at t=0 is {coherence[0]:.3f}, expected ~1.0"

    # Rough T2: time where |L(t)| drops to 1/e
    threshold = 1.0 / np.e
    below = np.where(coherence < threshold)[0]
    if len(below) > 0:
        t2_rough = time_us[below[0]]
        if t2_rough < 0.01 or t2_rough > 1000:
            return f"Rough T2 = {t2_rough:.4f} μs outside sanity window [0.01, 1000]"

    return None


def process_job(job: Job) -> None:
    """Run a job, check, save, update queue."""
    try:
        result = run_single_job(job)

        anomaly = sanity_check(result)
        if anomaly:
            fail_job(job, f"SANITY: {anomaly}")
            print(f"  ANOMALY [{job.job_id}]: {anomaly}")
            return

        result_path = save_result(result)
        complete_job(job, result_path)
        print(
            f"  OK [{job.job_id}] {result['isotope_config']} "
            f"B={result['magnetic_field_mT']}mT "
            f"{result['pulse_sequence']}-{result['n_pulses']} "
            f"seed={result['spatial_seed']} "
            f"({result['runtime_s']:.1f}s, {result['n_bath_spins']} bath spins)"
        )
    except Exception as e:
        fail_job(job, str(e))
        print(f"  FAIL [{job.job_id}]: {e}")
        raise


def run_worker(max_jobs: int | None = None) -> None:
    """Drain the job queue."""
    count = 0
    while True:
        if max_jobs is not None and count >= max_jobs:
            break
        job = claim_next_job()
        if job is None:
            print("Queue empty.")
            break
        print(f"Running job {job.job_id} ({count + 1}/{max_jobs or '∞'})...")
        process_job(job)
        count += 1
    print(f"Processed {count} jobs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CCE simulation jobs.")
    parser.add_argument("--job", type=str, help="Run a specific job by ID.")
    parser.add_argument("--worker", action="store_true", help="Run as queue worker.")
    parser.add_argument("--max-jobs", type=int, help="Max jobs for worker mode.")
    parser.add_argument("--smoke-test", action="store_true", help="Minimal smoke test.")
    args = parser.parse_args()

    if args.smoke_test:
        from job_queue import create_job
        print("=== SMOKE TEST ===")
        with open(CONFIG_DIR / "sweeps.yaml") as f:
            sweeps = yaml.safe_load(f)
        st = sweeps["smoke_test"]
        job = create_job(
            isotope_config=st["isotope_config"],
            B10_fraction=0.199, B11_fraction=0.801,
            N14_fraction=0.996, N15_fraction=0.004,
            magnetic_field_mT=st["magnetic_field_mT"],
            pulse_sequence=st["pulse_sequence"],
            n_pulses=1,
            cce_order=st["cce_order"],
            bath_radius_A=st["bath_radius_A"],
            n_time_points=st["n_time_points"],
            time_range_us=st["time_range_us"],
            spatial_seed=42,
        )
        job.status = "running"
        process_job(job)

    elif args.job:
        job = load_job(args.job)
        process_job(job)

    elif args.worker:
        run_worker(max_jobs=args.max_jobs)

    else:
        parser.print_help()
