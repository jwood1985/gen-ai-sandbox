"""
Convergence studies for CCE simulations.

Sweeps CCE order and bath radius to find converged parameters before
committing to the full production sweep.

Convergence criterion: T2 changes < 5% between successive bath radii.

Usage:
    python src/converge.py --config config/sweeps.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from build_bath import build_bath, set_first_shell_hyperfine, load_physics_config
from run_cce import setup_simulator, make_pulse_sequence, us_to_ms
from analyze import fit_stretched_exponential


CONFIG_DIR = Path(__file__).parent.parent / "config"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "fits"


def convergence_sweep(
    cce_orders: list[int],
    bath_radii: list[float],
    magnetic_field_mT: float,
    B10_fraction: float,
    N14_fraction: float,
    time_range_us: list[float],
    n_time_points: int,
    n_seeds: int = 10,
    convergence_threshold: float = 0.05,
) -> dict:
    """
    Sweep CCE order and bath radius to find converged parameters.

    Returns
    -------
    results : dict
        Nested dict: results[cce_order][bath_radius] = {
            't2_median_us': ..., 't2_iqr_us': ..., 'seeds': ...,
        }
    """
    physics = load_physics_config()
    time_us = np.linspace(time_range_us[0], time_range_us[1], n_time_points)
    time_ms = us_to_ms(time_us)

    results = {}

    for order in cce_orders:
        results[order] = {}
        print(f"\n=== CCE order {order} ===")

        for radius in bath_radii:
            t2_values = []

            for seed in range(n_seeds):
                bath, vac_pos = build_bath(
                    B10_fraction=B10_fraction,
                    N14_fraction=N14_fraction,
                    supercell_size=max(200.0, radius * 5),
                    seed=seed,
                )
                set_first_shell_hyperfine(bath, vac_pos, physics)

                pulses = make_pulse_sequence("hahn", 1)
                calc = setup_simulator(
                    physics=physics,
                    bath=bath,
                    magnetic_field_mT=magnetic_field_mT,
                    cce_order=order,
                    r_bath=radius,
                    pulses=pulses,
                )

                coherence = calc.compute(
                    time_ms,
                    method="gcce",
                    nbstates=30,
                    quantity="coherence",
                    interlaced=True,
                )

                coh_abs = np.abs(coherence)
                try:
                    t2, n_stretch = fit_stretched_exponential(time_us, coh_abs)
                    t2_values.append(t2)
                except RuntimeError:
                    pass

            if t2_values:
                t2_arr = np.array(t2_values)
                median = np.median(t2_arr)
                q25, q75 = np.percentile(t2_arr, [25, 75])
                results[order][radius] = {
                    "t2_median_us": float(median),
                    "t2_iqr_us": float(q75 - q25),
                    "n_seeds": len(t2_values),
                }
                print(
                    f"  r_bath={radius:5.1f} Å: "
                    f"T2 = {median:.4f} ± {q75-q25:.4f} μs "
                    f"(n={len(t2_values)})"
                )
            else:
                results[order][radius] = {
                    "t2_median_us": None,
                    "t2_iqr_us": None,
                    "n_seeds": 0,
                }
                print(f"  r_bath={radius:5.1f} Å: ALL FITS FAILED")

    return results


def check_convergence(
    results: dict,
    threshold: float = 0.05,
) -> dict[int, float | None]:
    """
    Find the smallest converged bath radius for each CCE order.

    Converged = T2 changes < threshold between successive radii.

    Returns dict: cce_order → converged_radius (or None).
    """
    converged = {}

    for order, radii_data in results.items():
        sorted_radii = sorted(radii_data.keys())
        converged[order] = None

        for i in range(1, len(sorted_radii)):
            r_prev = sorted_radii[i - 1]
            r_curr = sorted_radii[i]
            t2_prev = radii_data[r_prev]["t2_median_us"]
            t2_curr = radii_data[r_curr]["t2_median_us"]

            if t2_prev is None or t2_curr is None:
                continue

            rel_change = abs(t2_curr - t2_prev) / t2_prev
            if rel_change < threshold:
                converged[order] = r_prev
                break

    return converged


def main():
    parser = argparse.ArgumentParser(description="Run convergence study.")
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_DIR / "sweeps.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        sweeps = yaml.safe_load(f)

    conv = sweeps["convergence"]

    # Load isotope fractions for the specified config
    with open(CONFIG_DIR / "isotopes.yaml") as f:
        isotopes = yaml.safe_load(f)
    iso = isotopes["configurations"][conv["isotope_config"]]

    print("=== CONVERGENCE STUDY ===")
    print(f"Config: {conv['isotope_config']}")
    print(f"B field: {conv['magnetic_field_mT']} mT")
    print(f"CCE orders: {conv['cce_orders']}")
    print(f"Bath radii: {conv['bath_radii_A']} Å")
    print(f"Seeds: {conv['seeds']}")

    results = convergence_sweep(
        cce_orders=conv["cce_orders"],
        bath_radii=conv["bath_radii_A"],
        magnetic_field_mT=conv["magnetic_field_mT"],
        B10_fraction=iso["B10_fraction"],
        N14_fraction=iso["N14_fraction"],
        time_range_us=conv["time_range_us"],
        n_time_points=conv["n_time_points"],
        n_seeds=conv["seeds"],
        convergence_threshold=conv["convergence_threshold"],
    )

    converged = check_convergence(results, conv["convergence_threshold"])

    print("\n=== CONVERGENCE RESULTS ===")
    for order, radius in converged.items():
        if radius is not None:
            t2 = results[order][radius]["t2_median_us"]
            print(f"  CCE-{order}: converged at r_bath = {radius} Å (T2 = {t2:.4f} μs)")
        else:
            print(f"  CCE-{order}: NOT CONVERGED — increase bath radius range")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "convergence.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "converged": converged}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
