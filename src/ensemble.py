"""
Generate and enqueue the full production sweep.

Creates jobs for all combinations of:
  isotope config × purity level × magnetic field × pulse sequence × spatial seed

Usage:
    # Enqueue all production jobs:
    python src/ensemble.py --config config/sweeps.yaml

    # Enqueue with custom seed count:
    python src/ensemble.py --config config/sweeps.yaml --seeds 100

    # Dry run (count jobs without creating):
    python src/ensemble.py --config config/sweeps.yaml --dry-run
"""

import argparse
from pathlib import Path

import yaml

from job_queue import create_job, queue_summary


CONFIG_DIR = Path(__file__).parent.parent / "config"


def generate_production_jobs(
    sweeps_config: dict,
    isotopes_config: dict,
    n_seeds: int = 50,
    dry_run: bool = False,
) -> int:
    """
    Generate all production sweep jobs.

    Returns the total number of jobs created.
    """
    prod = sweeps_config["production"]
    iso_configs = isotopes_config["configurations"]
    purity_levels = prod["purity_levels"]
    fields = prod["magnetic_fields_mT"]
    sequences = prod["pulse_sequences"]
    cce_order = prod["cce_order"]
    bath_radius = prod["bath_radius_A"]

    job_count = 0

    for config_name in prod["isotope_configs"]:
        iso = iso_configs[config_name]

        # Pure-corner configuration (no purity sweep)
        for field in fields:
            for seq in sequences:
                for seed in range(n_seeds):
                    job_count += 1
                    if not dry_run:
                        create_job(
                            isotope_config=config_name,
                            B10_fraction=iso["B10_fraction"],
                            B11_fraction=iso["B11_fraction"],
                            N14_fraction=iso["N14_fraction"],
                            N15_fraction=iso["N15_fraction"],
                            magnetic_field_mT=field,
                            pulse_sequence=seq["name"],
                            n_pulses=seq["n_pulses"],
                            cce_order=cce_order,
                            bath_radius_A=bath_radius,
                            n_time_points=prod["n_time_points"],
                            time_range_us=prod["time_range_us"],
                            spatial_seed=seed,
                        )

        # Purity sweep: interpolate between natural abundance and target
        if config_name == "natural_abundance":
            continue

        for purity in purity_levels:
            # Determine fractions at this purity level
            # e.g., for h10B15N at 95% purity:
            #   B10 = 0.95, B11 = 0.05, N14 = 0.05, N15 = 0.95
            if iso["B10_fraction"] > 0.5:
                b10 = purity
            elif iso["B10_fraction"] < 0.5:
                b10 = 1.0 - purity
            else:
                b10 = iso["B10_fraction"]

            if iso["N15_fraction"] > 0.5:
                n14 = 1.0 - purity
            elif iso["N15_fraction"] < 0.5:
                n14 = purity
            else:
                n14 = iso["N14_fraction"]

            purity_label = f"{config_name}_p{int(purity*100)}"

            for field in fields:
                for seq in sequences:
                    for seed in range(n_seeds):
                        job_count += 1
                        if not dry_run:
                            create_job(
                                isotope_config=purity_label,
                                B10_fraction=b10,
                                B11_fraction=1.0 - b10,
                                N14_fraction=n14,
                                N15_fraction=1.0 - n14,
                                magnetic_field_mT=field,
                                pulse_sequence=seq["name"],
                                n_pulses=seq["n_pulses"],
                                cce_order=cce_order,
                                bath_radius_A=bath_radius,
                                n_time_points=prod["n_time_points"],
                                time_range_us=prod["time_range_us"],
                                spatial_seed=seed,
                            )

    return job_count


def main():
    parser = argparse.ArgumentParser(description="Generate production sweep jobs.")
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_DIR / "sweeps.yaml"),
    )
    parser.add_argument("--seeds", type=int, default=50, help="Seeds per config.")
    parser.add_argument("--dry-run", action="store_true", help="Count without creating.")
    args = parser.parse_args()

    with open(args.config) as f:
        sweeps = yaml.safe_load(f)
    with open(CONFIG_DIR / "isotopes.yaml") as f:
        isotopes = yaml.safe_load(f)

    # Override seed count
    sweeps["production"]["seeds"] = args.seeds

    print("=== PRODUCTION SWEEP ===")
    n_configs = len(sweeps["production"]["isotope_configs"])
    n_purities = len(sweeps["production"]["purity_levels"])
    n_fields = len(sweeps["production"]["magnetic_fields_mT"])
    n_seqs = len(sweeps["production"]["pulse_sequences"])
    print(f"  Isotope configs:  {n_configs}")
    print(f"  Purity levels:    {n_purities}")
    print(f"  Magnetic fields:  {n_fields}")
    print(f"  Pulse sequences:  {n_seqs}")
    print(f"  Seeds per combo:  {args.seeds}")

    if args.dry_run:
        print("\n  [DRY RUN — no jobs created]")

    total = generate_production_jobs(
        sweeps, isotopes, n_seeds=args.seeds, dry_run=args.dry_run,
    )

    print(f"\n  Total jobs: {total}")

    if not args.dry_run:
        print(f"\n  Queue status: {queue_summary()}")
    else:
        print("\n  Rerun without --dry-run to create jobs.")


if __name__ == "__main__":
    main()
