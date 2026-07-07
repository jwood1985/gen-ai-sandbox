"""
Analyze coherence decay curves: fit T2, build the isotope × DD matrix.

Fits each coherence curve to a stretched exponential:
    L(t) = exp(-(t/T2)^n)

where T2 is the coherence time and n is the stretch exponent.
The stretch exponent encodes the shape of the spectral density —
it is physics, not a nuisance parameter.

Outputs:
  - Per-config T2 + stretch exponent (median ± IQR over seeds)
  - 2D coherence matrix heatmap (isotope × pulse sequence)
  - Purity-vs-T2 curves (the procurement-decision figure)
  - T2 distribution widths (array homogeneity argument)

Usage:
    python src/analyze.py --results-dir results/raw --output results/fits
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


RESULTS_DIR = Path(__file__).parent.parent / "results"


# --- Fitting ---

def stretched_exponential(t, T2, n):
    """L(t) = exp(-(t/T2)^n)"""
    return np.exp(-((t / T2) ** n))


def fit_stretched_exponential(
    time_us: np.ndarray,
    coherence: np.ndarray,
    t2_bounds: tuple = (0.001, 10000.0),
    n_bounds: tuple = (0.5, 6.0),
) -> tuple[float, float]:
    """
    Fit coherence decay to stretched exponential.

    Returns (T2_us, stretch_exponent).
    Raises RuntimeError if fit fails.
    """
    # Initial guess: T2 at 1/e crossing, n=2 (Gaussian-like)
    threshold = 1.0 / np.e
    below = np.where(coherence < threshold)[0]
    t2_guess = time_us[below[0]] if len(below) > 0 else time_us[-1] / 2

    try:
        popt, _ = curve_fit(
            stretched_exponential,
            time_us,
            coherence,
            p0=[t2_guess, 2.0],
            bounds=([t2_bounds[0], n_bounds[0]], [t2_bounds[1], n_bounds[1]]),
            maxfev=5000,
        )
        return popt[0], popt[1]
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(f"Fit failed: {e}")


# --- Aggregation ---

def load_all_results(results_dir: Path) -> list[dict]:
    """Load all .npz result files."""
    results = []
    for path in sorted(results_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        meta["time_us"] = data["time_us"]
        meta["coherence"] = data["coherence"]
        results.append(meta)
    return results


def aggregate_by_config(results: list[dict]) -> dict:
    """
    Group results by (isotope_config, magnetic_field, pulse_sequence, n_pulses)
    and compute ensemble statistics over spatial seeds.

    Returns nested dict with T2 median, IQR, stretch exponent stats.
    """
    groups = defaultdict(list)
    for r in results:
        key = (
            r["isotope_config"],
            r["magnetic_field_mT"],
            r["pulse_sequence"],
            r["n_pulses"],
        )
        groups[key].append(r)

    aggregated = {}
    for key, group in groups.items():
        t2_vals = []
        n_vals = []
        for r in group:
            try:
                t2, n = fit_stretched_exponential(r["time_us"], r["coherence"])
                t2_vals.append(t2)
                n_vals.append(n)
            except RuntimeError:
                pass

        if t2_vals:
            t2_arr = np.array(t2_vals)
            n_arr = np.array(n_vals)
            aggregated[key] = {
                "t2_median_us": float(np.median(t2_arr)),
                "t2_q25_us": float(np.percentile(t2_arr, 25)),
                "t2_q75_us": float(np.percentile(t2_arr, 75)),
                "t2_iqr_us": float(np.percentile(t2_arr, 75) - np.percentile(t2_arr, 25)),
                "t2_values_us": t2_arr.tolist(),
                "n_median": float(np.median(n_arr)),
                "n_seeds": len(t2_vals),
                "config": key[0],
                "field_mT": key[1],
                "sequence": f"{key[2]}-{key[3]}",
            }

    return aggregated


# --- Figures ---

def plot_coherence_matrix(aggregated: dict, output_dir: Path) -> None:
    """
    2D heatmap: isotope configuration × pulse sequence, colored by T2.
    The money figure for the proposal.
    """
    if not HAS_MPL:
        print("  matplotlib not available — skipping heatmap")
        return

    # Extract unique configs and sequences at the reference field
    ref_field = 50.0  # mT
    configs = sorted(set(k[0] for k in aggregated if k[1] == ref_field))
    sequences = sorted(set(f"{k[2]}-{k[3]}" for k in aggregated if k[1] == ref_field))

    if not configs or not sequences:
        print("  No data at reference field — skipping heatmap")
        return

    matrix = np.full((len(configs), len(sequences)), np.nan)
    for (cfg, field, seq, npulses), stats in aggregated.items():
        if field != ref_field:
            continue
        seq_label = f"{seq}-{npulses}"
        if cfg in configs and seq_label in sequences:
            i = configs.index(cfg)
            j = sequences.index(seq_label)
            matrix[i, j] = stats["t2_median_us"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sequences)))
    ax.set_xticklabels(sequences, rotation=45, ha="right")
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.set_xlabel("Pulse Sequence")
    ax.set_ylabel("Isotope Configuration")
    ax.set_title(f"T₂ Coherence Matrix — V_B⁻ in hBN (B = {ref_field} mT)")
    cbar = fig.colorbar(im)
    cbar.set_label("T₂ (μs)")

    # Annotate cells with values
    for i in range(len(configs)):
        for j in range(len(sequences)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < np.nanmedian(matrix) else "black",
                        fontsize=9)

    fig.tight_layout()
    path = output_dir / "coherence_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_purity_curves(aggregated: dict, output_dir: Path) -> None:
    """
    Purity-vs-T2 curves: the procurement-decision figure.
    Shows how T2 improves with isotopic enrichment for each target.
    """
    if not HAS_MPL:
        print("  matplotlib not available — skipping purity curves")
        return

    ref_field = 50.0
    ref_seq = ("hahn", 1)

    # Group by base config, extract purity level from label
    purity_data = defaultdict(list)
    for (cfg, field, seq, npulses), stats in aggregated.items():
        if field != ref_field or (seq, npulses) != ref_seq:
            continue
        # Parse purity from label like "h10B15N_p95"
        if "_p" in cfg:
            base = cfg.rsplit("_p", 1)[0]
            purity = int(cfg.rsplit("_p", 1)[1]) / 100.0
        else:
            base = cfg
            purity = 1.0 if cfg != "natural_abundance" else None

        if purity is not None:
            purity_data[base].append((purity, stats["t2_median_us"], stats["t2_iqr_us"]))

    if not purity_data:
        print("  No purity sweep data — skipping purity curves")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for base, points in sorted(purity_data.items()):
        points.sort(key=lambda x: x[0])
        purities = [p[0] for p in points]
        t2s = [p[1] for p in points]
        iqrs = [p[2] for p in points]
        ax.errorbar(purities, t2s, yerr=iqrs, marker="o", label=base, capsize=3)

    ax.set_xlabel("Isotopic Purity")
    ax.set_ylabel("T₂ (μs)")
    ax.set_title(f"Coherence vs Isotopic Purity — Hahn Echo, B = {ref_field} mT")
    ax.legend()
    ax.set_xscale("log" if any(p < 0.5 for points in purity_data.values() for p, _, _ in points) else "linear")

    fig.tight_layout()
    path = output_dir / "purity_vs_t2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_t2_distributions(aggregated: dict, output_dir: Path) -> None:
    """
    T2 distribution widths per config — the array homogeneity argument.
    """
    if not HAS_MPL:
        print("  matplotlib not available — skipping distribution plot")
        return

    ref_field = 50.0

    fig, ax = plt.subplots(figsize=(10, 5))
    hahn_data = {
        k: v for k, v in aggregated.items()
        if k[1] == ref_field and k[2] == "hahn" and "t2_values_us" in v
    }

    if not hahn_data:
        print("  No Hahn echo data — skipping distribution plot")
        return

    labels = []
    data = []
    for key, stats in sorted(hahn_data.items(), key=lambda x: x[1].get("t2_median_us", 0)):
        labels.append(stats["config"])
        data.append(stats["t2_values_us"])

    ax.boxplot(data, labels=labels, vert=True)
    ax.set_ylabel("T₂ (μs)")
    ax.set_title(f"T₂ Distributions — Hahn Echo, B = {ref_field} mT")
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    path = output_dir / "t2_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Analyze CCE results.")
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR / "raw"),
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_DIR / "fits"),
    )
    args = parser.parse_args()

    raw_dir = Path(args.results_dir)
    out_dir = Path(args.output)
    fig_dir = RESULTS_DIR / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== ANALYSIS ===")
    print(f"Loading results from: {raw_dir}")
    results = load_all_results(raw_dir)
    print(f"  Loaded {len(results)} result files.")

    if not results:
        print("  No results to analyze.")
        return

    print("\nAggregating by configuration...")
    aggregated = aggregate_by_config(results)
    print(f"  {len(aggregated)} unique configurations.")

    # Print summary table
    print("\n  {:30s} {:>10s} {:>10s} {:>8s} {:>6s}".format(
        "Config", "T2 (μs)", "IQR (μs)", "n_exp", "Seeds"
    ))
    print("  " + "-" * 70)
    for key, stats in sorted(aggregated.items(), key=lambda x: x[1]["t2_median_us"]):
        print(
            f"  {stats['config']:30s} "
            f"{stats['t2_median_us']:10.4f} "
            f"{stats['t2_iqr_us']:10.4f} "
            f"{stats['n_median']:8.2f} "
            f"{stats['n_seeds']:6d}"
        )

    # Save aggregated fits
    fits_path = out_dir / "aggregated_t2.json"
    serializable = {str(k): v for k, v in aggregated.items()}
    with open(fits_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved fits: {fits_path}")

    # Generate proposal figures
    print("\nGenerating figures...")
    plot_coherence_matrix(aggregated, fig_dir)
    plot_purity_curves(aggregated, fig_dir)
    plot_t2_distributions(aggregated, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
