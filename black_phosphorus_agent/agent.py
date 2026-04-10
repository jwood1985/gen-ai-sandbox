"""
Black Phosphorus DFT Agent
==========================
A Claude-powered agent that orchestrates DFT calculations on orthorhombic
black phosphorus with nitrogen impurities.

Workflow
--------
1. Build a black-phosphorus supercell with ≈nATOM P atoms.
2. Relax the pristine structure to the DFT ground state.
3. For each doping site (bridge, interstitial, zigzag, armchair):
   a. Insert one N atom at the specified site.
   b. Re-relax the doped structure.
   c. Determine whether P-N covalent bonds have formed.
4. Summarise all results.

Usage
-----
    python -m black_phosphorus_agent.agent          # nATOM=50, all sites
    python -m black_phosphorus_agent.agent --site bridge --no-gpaw
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import anthropic

# ── local imports ──────────────────────────────────────────────────────────────
from .config import N_ATOM, SUPERCELL, RESULTS_DIR
from .structure import build_supercell, add_nitrogen
from .dft_tools import relax_structure, analyze_bonding

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DOPING_SITES = ["bridge", "interstitial", "zigzag", "armchair"]

# ── Tool definitions (JSON Schema) ────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "build_and_relax_pristine",
        "description": (
            "Build the orthorhombic black phosphorus supercell with approximately "
            "nATOM phosphorus atoms and relax it to the DFT ground state. "
            "Returns the relaxed total energy and convergence status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_atom": {
                    "type": "integer",
                    "description": "Target number of P atoms (nearest valid supercell is used).",
                },
                "use_gpaw": {
                    "type": "boolean",
                    "description": "Use GPAW for DFT; falls back to EMT if False or GPAW absent.",
                    "default": True,
                },
            },
            "required": ["n_atom"],
        },
    },
    {
        "name": "dope_and_relax",
        "description": (
            "Add one nitrogen atom to a specified doping site in the relaxed black "
            "phosphorus supercell and re-relax the doped system. "
            "Available sites: bridge, interstitial, zigzag, armchair."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "enum": DOPING_SITES,
                    "description": "Doping site for the nitrogen atom.",
                },
                "use_gpaw": {
                    "type": "boolean",
                    "description": "Use GPAW for DFT; falls back to EMT if False or GPAW absent.",
                    "default": True,
                },
            },
            "required": ["site"],
        },
    },
    {
        "name": "check_covalent_bonding",
        "description": (
            "Analyse the relaxed doped structure to determine whether the nitrogen "
            "atom forms covalent bonds with neighbouring phosphorus atoms. "
            "Returns bond lengths, a covalency flag, and the threshold distance used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "enum": DOPING_SITES,
                    "description": "Doping site to analyse (must have been relaxed first).",
                },
            },
            "required": ["site"],
        },
    },
    {
        "name": "summarise_results",
        "description": (
            "Collect all relaxation and bonding results across all doping sites and "
            "return a structured summary table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────

class DFTToolExecutor:
    """Stateful tool executor that keeps relaxed Atoms objects in memory."""

    def __init__(self, use_gpaw: bool = True, results_dir: str = RESULTS_DIR):
        self.use_gpaw    = use_gpaw
        self.results_dir = results_dir
        self.pristine_atoms   = None   # relaxed pristine supercell
        self.doped_atoms: dict = {}    # site → relaxed doped Atoms
        self.relax_results: dict = {}  # site → relax summary
        self.bond_results: dict  = {}  # site → bonding analysis

    def build_and_relax_pristine(self, n_atom: int, use_gpaw: bool = True) -> dict:
        nx, ny, nz = SUPERCELL
        atoms = build_supercell(nx, ny, nz)
        actual_n = len(atoms)
        log.info("Built %d-atom black phosphorus supercell (%d×%d×%d).", actual_n, nx, ny, nz)

        result = relax_structure(
            atoms,
            label="pristine",
            use_gpaw=use_gpaw and self.use_gpaw,
            results_dir=self.results_dir,
        )
        self.pristine_atoms = result["atoms"]
        self.relax_results["pristine"] = {k: v for k, v in result.items() if k != "atoms"}
        log.info("Pristine relaxation complete. E=%.4f eV, converged=%s", result["energy_eV"], result["converged"])
        return self.relax_results["pristine"]

    def dope_and_relax(self, site: str, use_gpaw: bool = True) -> dict:
        if self.pristine_atoms is None:
            raise RuntimeError("Run build_and_relax_pristine() first.")

        doped = add_nitrogen(self.pristine_atoms, site)
        log.info("Added N at '%s' site → %d atoms total.", site, len(doped))

        result = relax_structure(
            doped,
            label=f"doped_{site}",
            use_gpaw=use_gpaw and self.use_gpaw,
            results_dir=self.results_dir,
        )
        self.doped_atoms[site] = result["atoms"]
        self.relax_results[site] = {k: v for k, v in result.items() if k != "atoms"}
        log.info("Site '%s' relaxation complete. E=%.4f eV, converged=%s", site, result["energy_eV"], result["converged"])
        return self.relax_results[site]

    def check_covalent_bonding(self, site: str) -> dict:
        if site not in self.doped_atoms:
            raise RuntimeError(f"Site '{site}' not yet relaxed. Run dope_and_relax first.")

        result = analyze_bonding(
            self.doped_atoms[site],
            site=site,
            results_dir=self.results_dir,
        )
        self.bond_results[site] = result
        log.info(
            "Site '%s': covalent=%s, nearest P-N=%.3f Å (threshold=%.3f Å)",
            site,
            result["is_covalent"],
            result["pn_distances"][0]["distance_A"] if result["pn_distances"] else float("nan"),
            result["bond_threshold"],
        )
        return result

    def summarise_results(self) -> dict:
        rows = []
        for site in DOPING_SITES:
            relax = self.relax_results.get(site, {})
            bond  = self.bond_results.get(site, {})
            nearest = (
                round(bond["pn_distances"][0]["distance_A"], 3)
                if bond.get("pn_distances")
                else None
            )
            rows.append({
                "site":                site,
                "energy_eV":           relax.get("energy_eV"),
                "converged":           relax.get("converged"),
                "max_force_eV_per_A":  relax.get("max_force_eV_per_A"),
                "nearest_PN_dist_A":   nearest,
                "covalent_bonds":      len(bond.get("covalent_bonds", [])),
                "is_covalent":         bond.get("is_covalent"),
            })
        summary = {
            "pristine": self.relax_results.get("pristine", {}),
            "doping_sites": rows,
        }
        path = Path(self.results_dir) / "summary.json"
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info("Summary written to %s", path)
        return summary

    def dispatch(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "build_and_relax_pristine":
            return self.build_and_relax_pristine(**tool_input)
        elif tool_name == "dope_and_relax":
            return self.dope_and_relax(**tool_input)
        elif tool_name == "check_covalent_bonding":
            return self.check_covalent_bonding(**tool_input)
        elif tool_name == "summarise_results":
            return self.summarise_results()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# ── Agent loop ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a computational materials science agent specialising in density functional
theory (DFT) simulations of 2D materials.

Your task:
1. Call build_and_relax_pristine to build and relax the black phosphorus supercell.
2. For EACH of the four nitrogen doping sites (bridge, interstitial, zigzag, armchair):
   a. Call dope_and_relax for that site.
   b. Call check_covalent_bonding for that site.
3. Call summarise_results to collate all findings.
4. Write a concise scientific report covering:
   - Supercell size and DFT settings.
   - Pristine system total energy and convergence.
   - For each N site: total energy, convergence, nearest P-N distance, and whether
     a covalent bond is present (with physical reasoning).
   - An overall conclusion comparing all four sites.

Work systematically — complete every step before writing the report.
"""


def run_agent(
    n_atom: int = N_ATOM,
    sites: list[str] | None = None,
    use_gpaw: bool = True,
    results_dir: str = RESULTS_DIR,
    verbose: bool = False,
) -> str:
    """
    Run the DFT agent and return the final scientific report as a string.

    Parameters
    ----------
    n_atom      : target number of P atoms in the supercell
    sites       : list of doping sites to test (default: all four)
    use_gpaw    : prefer GPAW over EMT fallback
    results_dir : directory to write output files
    verbose     : print streaming tokens to stdout
    """
    sites = sites or DOPING_SITES

    client   = anthropic.Anthropic()
    executor = DFTToolExecutor(use_gpaw=use_gpaw, results_dir=results_dir)

    user_msg = (
        f"Please run the full DFT workflow for orthorhombic black phosphorus "
        f"with nATOM={n_atom}. Test nitrogen doping at these sites: "
        f"{', '.join(sites)}. Use GPAW={'yes' if use_gpaw else 'no (EMT fallback)'}."
    )

    messages = [{"role": "user", "content": user_msg}]
    final_text = ""

    log.info("Starting DFT agent. Sites: %s", sites)

    while True:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            if verbose:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
            response = stream.get_final_message()

        # Append assistant turn (full content, preserving tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
            break

        if response.stop_reason != "tool_use":
            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        # Execute all tool calls and feed results back
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            log.info("→ Tool call: %s(%s)", block.name, json.dumps(block.input)[:120])
            try:
                result = executor.dispatch(block.name, block.input)
                content = json.dumps(result, indent=2, default=str)
                is_error = False
            except Exception as exc:
                content = f"ERROR: {exc}"
                is_error = True
                log.error("Tool %s failed: %s", block.name, exc)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     content,
                "is_error":    is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    return final_text


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black Phosphorus DFT Agent")
    p.add_argument("--n-atom",      type=int,   default=N_ATOM,
                   help="Target P-atom count (default: %(default)s)")
    p.add_argument("--site",        choices=DOPING_SITES, action="append",
                   dest="sites",    metavar="SITE",
                   help="Doping site(s) to test (repeat for multiple). Default: all four.")
    p.add_argument("--no-gpaw",     action="store_true",
                   help="Skip GPAW and use EMT (fast, not real DFT)")
    p.add_argument("--results-dir", default=RESULTS_DIR,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--verbose",     action="store_true",
                   help="Stream agent tokens to stdout")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_agent(
        n_atom=args.n_atom,
        sites=args.sites,
        use_gpaw=not args.no_gpaw,
        results_dir=args.results_dir,
        verbose=args.verbose,
    )
    print("\n" + "=" * 72)
    print("AGENT REPORT")
    print("=" * 72)
    print(report)
