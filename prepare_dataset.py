"""
Dataset preparation for nanomaterials research fine-tuning.

Supports three data sources:
  1. A local directory of .txt or .pdf-extracted text files (e.g. paper abstracts)
  2. A single .jsonl file with a "text" field per line
  3. A Hugging Face dataset name (e.g. a curated nanomaterials corpus)

Outputs a tokenized HuggingFace Dataset ready for the trainer.

Usage:
    python prepare_dataset.py --source ./data/abstracts --output ./tokenized_data
    python prepare_dataset.py --source papers.jsonl --output ./tokenized_data
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


SAMPLE_NANOMATERIALS_TEXTS = [
    "Graphene oxide nanosheets were synthesized via a modified Hummers method and "
    "subsequently reduced using hydrazine hydrate. The resulting reduced graphene oxide "
    "exhibited a high specific surface area of 620 m2/g and excellent electrical "
    "conductivity of 12,000 S/m, making it suitable for supercapacitor electrodes.",

    "Silver nanoparticles with a mean diameter of 15 nm were prepared by chemical "
    "reduction of silver nitrate using sodium citrate as both reducing and capping agent. "
    "UV-Vis spectroscopy confirmed a surface plasmon resonance peak at 420 nm. "
    "The nanoparticles demonstrated potent antibacterial activity against E. coli and "
    "S. aureus with minimum inhibitory concentrations of 5 and 10 micrograms per mL.",

    "Titanium dioxide nanotubes were fabricated by anodization of titanium foil in an "
    "ethylene glycol electrolyte containing 0.3 wt% ammonium fluoride. The nanotube "
    "arrays had an average inner diameter of 80 nm and length of 15 micrometers. "
    "After annealing at 450 degrees C, the anatase phase showed enhanced photocatalytic "
    "degradation of methylene blue under UV irradiation with a rate constant of 0.042 per minute.",

    "Carbon nanotubes functionalized with carboxyl groups were dispersed in a polyvinyl "
    "alcohol matrix to form nanocomposite films. At 2 wt% CNT loading, the tensile "
    "strength increased by 78% and Young's modulus by 120% compared to neat PVA. "
    "The percolation threshold for electrical conductivity was observed at 0.5 wt%.",

    "Zinc oxide quantum dots with tunable photoluminescence were synthesized via a "
    "sol-gel method at room temperature. By controlling the precursor ratio and reaction "
    "time, emission wavelengths spanning 340 to 520 nm were achieved. The quantum yield "
    "reached 65% for the 3.2 nm diameter particles, suitable for bioimaging applications.",

    "Iron oxide nanoparticles coated with polyethylene glycol were evaluated as contrast "
    "agents for magnetic resonance imaging. The superparamagnetic Fe3O4 core had a "
    "diameter of 10 nm with a 5 nm PEG shell. The transverse relaxivity r2 was measured "
    "at 185 mM-1 s-1 at 3T, demonstrating strong negative contrast enhancement in "
    "T2-weighted images of liver tissue.",

    "Mesoporous silica nanoparticles with an MCM-41 structure were loaded with "
    "doxorubicin for pH-responsive drug delivery. The pore size of 2.8 nm and surface "
    "area of 1050 m2/g enabled a drug loading capacity of 32 wt%. Release studies showed "
    "less than 5% premature release at pH 7.4 but 85% release within 12 hours at pH 5.0, "
    "mimicking the acidic tumor microenvironment.",

    "Molybdenum disulfide nanosheets were exfoliated from bulk crystals using "
    "lithium intercalation followed by ultrasonication. The monolayer MoS2 flakes had "
    "lateral dimensions of 200 to 500 nm. Field-effect transistors fabricated from "
    "individual flakes exhibited an on/off current ratio exceeding 10^8 and electron "
    "mobility of 45 cm2/Vs at room temperature.",

    "Gold nanorods with an aspect ratio of 4.2 were synthesized via seed-mediated growth "
    "in a cetyltrimethylammonium bromide solution. The longitudinal surface plasmon "
    "resonance was tuned to 808 nm for photothermal therapy applications. Under NIR "
    "laser irradiation at 2 W/cm2, the nanorod solution temperature increased by 32 "
    "degrees C within 5 minutes, sufficient for hyperthermia-induced cancer cell death.",

    "Hydroxyapatite nanoparticles doped with 5 mol% strontium were prepared by a "
    "wet chemical precipitation method for bone tissue engineering. The rod-shaped "
    "particles had dimensions of 60 nm in length and 15 nm in width. In vitro studies "
    "with MC3T3-E1 osteoblast cells showed 40% higher alkaline phosphatase activity "
    "compared to undoped hydroxyapatite after 14 days of culture.",

    "Copper sulfide nanocrystals with a covellite phase were synthesized through hot "
    "injection of sulfur precursor into a copper oleate solution at 220 degrees C. "
    "The hexagonal nanoplatelets had a uniform thickness of 4 nm and lateral size of "
    "25 nm. Near-infrared absorption at 1050 nm and a photothermal conversion efficiency "
    "of 38% were measured, indicating potential for combined photoacoustic imaging and "
    "photothermal therapy.",

    "Cellulose nanocrystals extracted from cotton linters via sulfuric acid hydrolysis "
    "were used to reinforce polylactic acid composites. The rod-like nanocrystals had "
    "an average length of 180 nm and diameter of 12 nm with a crystallinity index of "
    "88%. At 5 wt% loading, the storage modulus of PLA increased by 95% while "
    "maintaining optical transparency above 85% in the visible range.",

    "Perovskite CsPbBr3 nanocrystals passivated with oleylamine and oleic acid ligands "
    "were deposited as quantum dot films for light-emitting diodes. The 8 nm cubic "
    "nanocrystals exhibited a narrow photoluminescence linewidth of 18 nm centered at "
    "515 nm with a quantum yield of 92%. The resulting LED achieved a peak external "
    "quantum efficiency of 6.3% and luminance of 15,000 cd/m2.",

    "Boron nitride nanosheets were produced by liquid-phase exfoliation in isopropanol "
    "and incorporated into epoxy resin as thermal interface materials. At 30 wt% BN "
    "loading, the through-plane thermal conductivity reached 5.4 W/mK, a 27-fold "
    "enhancement over neat epoxy. The composite maintained electrical resistivity above "
    "10^14 ohm-cm, confirming electrical insulation despite high thermal conductivity.",

    "Cadmium selenide quantum dots with a zinc sulfide shell were conjugated to "
    "anti-HER2 antibodies for targeted fluorescence imaging of breast cancer cells. "
    "The core-shell QDs had a diameter of 6 nm and emission at 625 nm with a quantum "
    "yield of 48%. Confocal microscopy confirmed selective binding to HER2-positive "
    "SK-BR-3 cells with a signal-to-noise ratio 12 times higher than non-targeted QDs.",
]


def load_texts_from_directory(directory: str) -> list[str]:
    """Load all .txt files from a directory, one text block per file."""
    texts = []
    for path in sorted(Path(directory).glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            texts.append(content)
    return texts


def load_texts_from_jsonl(filepath: str) -> list[str]:
    """Load texts from a .jsonl file where each line has a 'text' field."""
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    texts.append(text)
    return texts


def tokenize_and_chunk(texts: list[str], tokenizer, max_length: int = 512):
    """
    Tokenize texts and split into fixed-length chunks for causal LM training.

    Each chunk becomes one training example. The labels are identical to the
    input_ids (the trainer shifts them internally for next-token prediction).
    """
    all_token_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_token_ids.extend(ids)
        all_token_ids.append(tokenizer.eos_token_id)

    # Split into non-overlapping chunks of max_length
    chunks = []
    for i in range(0, len(all_token_ids) - max_length, max_length):
        chunk = all_token_ids[i : i + max_length]
        chunks.append({"input_ids": chunk, "labels": chunk})

    # Include the final partial chunk if it's at least half a block
    remainder = all_token_ids[len(chunks) * max_length :]
    if len(remainder) >= max_length // 2:
        padded = remainder + [tokenizer.eos_token_id] * (max_length - len(remainder))
        chunks.append({"input_ids": padded, "labels": padded})

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a nanomaterials text dataset for GPT-2 fine-tuning."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a directory of .txt files, a .jsonl file, or omit to use "
             "built-in sample nanomaterials texts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tokenized_data",
        help="Directory to save the tokenized dataset (default: ./tokenized_data).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name for the tokenizer (default: gpt2).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token sequence length per training example (default: 512).",
    )
    args = parser.parse_args()

    # --- Load texts ---
    if args.source is None:
        print("No --source provided. Using built-in sample nanomaterials texts "
              f"({len(SAMPLE_NANOMATERIALS_TEXTS)} passages).")
        texts = SAMPLE_NANOMATERIALS_TEXTS
    elif os.path.isdir(args.source):
        texts = load_texts_from_directory(args.source)
        print(f"Loaded {len(texts)} texts from directory: {args.source}")
    elif args.source.endswith(".jsonl"):
        texts = load_texts_from_jsonl(args.source)
        print(f"Loaded {len(texts)} texts from JSONL: {args.source}")
    else:
        raise ValueError(
            f"Unrecognized source: {args.source}. "
            "Provide a directory of .txt files or a .jsonl file."
        )

    if not texts:
        raise ValueError("No texts found. Check your --source path.")

    # --- Tokenize ---
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizing and chunking (max_length={args.max_length})...")
    chunks = tokenize_and_chunk(texts, tokenizer, max_length=args.max_length)
    print(f"Created {len(chunks)} training examples.")

    # --- Save ---
    dataset = Dataset.from_list(chunks)
    dataset.save_to_disk(args.output)
    print(f"Saved tokenized dataset to: {args.output}")

    # Summary
    total_tokens = sum(len(c["input_ids"]) for c in chunks)
    print(f"\nDataset summary:")
    print(f"  Source texts:      {len(texts)}")
    print(f"  Training examples: {len(chunks)}")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Sequence length:   {args.max_length}")


if __name__ == "__main__":
    main()
