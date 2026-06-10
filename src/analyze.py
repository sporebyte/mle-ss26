# src/analyze.py
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, Lipinski

from preprocess import load_raw

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

# ── Lipinski roperty ──────────────────────────────────────

def compute_properties(smiles: list[str]) -> dict:
    """Compute 5 molecular properties for a list of SMILES."""
    props = {k: [] for k in ["mw", "logp", "hbd", "hba", "qed"]}
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        props["mw"].append(Descriptors.MolWt(mol))
        props["logp"].append(Descriptors.MolLogP(mol))
        props["hbd"].append(Lipinski.NumHDonors(mol))
        props["hba"].append(Lipinski.NumHAcceptors(mol))
        props["qed"].append(QED.qed(mol))
    return props


def plot_property_distributions(gen_props: dict, train_props: dict,
                                out_path: str = str(OUTPUTS / "properties.png")):
    """Overlaid histograms: generated vs training, per property."""
    items = [
        ("mw",   "Molecular weight (Da)", (0, 800)),
        ("logp", "LogP",                  (-5, 10)),
        ("hbd",  "H-bond donors",         (0, 10)),
        ("hba",  "H-bond acceptors",      (0, 15)),
        ("qed",  "QED (drug-likeness)",   (0, 1)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (key, label, rng) in zip(axes, items):
        ax.hist(train_props[key], bins=40, range=rng, alpha=0.5,
                label="train", density=True)
        ax.hist(gen_props[key], bins=40, range=rng, alpha=0.5,
                label="generated", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Functional group analysis ─────────────────────────────────

FUNCTIONAL_GROUPS = {
    "amide":           "C(=O)N",
    "ester":           "C(=O)OC",
    "carboxylic acid": "C(=O)[OH]",
    "ketone":          "[#6]C(=O)[#6]",
    "ether":           "[#6]O[#6]",
    "amine":           "[NX3;H2,H1,H0;!$(NC=O)]",
    "sulfonamide":     "S(=O)(=O)N",
    "halide":          "[F,Cl,Br,I]",
    "nitro":           "[N+](=O)[O-]",
    "nitrile":         "C#N",
    "aromatic ring":   "a1aaaaa1",
}


def functional_group_freqs(smiles: list[str]) -> dict[str, float]:
    """Fraction of molecules containing each functional group."""
    patterns = {name: Chem.MolFromSmarts(pat)
                for name, pat in FUNCTIONAL_GROUPS.items()}
    counts = {name: 0 for name in patterns}
    n_valid = 0
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        n_valid += 1
        for name, pat in patterns.items():
            if mol.HasSubstructMatch(pat):
                counts[name] += 1
    return {name: counts[name] / n_valid for name in counts}


def plot_functional_groups(gen: list[str], train: list[str],
                           out_path: str = str(OUTPUTS / "functional_groups.png")):
    gen_freqs = functional_group_freqs(gen)
    train_freqs = functional_group_freqs(train)

    names = list(FUNCTIONAL_GROUPS.keys())
    x = np.arange(len(names))
    width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, [train_freqs[n] for n in names],
            width, label="train", alpha=0.7)
    plt.bar(x + width/2, [gen_freqs[n] for n in names],
            width, label="generated", alpha=0.7)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Fraction of molecules containing group")
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    submission_path = OUTPUTS / "submission.txt"
    input_path = PROJECT_ROOT / "data" / "smiles_clean.txt"
    gen = load_raw(submission_path)
    train_full = load_raw(input_path)
    train = random.Random(42).sample(train_full, 100_000)

    print(f"Generated: {len(gen):,}  |  Training subsample: {len(train):,}\n")

    # 1. Property distributions (MW, LogP, HBD, HBA, QED)
    print("Computing properties...")
    gen_props = compute_properties(gen)
    train_props = compute_properties(train)
    plot_property_distributions(gen_props, train_props)

    # 2. Functional group frequencies
    print("\nComputing functional group frequencies...")
    plot_functional_groups(gen, train)