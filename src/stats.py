# src/stats.py
import json
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
from rdkit import Chem
from rdkit import RDLogger
from matplotlib import pyplot as plt
from preprocess import load_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"


# Character categorization for SMILES

ATOM_CHARS = set("CNOSPFIBrClcnops")        # uppercase = aliphatic, lowercase = aromatic
BOND_CHARS = set("-=#:/\\")                 # single, double, triple, aromatic, stereo
RING_CHARS = set("0123456789")              # ring closure digits
BRACKET_CHARS = set("()[]")                 # branches and atom specs
OTHER_CHARS = set("+-.@%")                  # charges, fragment dot, chirality, %nn rings


def categorize(c: str) -> str:
    if c in ATOM_CHARS:    return "atom"
    if c in BOND_CHARS:    return "bond"
    if c in RING_CHARS:    return "ring digit"
    if c in BRACKET_CHARS: return "bracket"
    return "other"


# Stats computation

def heavy_atom_count(s: str) -> int:
    return Chem.MolFromSmiles(s).GetNumHeavyAtoms()

def compute_stats(smiles: list[str]) -> dict:
    print("Computing string stats")
    lengths = [len(s) for s in smiles]
    char_counts = dict(Counter(c for s in smiles for c in s))
    n_with_dot = sum(1 for s in smiles if '.' in s)

    print("Computing heavy atom counts")
    with Pool() as pool:
        heavy_atoms = pool.map(heavy_atom_count, smiles)

    return {
        "n_total": len(smiles),
        "lengths": lengths,
        "char_counts": char_counts,
        "n_with_dot": n_with_dot,
        "heavy_atoms": heavy_atoms,
    }


def get_stats(smiles_path: str, cache_path: str = str(OUTPUTS / "stats.json")) -> dict:
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached stats from {cache}")
        with open(cache) as f:
            return json.load(f)

    smiles = load_raw(smiles_path)
    stats = compute_stats(smiles)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'w') as f:
        json.dump(stats, f)
    print(f"Cached stats to {cache}")
    return stats


# Plotting

def plot_stats_comparison(
    train_stats: dict,
    gen_stats: dict,
    out_path: str = str(OUTPUTS / "raw_stats.png")
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    datasets = [
        ("Training", train_stats),
        ("Generated", gen_stats),
    ]

    for row, (title_prefix, stats) in enumerate(datasets):

        # Length distribution
        axes[row, 0].hist(stats["lengths"], bins=50)
        axes[row, 0].set_title(f"{title_prefix}: Length")

        # Atom frequency
        atom_counts = {
            c: n for c, n in stats["char_counts"].items()
            if c in ATOM_CHARS
        }

        chars_sorted = sorted(
            atom_counts.items(),
            key=lambda kv: kv[1],
            reverse=True
        )

        chars, counts = zip(*chars_sorted)

        axes[row, 1].bar(chars, counts)
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_title(f"{title_prefix}: Atom freq")

        # Heavy atom count
        axes[row, 2].hist(stats["heavy_atoms"], bins=50)
        axes[row, 2].set_title(f"{title_prefix}: Heavy atoms")

        # Character composition
        category_counts = Counter()

        for c, n in stats["char_counts"].items():
            category_counts[categorize(c)] += n

        cats, vals = zip(*category_counts.most_common())

        axes[row, 3].bar(cats, vals)
        axes[row, 3].set_title(f"{title_prefix}: Categories")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    train_stats = compute_stats(
        load_raw(PROJECT_ROOT / "data" / "smiles_clean.txt")
    )
    gen_stats = compute_stats(
        load_raw(OUTPUTS / "submission.txt")
    )
    plot_stats_comparison(train_stats, gen_stats)