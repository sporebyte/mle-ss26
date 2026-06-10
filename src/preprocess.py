# src/preprocess.py
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm

# Relative path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent          

# Load raw SMILE strings from a text file, stripping whitespace and ignoring empty lines
def load_raw(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# Canonicalize a SMILES string using RDKit; returns None if the input is invalid
def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

# Main preprocessing function: loads raw SMILES, canonicalizes them, removes duplicates, and saves the cleaned list
def preprocess(in_path: str, out_path: str):
    raw = load_raw(in_path)
    print(f"Loaded {len(raw):,} raw SMILES")

    canon = []
    n_invalid = 0
    for s in tqdm(raw, desc="Canonicalizing"):
        c = canonicalize(s)
        if c is None:
            n_invalid += 1
        else:
            canon.append(c)
    print(f"Removed {n_invalid:,} invalid SMILES")

    before = len(canon)
    clean = list(dict.fromkeys(canon))
    print(f"Removed {before - len(clean):,} duplicates")
    print(f"Final dataset: {len(clean):,} SMILES")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(clean))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    preprocess(
        in_path=PROJECT_ROOT / "data" / "smiles_train.txt",
        out_path=PROJECT_ROOT / "data" / "smiles_clean.txt",
    )
