# src/generate.py
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

from smiles_model import SMILESTokenizer, RNNModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> RNNModel:
    """Reconstruct the model from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RNNModel(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_id=cfg["pad_id"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


@torch.no_grad()
def sample_batch(
    model: RNNModel,
    tokenizer: SMILESTokenizer,
    batch_size: int,
    max_len: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """Sample a batch of SMILES strings in parallel, one token at a time."""
    # Start each sequence with <SOS>
    current = torch.full(
        (batch_size, 1), tokenizer.sos_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_ids = [current]
    hidden = None

    for _ in range(max_len):
        # Feed only the most recent token (using hidden state for context)
        logits, hidden = model(current, hidden)
        # logits: (batch, 1, vocab_size) - take last position
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        current = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        all_ids.append(current)

        # Mark sequences that just emitted <EOS>
        finished = finished | (current.squeeze(-1) == tokenizer.eos_id)
        if finished.all():
            break

    # Decode each sequence
    sequences = torch.cat(all_ids, dim=1).tolist()
    return [tokenizer.decode(seq) for seq in sequences]


# -- Generate only canonical SMILEs --
def is_valid_smiles(s: str) -> bool:
    if not s:
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None

def canonicalize(s: str) -> str | None:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

# -- Main function --
def generate(
    checkpoint_path: str = str(OUTPUTS / "checkpoints" / "best.pt"),
    tokenizer_path: str = str(OUTPUTS / "tokenizer.json"),
    output_path: str = str(OUTPUTS / "submission.txt"),
    n_required: int = 10000, # generate 10 000 molecules
    batch_size: int = 512, # larger batch size means faster generation
    max_len: int = 150, # training data has max length of 143, so 150 should be enough
    temperature: float = 1.0, # can be overriden when running generate.py for sweeping temperatures
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Temperature: {temperature}")

    # Load
    tokenizer = SMILESTokenizer.load(tokenizer_path)
    model = load_model(checkpoint_path, device)

    # Sample until we have enough valid, unique, canonical SMILES
    seen: set[str] = set()
    valid_canonical: list[str] = []
    n_attempts = 0
    n_invalid = 0

    pbar = tqdm(total=n_required, desc="Generating")
    while len(valid_canonical) < n_required:
        raw_batch = sample_batch(
            model, tokenizer, batch_size, max_len, temperature, device
        )
        n_attempts += len(raw_batch)

        for s in raw_batch:
            canon = canonicalize(s)
            if canon is None:
                n_invalid += 1
                continue
            if canon in seen:
                continue
            seen.add(canon)
            valid_canonical.append(canon)
            pbar.update(1)
            if len(valid_canonical) >= n_required:
                break
    pbar.close()

    # Stats
    validity = (n_attempts - n_invalid) / n_attempts
    print(f"\nAttempts: {n_attempts:,}")
    print(f"Raw validity: {validity:.4f}")
    print(f"Unique canonical molecules: {len(valid_canonical):,}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(valid_canonical[:n_required]))
    print(f"Saved {n_required:,} SMILES to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(OUTPUTS / "checkpoints" / "best.pt"))
    parser.add_argument("--tokenizer", default=str(OUTPUTS / "tokenizer.json"))
    parser.add_argument("--output", default=str(OUTPUTS / "submission.txt"))
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        n_required=args.n,
        batch_size=args.batch_size,
        max_len=args.max_len,
        temperature=args.temperature,
        seed=args.seed,
    )