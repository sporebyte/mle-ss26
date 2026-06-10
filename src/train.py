# src/train.py
"""
Contains:

1 - setup: seed, device, output directories
2 - data loading, train/val split, tokenizer building
3 - dataset and dataloader creation for PyTorch
4 - model building 
5 - optimizer and loss function setup
6 - main training loop with per-epoch validation and checkpoint saving
7 - save loss history for plotting
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # progress bar for training epochs

from preprocess import load_raw
from smiles_model import SMILESTokenizer, SMILESDataset, make_collate_fn, RNNModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Reproducibility ───────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed) # seeds the Apple Silicon GPU RNG


# ── Device ────────────────────────────────────────────────────

# Pick best available compute device
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Train/val split ───────────────────────────────────────────

def split_data(smiles: list[str], val_frac: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    shuffled = smiles[:] # copy to avoid shuffling original list
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


# ── One epoch ─────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train() # switches the model to training mode
    total_loss = 0.0
    n_batches = 0

    # iterate over DataLoader batches with a progress bar:
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        # Shift: input is everything except the last token,
        # target is everything except the first token.
        x = batch[:, :-1]
        y = batch[:, 1:]

        optimizer.zero_grad() # clear gradients from the previous batch
        logits, _ = model(x) # forward pass: get predicted token probabilities
        loss = criterion( # criterion is nn.CrossEntropyLoss, reshape tensors
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward() # backpropagation: compute gradients of loss w.r.t. model parameters
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches # average loss across the epoch


@torch.no_grad() # disable gradient tracking for validation, saves memory and computations
def validate(model, loader, criterion, device):
    model.eval() # switches the model to evaluation mode
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits, _ = model(x)
        loss = criterion( # criterion is nn.CrossEntropyLoss, reshape tensors
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


# ── Main training loop ────────────────────────────────────────

def train(
    data_path: str = str(PROJECT_ROOT / "data" / "smiles_clean.txt"),
    out_dir: str = str(PROJECT_ROOT / "outputs"),
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    grad_clip: float = 1.0,
    embedding_dim: int = 128,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.2,
    val_frac: float = 0.05,
    seed: int = 42,
    num_workers: int = 4,
):
    set_seed(seed)
    device = get_device()
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print("Loading SMILES...")
    smiles = load_raw(data_path)

    # --- Tokenizer ---
    print("Building tokenizer...")
    tokenizer = SMILESTokenizer(smiles)
    tokenizer.save(out / "tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # --- Split ---
    train_smiles, val_smiles = split_data(smiles, val_frac, seed)
    print(f"Train: {len(train_smiles):,} | Val: {len(val_smiles):,}")

    # --- Datasets and loaders ---
    train_ds = SMILESDataset(train_smiles, tokenizer)
    val_ds = SMILESDataset(val_smiles, tokenizer)
    collate = make_collate_fn(tokenizer.pad_id)

    common = dict( # common DataLoader args for both train and val loaders
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    # --- Model, optimizer, loss ---
    """build the model with the configured hyperparameters and ship it to the GPU"""
    model = RNNModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # count trainable parameters
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # --- Train ---
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []} # a dict to record the loss curves for later plotting

    for epoch in range(1, epochs + 1): #  loop from 1 to 30 epochs, formatted for printing 1-indexed epoch numbers
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip
        )
        val_loss = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f}"
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = out / "checkpoints" / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "embedding_dim": embedding_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "pad_id": tokenizer.pad_id,
                    },
                },
                ckpt_path,
            )
            print(f"  ↳ saved {ckpt_path} (new best)")

    # Save history for plotting
    import json
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()