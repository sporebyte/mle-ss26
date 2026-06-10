# src/smiles_model.py
"""
tokenizer + dataset + model for SMILES generation

Contains:
  - tokenize / SMILES_REGEX  : atom-wise SMILES tokenization (Schwaller et al.)
  - SMILESTokenizer          : vocab + encode/decode + save/load
  - SMILESDataset            : PyTorch Dataset over a list of SMILES
  - make_collate_fn          : per-batch padding collate function
  - RNNModel                 : embedding -> stacked LSTM -> linear head
"""

import re
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ── Tokenization ────────────────────────────────────────────────

# Schwaller et al. SMILES tokenization regex, same one DeepChem uses
# Matches atoms (Cl, Br, [nH], [N+], etc.) and structural characters as single tokens

# re.compile() pre-complies a regex pattern into a reusable object
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"
)


# Tokenize function applies the regex to split a SMILES string into a list of tokens
def tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into atom-wise tokens."""
    return SMILES_REGEX.findall(smiles)


class SMILESTokenizer:
    """Atom-wise SMILES tokenizer with vocab built from training data."""

    # Special tokens for padding, start of sequence, and end of sequence
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    SPECIAL_TOKENS = [PAD, SOS, EOS]

    def __init__(self, smiles_list: list[str] | None = None):
        if smiles_list is not None:
            self._build_vocab(smiles_list)

    def _build_vocab(self, smiles_list: list[str]):
        """Scan training data, collect unique tokens, build mappings."""
        token_counter = Counter() # counts how many times each token appears in the dataset
        for s in smiles_list:
            token_counter.update(tokenize(s))

        # Special tokens first (IDs 0, 1, 2), then data tokens sorted for determinism.
        data_tokens = sorted(token_counter.keys())
        all_tokens = self.SPECIAL_TOKENS + data_tokens

        # Dict comprehension:
        # create token_to_id mapping: {token: index}
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # store the integer ID for each special token for easy access
        self.pad_id = self.token_to_id[self.PAD]
        self.sos_id = self.token_to_id[self.SOS]
        self.eos_id = self.token_to_id[self.EOS]

    # @property decorator to make vocab_size an attribute that computes len(token_to_id)
    # returns the number of unique tokens in the vocabulary, including special tokens 
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, smiles: str) -> list[int]:
        """turns SMILES string into a list of integer IDs that the model can process"""
        tokens = tokenize(smiles)

        # List comprehension: 
        # For each token t in the token list, look up its integer ID and put it in a new list
        ids = [self.token_to_id[t] for t in tokens]
        return [self.sos_id] + ids + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """IDs -> SMILES string. Stops at <EOS>, strips special tokens"""
        tokens = []
        for i in ids:
            tok = self.id_to_token[i]
            if tok == self.EOS:
                break
            if tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    # Save the tokenizer method output to JSON file
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    # @classmethod is used to create a method that works with the class instead of an object instance
    # receives the class itself as the first argument using cls, and a path to the JSON file containing the token_to_id mapping
    @classmethod
    def load(cls, path) -> "SMILESTokenizer":
        with open(path) as f:
            token_to_id = json.load(f) # parse the JSON into python dict
        tok = cls()  # empty SMILESTokenizer instance
        tok.token_to_id = token_to_id
        tok.id_to_token = {i: t for t, i in token_to_id.items()}
        tok.pad_id = token_to_id[cls.PAD]
        tok.sos_id = token_to_id[cls.SOS]
        tok.eos_id = token_to_id[cls.EOS]
        return tok


# ── Dataset ───────────────────────────────────────────────────

class SMILESDataset(Dataset): # subclass PyTorch's Dataset class
    """Wraps a list of SMILES + a tokenizer. Returns integer ID tensors"""

    def __init__(self, smiles: list[str], tokenizer: SMILESTokenizer):
        self.smiles = smiles
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.smiles[idx])
        return torch.tensor(ids, dtype=torch.long)

# Resolves the PyTorch DataLoader's __getitem__ outputs into a single batch tensor
# ... padding the variable-length sequences in the batch to the same length using the pad_id for padding
def make_collate_fn(pad_id: int):
    """Returns a collate function that pads variable-length sequences per batch"""
    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        # pad_sequence stacks into (batch_size, max_len_in_batch)
        return pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return collate


# ── Model ─────────────────────────────────────────────────────

class RNNModel(nn.Module): # subclass PyTorch's nn.Module to define the neural network architecture
    """Atom-wise LSTM language model for SMILES generation"""

    # Define the model architecture and parameters
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__() # call the parent class constructor nn.Module
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id # ignore padding token in the embedding layer
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size) # final linear layer, score per vocab token

    # PyTorch's forward method defines the computation performed at every call of the model
    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x:      (batch, seq_len)   integer IDs
        hidden: optional initial (h, c) for the LSTM (used during generation)

        returns:
            logits (batch, seq_len, vocab_size)
            (h, c) final hidden state
        """
        x = self.embedding(x)               # (batch, seq_len, embedding_dim)
        out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_dim)
        logits = self.head(out)             # (batch, seq_len, vocab_size)
        return logits, hidden