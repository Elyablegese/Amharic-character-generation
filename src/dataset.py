import torch
from torch.utils.data import Dataset
import json
import os

class CharDataset(Dataset):
    def __init__(self, path, vocab_path, block_size=512):
        assert os.path.exists(path), f"Text file not found: {path}"
        assert os.path.exists(vocab_path), f"Vocab file not found: {vocab_path}"

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        assert isinstance(self.vocab, dict), "Vocab file must be a dictionary"

        self.idx2char = {v: k for k, v in self.vocab.items()}
        self.data = [self.vocab[c] for c in text if c in self.vocab]
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        if len(chunk) < self.block_size + 1:
            raise IndexError("Sequence too short")
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
