import torch
from torch.utils.data import Dataset
import json

class CharDataset(Dataset):
    def __init__(self, path, vocab_path, block_size=512):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.idx2char = {v: k for k, v in self.vocab.items()}
        self.data = [self.vocab[c] for c in text if c in self.vocab]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
