import torch
import argparse
import os
import json
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from dataset import CharDataset
from model import GPTLM
from tqdm import tqdm

# ------------------ CLI Config ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument("--train_file", type=str, default="data/input.txt")
parser.add_argument("--vocab_file", type=str, default="data/vocab.json")
parser.add_argument("--save_path", type=str, default="checkpoints")
parser.add_argument("--n_embed", type=int, default=512)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
torch.manual_seed(1337)

# ------------------ Load and Prepare Data ------------------
if not os.path.exists(args.vocab_file):
    with open(args.train_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    with open(args.vocab_file, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False)
else:
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        stoi = json.load(f)

dataset = CharDataset(args.train_file, args.vocab_file, args.block_size)
train_len = int(0.9 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTLM(vocab_size=len(stoi), n_embed=args.n_embed, block_size=args.block_size,
              n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout, device=device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
scaler = GradScaler()

# ------------------ Training Loop ------------------
best_val_loss = float('inf')
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        with autocast():
            logits, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # ------------------ Validation ------------------
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                _, loss = model(xb, yb)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{args.save_path}/best_model.pt")
        print("✅ Saved best model.")

print("🎉 Training complete.")
