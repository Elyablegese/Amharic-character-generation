import torch
import argparse
import time
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import matplotlib.pyplot as plt
from dataset import CharDataset
from model import get_model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--block_size", type=int, default=512)
parser.add_argument("--train_file", type=str, default="data/amharic_char_train.txt")
parser.add_argument("--vocab_file", type=str, default="data/vocab.json")
parser.add_argument("--save_path", type=str, default="checkpoints")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CharDataset(args.train_file, args.vocab_file, args.block_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

model = get_model(len(dataset.vocab)).to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(dataloader)*args.epochs)

scaler = torch.cuda.amp.GradScaler()  # mixed precision

train_losses = []
model.train()
start = time.time()
for epoch in range(args.epochs):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(x, labels=y)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"{args.save_path}/model_epoch{epoch+1}.pt")

print("Training complete in", round(time.time() - start, 2), "seconds")
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")