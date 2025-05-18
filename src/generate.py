import torch
import json
import argparse
from model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="አማርኛ")
parser.add_argument("--model_path", type=str, default="checkpoints/model_epoch10.pt")
parser.add_argument("--vocab_file", type=str, default="data/vocab.json")
args = parser.parse_args()

BLOCK_SIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.vocab_file, "r", encoding="utf-8") as f:
    vocab = json.load(f)
idx2char = {v: k for k, v in vocab.items()}

model = get_model(len(vocab)).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

def generate(prompt, max_new=100):
    input_ids = torch.tensor([vocab[c] for c in prompt if c in vocab], dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return ''.join([idx2char[i.item()] for i in input_ids[0]])

print("Generated Text:")
print(generate(args.prompt))