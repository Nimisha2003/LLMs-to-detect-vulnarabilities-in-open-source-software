import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import json
import time
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import CustomTokenizer
from model import SimpleTransformerClassifier

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256           # shorter sequences = faster, safer
BATCH_SIZE = 2          # drop to 1 if memory spikes
NUM_EPOCHS = 3          # start small, scale later
LR = 1e-3
CHECKPOINT_PATH = "vuln_model.pth"
TOKENIZER_PATH = "tokenizer.json"
DATA_PATH = "merged_dataset.jsonl"

torch.set_num_threads(1)  # avoid CPU thread contention

# -----------------------------
# Dataset
# -----------------------------
class CodeDataset(Dataset):
    def __init__(self, code_lines, labels, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [self.tokenizer.encode(line, max_length=self.max_len) for line in code_lines]
        self.labels = [float(lbl) for lbl in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def collate_pad(batch):
    sequences, labels = zip(*batch)
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        padded.append(seq)
    x = torch.stack(padded, dim=0)
    y = torch.stack(labels, dim=0)
    return x, y

# -----------------------------
# Load dataset
# -----------------------------
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

code_samples = [item.get("code", "") for item in data]
labels = [int(item.get("label", 0)) for item in data]

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = CustomTokenizer()
tokenizer.build_vocab(code_samples)

# -----------------------------
# Dataset + Split
# -----------------------------
dataset = CodeDataset(code_samples, labels, tokenizer, max_len=MAX_LEN)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

# -----------------------------
# Model + Training
# -----------------------------
model = SimpleTransformerClassifier(vocab_size=tokenizer.vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def save_checkpoint():
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": {str(k): v for k, v in tokenizer.id_to_token.items()},
            "vocab_size": tokenizer.vocab_size
        }, f)
    print("Checkpoint saved.")

# -----------------------------
# Training loop
# -----------------------------
try:
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        start = time.time()
        for i, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).view(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if i % 200 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / max(1, len(val_loader))

        dur = time.time() - start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | {dur:.1f}s")
        save_checkpoint()

    print("Training completed.")
except KeyboardInterrupt:
    print("\nInterrupted. Saving checkpoint...")
    save_checkpoint()
    sys.exit(0)