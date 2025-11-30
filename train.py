import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import json
from tokenizer import CustomTokenizer
from model import SimpleTransformerClassifier
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Force CPU usage to avoid DLL issues
torch.device('cpu')

class CodeDataset(Dataset):
    def __init__(self, code_lines, labels, tokenizer):
        self.data = [tokenizer.encode(line) for line in code_lines]
        self.labels = labels
        self.max_len = max(len(seq) for seq in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x += [0] * (self.max_len - len(x))  # padding
        return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.float32)

# --------------------
# Load dataset
# --------------------
with open("dataset.json", "r") as f:
    data = json.load(f)

code_samples = [item["code"] for item in data]
labels = [item["label"] for item in data]

# --------------------
# Tokenization
# --------------------
tokenizer = CustomTokenizer()
tokenizer.build_vocab(code_samples)

# --------------------
# Dataset + Split
# --------------------
dataset = CodeDataset(code_samples, labels, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# --------------------
# Model + Training
# --------------------
model = SimpleTransformerClassifier(vocab_size=tokenizer.vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --------------------
# Save model + tokenizer
# --------------------
torch.save(model.state_dict(), "vuln_model.pth")

with open("tokenizer.json", "w") as f:
    json.dump({
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": {str(k): v for k, v in tokenizer.id_to_token.items()},
        "vocab_size": tokenizer.vocab_size
    }, f)

print("✅ Model and tokenizer saved successfully!")
