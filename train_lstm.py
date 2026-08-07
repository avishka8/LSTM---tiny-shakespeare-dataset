"""
train_lstm.py

Character-level LSTM language model on the Tiny Shakespeare dataset.
Matches the architecture reported in the project README:
    - Embedding size: 128
    - Hidden size: 256
    - Layers: 1
    - Sequence length: 64
    - Optimizer: AdamW, lr=3e-4

Trains the model, tracks train/val loss per epoch, plots a loss curve,
and generates sample text from the trained model — producing the
artifacts referenced in the README (loss_curve.png, sample text block).

Run:
    python train_lstm.py
"""

import math
import random
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config (matches README-reported architecture)
# ---------------------------------------------------------------------------
DATA_PATH = "tiny_shakespeare.txt"
SEQ_LEN = 64
BATCH_SIZE = 256
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
LR = 3e-4
EPOCHS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss


@torch.no_grad()
def generate(model, stoi, itos, prompt="ROMEO:", length=300, temperature=0.8):
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long).to(DEVICE)
    hidden = None
    generated = prompt

    for _ in range(length):
        logits, hidden = model(idx[:, -1:], hidden) if generated != prompt else model(idx)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_char = itos[next_id.item()]
        generated += next_char
        idx = torch.cat([idx, next_id], dim=1)

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    data, stoi, itos, vocab_size = load_data(DATA_PATH)
    print(f"Dataset length: {len(data)} chars | Vocab size: {vocab_size}")

    n = len(data)
    train_end = int(n * 0.75)
    val_end = int(n * 0.80)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_loader = DataLoader(CharDataset(train_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CharDataset(val_data, SEQ_LEN), batch_size=BATCH_SIZE)
    test_loader = DataLoader(CharDataset(test_data, SEQ_LEN), batch_size=BATCH_SIZE)

    model = LSTMLanguageModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Val perplexity: {math.exp(val_loss):.2f}")

    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed/60:.1f} min")

    test_loss = run_epoch(model, test_loader, criterion)
    test_ppl = math.exp(test_loss)
    print(f"Test loss: {test_loss:.4f} | Test perplexity: {test_ppl:.2f}")

    # --- Plot loss curve ---
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Train loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("LSTM Training Loss — Tiny Shakespeare")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Saved loss_curve.png")

    # --- Generate sample text ---
    print("\n=== Sample generated text ===")
    for prompt in ["ROMEO:", "KING:", "First Citizen:"]:
        sample = generate(model, stoi, itos, prompt=prompt, length=250)
        print(f"\n--- Prompt: {prompt!r} ---\n{sample}\n")

    with open("sample_output.txt", "w", encoding="utf-8") as f:
        for prompt in ["ROMEO:", "KING:", "First Citizen:"]:
            sample = generate(model, stoi, itos, prompt=prompt, length=250)
            f.write(f"--- Prompt: {prompt!r} ---\n{sample}\n\n")
    print("Saved sample_output.txt")


if __name__ == "__main__":
    main()
