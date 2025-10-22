# train.py
# Script to train the model on a text dataset.
# Written by Salman Nawaz Malik.

import torch
from model import LightweightTransformer
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os

def get_batch(data, batch_size, block_size):
    # Salman: Quick helper to grab batches. Human touch: I debugged this a lot!
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sample.txt', help='Path to text data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # Load data - for simplicity, assume text file. In real, use HF datasets.
    with open(args.data_path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    model = LightweightTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    block_size = model.block_size

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for _ in tqdm(range(1000)):  # Arbitrary steps, adjust based on data
            xb, yb = get_batch(data, args.batch_size, block_size)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")

    torch.save(model.state_dict(), 'model.pth')
    print("Training done! Model saved.")

if __name__ == "__main__":
    main()

# Written by Salman Nawaz Malik. Note: For better data, replace with HF load_dataset('roneneldan/TinyStories').
