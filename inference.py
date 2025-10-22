# inference.py
# Script for running inference with the trained model.
# Written by Salman Nawaz Malik. 

import torch
from model import LightweightTransformer
import argparse

def generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--max_length', type=int, default=50)
    args = parser.parse_args()

    # Load model
    model = LightweightTransformer()  # Assume vocab from training
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Tokenize prompt (simplified, assume same stoi as training)
    with open('data/sample.txt', 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    prompt_tokens = [stoi[ch] for ch in args.prompt]
    idx = torch.tensor([prompt_tokens], dtype=torch.long)

    generated = generate(model, idx, args.max_length)
    output = ''.join([itos[int(i)] for i in generated[0]])
    print(output)

if __name__ == "__main__":
    main()

# Salman Nawaz Malik: Pro tip – tweak temperature for more creative outputs. I love seeing what it generates!
