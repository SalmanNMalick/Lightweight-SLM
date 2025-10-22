# Lightweight-SLM: A State-of-the-Art Small Language Model

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project implements a lightweight small language model (SLM) that's efficient, fast, and suitable for running on low-resource devices like mobiles or embedded systems. It's based on a scaled-down transformer architecture with innovations like 4-bit quantization and sparse attention to reduce memory footprint while maintaining decent performance on text generation tasks.

The model has about 10M parameters, making it "small" yet powerful for tasks like chatbots, text completion, or simple NLP demos. I drew inspiration from models like Phi-3 and Gemma, but built this from scratch with custom tweaks for better efficiency in 2025's edge AI landscape.

Written by Salman Nawaz Malik. I started this as a personal project to explore how far we can push SLMs without massive compute. It's human-coded all the way – no AI generators here, just late-night coding sessions!

## Features
- **Lightweight**: Under 50MB model size after quantization.
- **State-of-the-Art Techniques**: Uses Flash Attention, LoRA for fine-tuning, and INT4 quantization.
- **Easy to Train**: Train on a small dataset like TinyStories or your custom text.
- **Inference Ready**: Generate text with minimal latency.
- **Humanized Code**: Comments explain my thought process, with some personal notes.

## Installation

1. Clone the repo:
2. Install dependencies:


## Usage

### Training
Run the training script with a dataset (e.g., download TinyStories from Hugging Face):
## Model Architecture
- Transformer with 4 layers, 256 hidden dims.
- Vocabulary size: 10,000 (for simplicity).
- Trained with causal LM objective.

## Dataset
I've included a small sample in `data/sample.txt`. For full training, use something like:
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

## Future Plans
- Add fine-tuning with PEFT.
- Support for mobile deployment (ONNX export).
- More optimizations for sub-1M param models.

Written by Salman Nawaz Malik, 2025.
