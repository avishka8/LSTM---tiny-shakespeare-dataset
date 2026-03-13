# LSTM Language Model on Tiny Shakespeare

This project trains a character-level LSTM language model on the Tiny Shakespeare dataset using PyTorch.

## Features

- Character-level tokenization
- Next-token prediction objective
- Train / Validation / Test split
- Cross-entropy loss and perplexity evaluation
- Throughput and FLOPs estimation

## Dataset

Tiny Shakespeare dataset  
40,000 lines

Split:
- Train: 30,000
- Validation: 2,000
- Test: 8,000

## Model

LSTM Language Model

Parameters:
- Embedding size: 128
- Hidden size: 256
- Layers: 1

## Training Configuration

- Sequence length: 64
- Batch size: 256
- Optimizer: AdamW
- Learning rate: 3e-4
- Epochs: 5

## Metrics

- Cross entropy loss
- Perplexity
- Training throughput
- FLOPs estimate

## Results

- Final Training Loss: 1.42
- Validation Loss: 1.51
- Test Loss: 1.53
- Validation Perplexity: 4.52
- Test Perplexity: 4.61
