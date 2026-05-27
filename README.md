# Minimal LLM

A minimal implementation of a Large Language Model (LLM) from scratch using PyTorch. This project demonstrates how to build and train a causal GPT-style Transformer for next-token prediction.

## Overview

This project implements a complete pipeline for training a minimal language model:

- **MinimalAttention**: Single-head causal attention with look-ahead masking
- **TransformerBlock**: Pre-LN architecture with residual connections
- **MinimalLLM**: Decoder-only model with token and positional embeddings
- **CharacterTokenizer**: Simple character-level encoding/decoding
- **Training Loop**: AdamW optimizer with cross-entropy loss
- **Text Generation**: Autoregressive sampling with multinomial distribution

## Requirements

- Python 3.8+
- PyTorch 2.0+

## Installation

Install PyTorch using pip:

```bash
pip install torch
```

For GPU support, follow the official PyTorch installation guide:
https://pytorch.org/get-started/locally/

## Usage

Run the training script:

```bash
python train_minimal_llm.py
```

## Example Output

```
==================================================
Minimal LLM Training
==================================================

Vocabulary size: 22
Vocabulary: [' ', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y']

Model parameters: 406,550
Training on: cpu
Dataset size: 158 tokens
Batch size: 8, Sequence length: 32
--------------------------------------------------
Step 0: Loss = 3.3215
Step 100: Loss = 0.3792
Step 200: Loss = 0.1383
...
Step 900: Loss = 0.0837
--------------------------------------------------
Training complete!

==================================================
Text Generation
==================================================

Prompt: 'the big '
Generated: the big dog played with the

Prompt: 'the small '
Generated: the small house.e. the cat and

Prompt: 'the cat and '
Generated: the cat and the dog are friends.
```

## Architecture

### Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding Dimension | 128 | Size of token and positional embeddings |
| Number of Layers | 2 | Number of Transformer blocks |
| Max Sequence Length | 32 | Maximum input sequence length |
| Vocabulary Size | 22 | Character-level vocabulary |

### Model Components

1. **Token Embedding**: Maps input tokens to dense vectors
2. **Positional Embedding**: Adds position information to token embeddings
3. **Transformer Blocks**: Two layers of self-attention with feed-forward networks
4. **Layer Normalization**: Applied before attention and feed-forward layers
5. **LM Head**: Projects hidden states to vocabulary logits

## How It Works

### 1. Data Preparation

The model uses a simple character-level tokenizer:

```python
text = "the big cat sat on the big mat..."
chars = sorted(list(set(text)))  # Get unique characters
char_to_int = {ch: i for i, ch in enumerate(chars)}  # Create mapping
```

### 2. Causal Attention

The attention mechanism uses a causal mask to prevent looking ahead:

```python
mask = torch.tril(torch.ones(T, T)).view(1, T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### 3. Training

The model is trained to predict the next token using cross-entropy loss:

```python
logits = model(x)  # Forward pass
loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
loss.backward()
optimizer.step()
```

### 4. Text Generation

Text is generated autoregressively by sampling from the output distribution:

```python
probs = F.softmax(logits[:, -1, :], dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

## Customization

### Using Custom Text Data

Modify the `text` variable in the `main()` function:

```python
text = """
Your custom training text here.
The more text you provide, the better the model will learn.
"""
```

### Adjusting Hyperparameters

```python
model = MinimalLLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=256,      # Increase for more capacity
    num_layers=4,       # More layers for complex patterns
    max_seq_len=64      # Longer context
)
```

### Training Configuration

```python
model = train_model(
    model,
    data,
    epochs=2000,        # More training steps
    batch_size=16,      # Larger batches
    max_seq_len=32,
    lr=1e-3,
    device="cuda"       # Use GPU if available
)
```

## Limitations

- **Small Vocabulary**: Character-level tokenization is inefficient for large-scale applications
- **Limited Context**: Short sequence length restricts long-range dependencies
- **Single-Head Attention**: Multi-head attention would improve representation capacity
- **Small Dataset**: The example uses minimal training data for demonstration

## Future Improvements

1. **Byte-Pair Encoding (BPE)**: Implement subword tokenization for better efficiency
2. **Multi-Head Attention**: Add support for multiple attention heads
3. **Positional Encodings**: Use learned or sinusoidal positional encodings
4. **Evaluation Metrics**: Add perplexity and other evaluation metrics
5. **Checkpointing**: Save and load model weights
6. **Larger Datasets**: Support for loading external text corpora

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is provided for educational purposes.