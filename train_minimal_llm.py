"""
Minimal LLM Training Script
A complete implementation of a causal GPT-style Transformer for next-token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Network Architecture
# ============================================================================

class MinimalAttention(nn.Module):
    """Single-head causal attention module."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Combine Q, K, V projections into one linear layer for efficiency
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        
        # Project to Query, Key, and Value matrices
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        
        # Apply causal mask (look-ahead mask)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get weights and compute weighted sum of values
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """A single Transformer block with pre-LN architecture."""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MinimalAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Pre-LN residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MinimalLLM(nn.Module):
    """Minimal Decoder-Only Language Model."""
    
    def __init__(self, vocab_size, embed_dim=128, num_layers=2, max_seq_len=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Combine token and positional representations
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        return logits


# ============================================================================
# Data Preparation
# ============================================================================

class CharacterTokenizer:
    """Simple character-level tokenizer for minimal LLM."""
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, s):
        return [self.char_to_int[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.int_to_char[i] for i in l])
    
    def encode_tensor(self, text, dtype=torch.long):
        return torch.tensor(self.encode(text), dtype=dtype)


def get_batch(data, batch_size=4, max_seq_len=16):
    """Create mini-batches with shifted targets for next-token prediction."""
    ix = torch.randint(len(data) - max_seq_len, (batch_size,))
    x = torch.stack([data[i:i+max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+max_seq_len+1] for i in ix])  # Target shifted by 1
    return x, y


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, data, epochs=500, batch_size=8, max_seq_len=16, lr=1e-3, device="cpu"):
    """Train the minimal LLM on the provided data."""
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training on: {device}")
    print(f"Dataset size: {len(data)} tokens")
    print(f"Batch size: {batch_size}, Sequence length: {max_seq_len}")
    print("-" * 50)
    
    model.train()
    for step in range(epochs):
        xb, yb = get_batch(data, batch_size=batch_size, max_seq_len=max_seq_len)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        logits = model(xb)
        
        # Reshape tensors for PyTorch's cross entropy loss
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("-" * 50)
    print("Training complete!")
    return model


# ============================================================================
# Text Generation
# ============================================================================

def generate(model, prompt, tokenizer, max_new_tokens=20, max_seq_len=16, device="cpu"):
    """Generate text autoregressively from a prompt."""
    model.eval()
    
    # Encode the prompt
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print(f"\nPrompt: '{prompt}'")
    print("Generated:", end=" ")
    print(prompt, end="")
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds max length
        context_cond = context[:, -max_seq_len:]
        
        with torch.no_grad():
            logits = model(context_cond)
            
            # Focus only on the last time step output
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Sample next index from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
    
    # Decode and return the full generated text
    generated_text = tokenizer.decode(context[0].tolist())
    print(generated_text[len(prompt):])
    
    return generated_text


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function to train and test the minimal LLM."""
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 50)
    print("Minimal LLM Training")
    print("=" * 50)
    
    # Sample text corpus (character-level)
    text = "the big cat sat on the big mat. the small dog ran to the small house. the cat and the dog are friends. the big dog played with the small cat in the big house."
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.chars}")
    
    # Convert corpus to tensor
    data = tokenizer.encode_tensor(text)
    
    # Model hyperparameters
    embed_dim = 128
    num_layers = 2
    max_seq_len = 32
    
    # Initialize model
    model = MinimalLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Train the model
    model = train_model(
        model,
        data,
        epochs=1000,
        batch_size=8,
        max_seq_len=max_seq_len,
        lr=1e-3,
        device=device
    )
    
    # Generate text
    print("\n" + "=" * 50)
    print("Text Generation")
    print("=" * 50)
    
    prompts = ["the big ", "the small ", "the cat and "]
    
    for prompt in prompts:
        generate(model, prompt, tokenizer, max_new_tokens=20, max_seq_len=max_seq_len, device=device)
        print("-" * 30)


if __name__ == "__main__":
    main()