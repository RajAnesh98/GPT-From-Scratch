# models/gpt_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BLOCK_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE

class AttentionHead(nn.Module):
    """
    Implements a single self-attention head.
    """
    def __init__(self, head_dim):
        super().__init__()
        # Linear projections for keys, queries, and values.
        self.key = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_dim, bias=False)
        # Create a lower-triangular mask for causal (autoregressive) attention.
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, x):
        """
        Computes self-attention for one head.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, T, EMBED_DIM)
        
        Returns:
            torch.Tensor: Output tensor after attention.
        """
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute scaled dot-product attention scores.
        attn_scores = q @ k.transpose(-2, -1) * (x.size(-1) ** -0.5)
        # Apply causal mask.
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        v = self.value(x)
        out = attn_weights @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        # Create a list of attention heads.
        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(num_heads)])
        # Linear projection to combine the outputs of all heads.
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, x):
        # Concatenate the outputs from all heads along the channel dimension.
        concat_out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(concat_out))
        return out

class FeedForward(nn.Module):
    """
    Feed-forward network used within the transformer block.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(DROPOUT_RATE),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single transformer block that includes multi-head attention and a feed-forward network.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.mha = MultiHeadAttention(num_heads, head_dim)
        self.ffn = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Residual connection around the multi-head attention and normalization.
        x = x + self.mha(self.norm1(x))
        # Residual connection around the feed-forward network and normalization.
        x = x + self.ffn(self.norm2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    GPT-style transformer language model.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # Token and positional embedding layers.
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        # Stack multiple transformer blocks.
        self.blocks = nn.Sequential(*[TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        # Linear layer to map transformer outputs to vocabulary logits.
        self.head = nn.Linear(EMBED_DIM, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx (torch.Tensor): Input indices with shape (B, T).
            targets (torch.Tensor, optional): Target indices for computing loss.
        
        Returns:
            tuple: (logits, loss) where logits are raw predictions and loss is computed via cross-entropy (if targets provided).
        """
        B, T = idx.shape
        # Get token embeddings.
        token_emb = self.token_embedding(idx)  # Shape: (B, T, EMBED_DIM)
        # Create positional indices and get positional embeddings.
        pos_indices = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos_indices)  # Shape: (T, EMBED_DIM)
        x = token_emb + pos_emb
        # Pass through transformer blocks.
        x = self.blocks(x)
        x = self.final_norm(x)
        # Compute logits over the vocabulary.
        logits = self.head(x)
        
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss computation.
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generates tokens from the model.
        
        Args:
            idx (torch.Tensor): Initial input indices (B, T).
            max_new_tokens (int): Number of tokens to generate.
        
        Returns:
            torch.Tensor: Extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            # Crop context to the most recent BLOCK_SIZE tokens.
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            # Focus only on the logits of the last token.
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution.
            next_token = torch.multinomial(probs, num_samples=1)
            # Append the new token to the sequence.
            idx = torch.cat([idx, next_token], dim=1)
        return idx
