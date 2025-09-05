import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return self.weight * (x / (norm + self.eps))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer."""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Apply scaled dot-product attention with causal masking
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            attn_mask=None,
            is_causal=True
        )
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class ExpertMLP(nn.Module):
    """Single expert MLP layer."""
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
