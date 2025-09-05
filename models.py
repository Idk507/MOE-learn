import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .layers import RMSNorm, CausalSelfAttention
from .moe_layers import MoEMLP, DenseMLP

class TransformerBlock(nn.Module):
    """Single transformer block with optional MoE."""
    
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_mlp: int, 
        use_moe: bool, 
        n_experts: int, 
        capacity_factor: float, 
        lbl_coef: float, 
        zloss_coef: float,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = RMSNorm(d_model)
        
        if use_moe:
            self.mlp = MoEMLP(
                d_model, d_mlp, n_experts, 
                capacity_factor, lbl_coef, zloss_coef, top_k=1
            )
        else:
            self.mlp = DenseMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            x: Output tensor [B, T, D]
            aux: Auxiliary loss from MoE layer
        """
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        
        # MLP with residual connection
        y, aux = self.mlp(self.ln2(x))
        x = x + y
        
        return x, aux


class TinyGPT(nn.Module):
    """Small GPT model with optional Mixture of Experts."""
    
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.d_model))
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        blocks = []
        for i in range(config.n_layer):
            # Use MoE every other block (starting from block 1)
            use_moe_this = config.use_moe and (i % 2 == 1)
            blocks.append(TransformerBlock(
                config.d_model, 
                config.n_head, 
                config.d_mlp, 
                use_moe_this,
                config.n_experts, 
                config.capacity_factor, 
                config.load_balance_coef, 
                config.zloss_coef,
                config.dropout
            ))
        self.blocks = nn.ModuleList(blocks)
        
        # Final layer norm and output projection
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional)
            
        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            aux_total: Total auxiliary loss from MoE layers
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"
        
        # Token and position embeddings
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)

        # Forward through transformer blocks
        aux_total = x.new_tensor(0.0)
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss, aux_total

    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int = 200, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (optional)
            
        Returns:
            Generated token sequence [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop sequence to block_size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                logits, _, _ = self.forward(idx_cond)
                logits = logits[:, -1, :] / max(1e-6, temperature)
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, next_id), dim=1)
        
        return idx
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_active_parameters(self) -> int:
        """Count parameters that are active during inference (approximation for MoE)."""
        total = 0
        moe_expert_params = 0
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                total += sum(p.numel() for p in module.parameters())
            elif hasattr(module, 'experts'):  # MoE layer
                # Count only one expert's parameters (since only one is active)
                expert_params = sum(p.numel() for p in module.experts[0].parameters())
                router_params = sum(p.numel() for p in module.router.parameters())
                moe_expert_params += expert_params + router_params
                
        # Subtract inactive expert parameters
        if hasattr(self.config, 'n_experts') and self.config.use_moe:
            inactive_experts = self.config.n_experts - 1
            moe_layers = sum(1 for i in range(self.config.n_layer) if i % 2 == 1)
            expert_size = moe_expert_params // moe_layers if moe_layers > 0 else 0
            total -= inactive_experts * expert_size * moe_layers
            
        return total
