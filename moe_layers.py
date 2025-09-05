import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .layers import ExpertMLP

class Top1Router(nn.Module):
    """Top-1 router for MoE (Switch Transformer style)."""
    
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts)
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, T, D]
            
        Returns:
            top1: Expert assignments [B*T]
            w: Router weights [B*T, 1]
            lb_loss: Load balancing loss
            z_loss: Z loss (router z-loss)
        """
        B, T, D = x.shape
        h = x.reshape(B * T, D)
        
        # Compute router logits and probabilities
        logits = self.proj(h)
        probs = F.softmax(logits, dim=-1)
        
        # Top-1 routing
        top1 = probs.argmax(dim=-1)
        w = probs.gather(1, top1.unsqueeze(-1))
        
        # Load balance loss computation
        with torch.no_grad():
            assign = F.one_hot(top1, num_classes=self.n_experts).float()
        
        importance = probs.mean(dim=0)  # Average probability mass per expert
        load = assign.mean(dim=0)       # Fraction of tokens assigned to each expert
        lb_loss = self.n_experts * torch.sum(importance * load)
        
        # Z loss (encourages router to be confident)
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        
        return top1, w, lb_loss, z_loss


class MoEMLP(nn.Module):
    """Mixture of Experts MLP layer with top-1 routing."""
    
    def __init__(
        self, 
        d_model: int, 
        d_hidden: int, 
        n_experts: int, 
        capacity_factor: float = 1.25,
        lbl_coef: float = 0.01, 
        zloss_coef: float = 0.0, 
        top_k: int = 1
    ):
        super().__init__()
        assert top_k == 1, "This implementation supports only top-1 routing (Switch-style)."
        
        self.router = Top1Router(d_model, n_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, d_hidden) for _ in range(n_experts)
        ])
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.lbl_coef = lbl_coef
        self.zloss_coef = zloss_coef

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            y: Output tensor [B, T, D]
            aux: Auxiliary loss (load balancing + z loss)
        """
        B, T, D = x.shape
        N = B * T
        
        # Route tokens to experts
        top1, w, lb_loss, z_loss = self.router(x)
        
        # Calculate per-expert capacity
        cap = int(self.capacity_factor * (N / self.n_experts) + 1)
        
        # Flatten input for expert processing
        flat_x = x.reshape(N, D)
        out = torch.zeros_like(flat_x)

        # Process tokens for each expert
        for e in range(self.n_experts):
            # Get indices of tokens routed to this expert
            idx = (top1 == e).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
                
            # Apply capacity constraint (drop overflow tokens)
            if idx.numel() > cap:
                idx = idx[:cap]
            
            # Process tokens through expert
            xe = flat_x[idx]
            ye = self.experts[e](xe)
            
            # Apply router weights and accumulate output
            we = w[idx].reshape(-1, 1)
            out[idx] = ye * we

        # Reshape output and compute auxiliary loss
        y = out.reshape(B, T, D)
        aux = self.lbl_coef * lb_loss + self.zloss_coef * z_loss
        
        return y, aux


class DenseMLP(nn.Module):
    """Dense MLP layer (baseline without MoE)."""
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.ff = ExpertMLP(d_model, d_hidden)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            y: Output tensor [B, T, D]  
            aux: Zero auxiliary loss for compatibility
        """
        y = self.ff(x)
        aux = x.new_tensor(0.0)
        return y, aux
