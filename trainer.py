import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from .model import TinyGPT
from .data_utils import DataLoader

class Trainer:
    """Training utilities for the MoE Transformer."""
    
    def __init__(self, model: TinyGPT, config, data_loader: DataLoader):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        
        # Initialize optimizer and scaler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(config.dtype == torch.float16)
        )
        
        # Determine model dtype
        self.model_dtype = torch.bfloat16 if config.dtype == torch.bfloat16 else torch.float16
        
        # Move model to device and set dtype
        self.model = self.model.to(device=config.device, dtype=self.model_dtype)
        
        # Compile model if requested
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
    
    def cosine_lr_schedule(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        warmup_steps = int(self.config.warmup_steps * self.config.max_steps)
        
        if step < warmup_steps:
            return self.config.lr * (step + 1) / warmup_steps
        
        progress = (step - warmup_steps) / max(1, (self.config.max_steps - warmup_steps))
        return self.config.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def update_lr(self, step: int):
        """Update learning rate based on schedule."""
        lr = self.cosine_lr_schedule(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    @torch.no_grad()
    def estimate_loss(self, eval_iters: int = 20) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.model.eval()
        losses = {}
        auxes = {}
        
        for split in ["train", "val"]:
            total_loss = 0.0
            total_aux = 0.0
            
            for _ in range(eval_iters):
                xb, yb = self.data_loader.get_batch(split)
                
                with torch.autocast(
                    device_type="cuda" if self.config.device == "cuda" else "cpu", 
                    dtype=self.model_dtype
                ):
                    _, loss, aux = self.model(xb, yb)
                
                total_loss += loss.item()
                total_aux += aux.item()
            
            losses[split] = total_loss / eval_iters
            auxes[f"aux_{split}"] = total_aux / eval_iters
        
        self.model.train()
        return {**losses, **auxes}
    
    def train_step(self, step: int) -> Dict[str, float]:
        """Execute a single training step."""
        # Update learning rate
        lr = self.update_lr(step)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_aux = 0.0
        
        # Gradient accumulation
        for micro_step in range(self.config.grad_accum_steps):
            xb, yb = self.data_loader.get_batch("train")
            
            with torch.autocast(
                device_type="cuda" if self.config.device == "cuda" else "cpu", 
                dtype=self.model_dtype
            ):
                logits, loss, aux = self.model(xb, yb)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.grad_accum_steps
                aux = aux / self.config.grad_accum_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss + aux).backward()
            
            total_loss += loss.item()
            total_aux += aux.item()
        
        # Optimizer step with gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'loss': total_loss,
            'aux': total_aux,
            'lr': lr
        }
    
    def train(self, eval_interval: int = 100, log_interval: int = 50) -> Dict[str, list]:
        """Full training loop."""
        print("Starting training...")
        print(f"Total parameters: {self.model.count_parameters()/1e6:.2f}M")
        print(f"Using MoE: {self.config.use_moe}, n_experts={self.config.n_experts}")
        print(f"Device: {self.config.device}, dtype: {self.model_dtype}")
        
        self.model.train()
        
        # Training metrics
        train_losses = []
        val_losses = []
        aux_losses = []
        learning_rates = []
        
        for step in range(self.config.max_steps):
            # Training step
            step_metrics = self.train_step(step)
            
            # Log progress
            if step % log_interval == 0 or step == self.config.max_steps - 1:
                print(f"Step {step}/{self.config.max_steps}: "
                      f"loss={step_metrics['loss']:.4f}, "
                      f"aux={step_metrics['aux']:.4f}, "
                      f"lr={step_metrics['lr']:.2e}")
            
            # Evaluate model
            if step % eval_interval == 0 or step == self.config.max_steps - 1:
                eval_metrics = self.estimate_loss()
                print(f"Step {step} eval: "
                      f"train_loss={eval_metrics['train']:.4f}, "
                      f"val_loss={eval_metrics['val']:.4f}")
                
                train_losses.append(eval_metrics['train'])
                val_losses.append(eval_metrics['val'])
                aux_losses.append(eval_metrics['aux_train'])
                learning_rates.append(step_metrics['lr'])
        
        print("Training completed!")
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'aux_losses': aux_losses,
            'learning_rates': learning_rates
        }
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        temperature: float = 0.8, 
        top_k: Optional[int] = 50
    ) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        
        # Encode prompt
        context = torch.tensor(
            [[self.data_loader.stoi.get(c, 0) for c in prompt]], 
            dtype=torch.long, 
            device=self.config.device
        )
        
        # Generate tokens
        with torch.no_grad():
            generated = self.model.generate(
                context, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )
        
        # Decode to text
        generated_text = self.data_loader.decode(generated[0])
        
        self.model.train()
        return generated_text
