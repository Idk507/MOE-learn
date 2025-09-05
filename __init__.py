"""
MoE Transformer - A Mixture of Experts implementation for character-level language modeling.

This package provides a complete implementation of a Transformer model with optional
Mixture of Experts (MoE) layers, following the Switch Transformer architecture.
"""

from .config import Config
from .data_utils import DataLoader
from .layers import RMSNorm, CausalSelfAttention, ExpertMLP
from .moe_layers import Top1Router, MoEMLP, DenseMLP
from .model import TransformerBlock, TinyGPT
from .trainer import Trainer
from .utils import (
    count_parameters,
    count_trainable_parameters,
    save_model,
    load_model,
    save_config,
    load_config,
    get_model_info,
    print_model_info,
    set_seed,
    get_device_info,
    print_device_info
)

__version__ = "0.1.0"
__author__ = "MoE Transformer Implementation"

__all__ = [
    # Core classes
    "Config",
    "DataLoader", 
    "TinyGPT",
    "Trainer",
    
    # Layer components
    "RMSNorm",
    "CausalSelfAttention",
    "ExpertMLP",
    "TransformerBlock",
    
    # MoE components
    "Top1Router",
    "MoEMLP", 
    "DenseMLP",
    
    # Utilities
    "count_parameters",
    "count_trainable_parameters",
    "save_model",
    "load_model", 
    "save_config",
    "load_config",
    "get_model_info",
    "print_model_info",
    "set_seed",
    "get_device_info",
    "print_device_info",
]
