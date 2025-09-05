import torch
import json
import os
from typing import Dict, Any

def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model: torch.nn.Module, config: Any, path: str):
    """Save model state dict and config to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model: torch.nn.Module, path: str):
    """Load model state dict from file."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return checkpoint.get('config', None)

def save_config(config: Any, path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    config_dict = vars(config) if hasattr(config, '__dict__') else config
    
    # Convert torch dtypes to strings for JSON serialization
    json_config = {}
    for k, v in config_dict.items():
        if isinstance(v, torch.dtype):
            json_config[k] = str(v)
        else:
            json_config[k] = v
    
    with open(path, 'w') as f:
        json.dump(json_config, f, indent=2)
    print(f"Config saved to {path}")

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert string dtypes back to torch dtypes
    if 'dtype' in config_dict:
        if config_dict['dtype'] == 'torch.float16':
            config_dict['dtype'] = torch.float16
        elif config_dict['dtype'] == 'torch.bfloat16':
            config_dict['dtype'] = torch.bfloat16
        elif config_dict['dtype'] == 'torch.float32':
            config_dict['dtype'] = torch.float32
    
    print(f"Config loaded from {path}")
    return config_dict

def get_model_info(model: torch.nn.Module, config: Any) -> Dict[str, Any]:
    """Get comprehensive model information."""
    info = {
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_trainable_parameters(model),
        'model_size_mb': count_parameters(model) * 4 / 1024 / 1024,  # Assuming float32
        'architecture': {
            'n_layers': config.n_layer,
            'n_heads': config.n_head,
            'd_model': config.d_model,
            'd_mlp': config.d_mlp,
            'block_size': config.block_size,
        }
    }
    
    if hasattr(config, 'use_moe') and config.use_moe:
        info['moe'] = {
            'enabled': True,
            'n_experts': config.n_experts,
            'top_k': config.top_k,
            'capacity_factor': config.capacity_factor,
            'load_balance_coef': config.load_balance_coef,
            'zloss_coef': config.zloss_coef,
        }
        
        # Estimate active parameters (approximation)
        moe_layers = sum(1 for i in range(config.n_layer) if i % 2 == 1)
        expert_params_per_layer = config.d_model * config.d_mlp * 2  # fc1 + fc2
        active_expert_params = moe_layers * expert_params_per_layer
        inactive_expert_params = moe_layers * expert_params_per_layer * (config.n_experts - 1)
        
        info['moe']['active_parameters'] = info['total_parameters'] - inactive_expert_params
        info['moe']['moe_layers'] = moe_layers
    else:
        info['moe'] = {'enabled': False}
    
    return info

def print_model_info(model: torch.nn.Module, config: Any):
    """Print comprehensive model information."""
    info = get_model_info(model, config)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.2f}M)")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Model Size: {info['model_size_mb']:.2f} MB")
    
    print("\nArchitecture:")
    for k, v in info['architecture'].items():
        print(f"  {k}: {v}")
    
    print(f"\nMoE Configuration:")
    if info['moe']['enabled']:
        print(f"  Enabled: Yes")
        print(f"  Number of Experts: {info['moe']['n_experts']}")
        print(f"  Top-K: {info['moe']['top_k']}")
        print(f"  MoE Layers: {info['moe']['moe_layers']}")
        print(f"  Active Parameters: {info['moe']['active_parameters']:,} ({info['moe']['active_parameters']/1e6:.2f}M)")
        print(f"  Capacity Factor: {info['moe']['capacity_factor']}")
        print(f"  Load Balance Coef: {info['moe']['load_balance_coef']}")
        print(f"  Z-Loss Coef: {info['moe']['zloss_coef']}")
    else:
        print(f"  Enabled: No (Dense baseline)")
    
    print("=" * 50)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_device_info() -> Dict[str, Any]:
    """Get device and CUDA information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,   # GB
            'bf16_supported': torch.cuda.is_bf16_supported(),
        })
    
    return info

def print_device_info():
    """Print device and CUDA information."""
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    
    if info['cuda_available']:
        print(f"CUDA Available: Yes")
        print(f"Device Count: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Memory Allocated: {info['memory_allocated']:.2f} GB")
        print(f"Memory Reserved: {info['memory_reserved']:.2f} GB")
        print(f"BFloat16 Supported: {info['bf16_supported']}")
    else:
        print("CUDA Available: No")
        print("Using CPU")
    
    print("=" * 50)
