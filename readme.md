
# MoE Transformer - Mixture of Experts Implementation

A clean, educational implementation of a Mixture of Experts (MoE) Transformer for character-level language modeling, based on the Switch Transformer architecture.

A Mixture of Experts (MoE) replaces dense feed-forward (FFN) blocks with many small FFNs called experts, and a learned router/gating network that sends each token (or example) to only a few experts. That gives huge parameter capacity while keeping per-token compute low — so you can pretrain enormous models quickly — but it adds engineering complexity: memory to store all experts, routing logic, load-balancing, and cross-device communication.

What is a Mixture of Experts?
Think of an MoE layer like a call center with many specialists (experts). For each incoming “call” (an input token’s hidden vector), the router decides which specialists to forward the call to. Only those chosen experts compute and respond; their outputs are combined (often weighted) to produce the layer’s output.

<img width="881" height="425" alt="image" src="https://github.com/user-attachments/assets/b022b257-a915-42a3-addd-68d65d29db9e" />

## Features

- 🧠 **Mixture of Experts (MoE)**: Top-1 routing with load balancing
- 🔤 **Character-level modeling**: Trained on Shakespeare text
- ⚡ **Efficient training**: Mixed precision, gradient accumulation
- 📊 **Comprehensive logging**: Loss tracking and model analysis
- 🎯 **Modular design**: Easy to understand and modify
- 🔧 **Configurable**: Dense vs MoE comparison built-in

## Architecture

The model implements a standard Transformer architecture with optional MoE layers:

- **Base Transformer**: Multi-head attention + MLP blocks
- **MoE Enhancement**: Replace every other MLP with mixture of experts  
- **Switch Routing**: Top-1 token routing with capacity constraints
- **Load Balancing**: Auxiliary loss to encourage expert utilization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```python
python main.py
```

This will:
- Download tiny Shakespeare dataset
- Train the MoE model for 400 steps
- Generate sample text
- Optionally compare MoE vs Dense models

### 3. Basic Usage

```python
from config import Config
from data_utils import DataLoader
from model import TinyGPT
from trainer import Trainer

# Setup
config = Config()
data_loader = DataLoader(config)
model = TinyGPT(config, data_loader.vocab_size)
trainer = Trainer(model, config, data_loader)

# Train
trainer.train()

# Generate
text = trainer.generate_text("JULIET: ", max_new_tokens=100)
print(text)
```

## Configuration

Key hyperparameters in `config.py`:

```python
# Model architecture
use_moe = True          # Enable MoE layers
n_layer = 6            # Number of transformer blocks  
n_head = 8             # Number of attention heads
d_model = 512          # Model dimension

# MoE settings
n_experts = 4          # Number of experts per MoE layer
top_k = 1              # Number of experts to route to
capacity_factor = 1.25  # Per-expert capacity limit
load_balance_coef = 0.01  # Load balancing loss weight
```

## File Structure

```
├── config.py           # Configuration settings
├── data_utils.py       # Data loading and processing  
├── layers.py           # Core neural network layers
├── moe_layers.py       # Mixture of experts components
├── model.py            # Main transformer model
├── trainer.py          # Training loop and utilities
├── utils.py            # Helper functions
├── main.py             # Example usage script
├── __init__.py         # Package initialization
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Key Components

### MoE Router (`moe_layers.py`)
- **Top1Router**: Routes each token to single expert
- **Load Balancing**: Ensures even expert utilization  
- **Capacity Limits**: Handles token overflow gracefully

### Expert Networks (`moe_layers.py`)
- **ExpertMLP**: Individual feed-forward expert
- **MoEMLP**: Coordinates routing and expert execution
- **DenseMLP**: Standard MLP for baseline comparison

### Training Features (`trainer.py`)
- **Mixed Precision**: FP16/BF16 support for efficiency
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Cosine Scheduling**: Learning rate warmup and decay
- **Loss Estimation**: Regular validation evaluation

## Model Comparison

The implementation supports easy comparison between MoE and dense models:

```python
# In main.py
compare_moe_vs_dense()  # Trains both models and compares performance
```

Typical results:
- **MoE**: Better loss with same parameter count
- **Dense**: Simpler but potentially less expressive
- **Trade-offs**: MoE complexity vs performance gains

## Advanced Usage

### Custom Expert Configuration

```python
config = Config()
config.n_experts = 8           # More experts
config.capacity_factor = 2.0   # Higher capacity  
config.load_balance_coef = 0.1 # Stronger load balancing
```

### Save/Load Models

```python
from utils import save_model, load_model

# Save
save_model(model, config, "checkpoints/model.pt")

# Load  
load_model(model, "checkpoints/model.pt")
```

### Generate with Different Parameters

```python
# More creative
text = trainer.generate_text("ROMEO: ", temperature=1.2, top_k=20)

# More conservative  
text = trainer.generate_text("ROMEO: ", temperature=0.5, top_k=5)
```

## Implementation Details

### Switch Transformer Features
- **Top-1 routing**: Each token goes to single expert
- **Capacity factor**: Limits tokens per expert (1.25x average)  
- **Load balancing loss**: Encourages uniform expert usage
- **Z-loss**: Regularizes router confidence

### Efficiency Optimizations
- **Gradient accumulation**: Simulate larger batches
- **Mixed precision**: Reduce memory usage  
- **Causal masking**: Efficient attention computation
- **Parameter sharing**: Router weights shared across layers

## Extending the Code

The modular design makes it easy to:

1. **Add new expert types**: Modify `ExpertMLP` class
2. **Implement different routing**: Create new router classes  
3. **Change architectures**: Modify `TransformerBlock`
4. **Add new datasets**: Extend `DataLoader` class

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` or `d_model` if OOM
2. **Training Speed**: Enable `compile_model` for PyTorch 2.0+
3. **Expert Utilization**: Monitor auxiliary losses during training
4. **Generation Quality**: Experiment with `temperature` and `top_k`

## References

- [Switch Transformer Paper](https://arxiv.org/abs/2101.03961)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)

## License

MIT License - see LICENSE file for details.
