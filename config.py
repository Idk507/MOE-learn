import torch

class Config:
    """Configuration class for MoE Transformer model."""
    
    def __init__(self):
        # Model architecture
        self.use_moe = True  # set False for dense baseline
        self.n_layer = 6
        self.n_head = 8
        self.d_model = 512
        self.d_mlp = 2048  # dense MLP hidden
        self.vocab_limit = None  # None = use all chars
        self.block_size = 256  # sequence length
        
        # Training parameters
        self.batch_size = 24  # tokens per batch = batch_size * block_size
        self.grad_accum_steps = 2  # effective batch = batch_size * grad_accum_steps
        self.max_steps = 400  # quick demo; increase for better loss
        self.lr = 3e-4
        self.weight_decay = 0.1
        self.warmup_steps = 0.1
        self.compile_model = False  # torch.compile may slow first step
        self.dropout = 0.0
        
        # MoE specifics
        self.n_experts = 4
        self.top_k = 1  # switch-style
        self.capacity_factor = 1.25  # per-expert token capacity
        self.load_balance_coef = 0.01
        self.zloss_coef = 0.001
        
        # Precision + device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.seed = 42
        
    def __repr__(self):
        """String representation of config."""
        config_str = "Config(\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}={value},\n"
        config_str += ")"
        return config_str
