class cfg:
  use_moe = True # set False for dense baseline
  n_layer = 6
  n_head = 8
  d_model = 512
  d_mlp = 2048  # dense MLP hidden
  vocab_limit = None # None = use all chars
  block_size = 256 #sequence length
  batch_size = 24 #tokens per batch = batch_size * block_size
  grad_accum_steps = 2 #effective batch = batch_size * grad_accum_steps
  max_steps = 400 #quick demo ; increase for better loss
  lr = 3e-4
  weight_decay = 0.1
  warmup_steps = 0.1
  compile_model = False #torch.compile may slow first step
  dropout = 0.0

  #MOE specifics
  n_experts = 4
  top_k = 1 #switch-style
  capacity_factor = 1.25 #per-expert token capacity
  load_balance_coef = 0.01
  zloss_coef = 0.001

  #precision + device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
  seed = 42
