class RMSNorm(nn.Module):
  def __init__(self, d, eps = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(d))
    self.eps = eps

  def forward(self,x):
    norm = x.norm(dim = -1 ,keepdim = True)*(1.0/ math.sqrt(x.shape[-1]))
    return self.weight * (x/ (norm + self.eps))
