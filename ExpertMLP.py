class ExpertMLP(nn.Module):
  def __init__(self, d_model, d_hidden):
    super().__init__() # Add this line
    self.fc1 = nn.Linear(d_model,d_hidden)
    self.fc2 = nn.Linear(d_hidden,d_model)

  def forward(self,x):
    return self.fc2(F.gelu(self.fc1(x)))

