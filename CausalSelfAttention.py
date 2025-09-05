class CausalSelfAttention(nn.Module):
  def __init__(self,d_model,n_head,dropout = 0.0):
    super().__init__()
    assert d_model % n_head == 0
    self.n_head = n_head
    self.head_dim = d_model // n_head
    self.qkv = nn.Linear(d_model , 3*d_model,bias = False)
    self.proj = nn.Linear(d_model,d_model,bias = False)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    qkv = self.qkv(x).view(B,T,3,self.n_head,self.head_dim).transpose(1,2)
    q,k,v = qkv[:,0],qkv[:,1],qkv[:,2]
    y = F.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask= None,
    )
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.proj(y)
    return y
