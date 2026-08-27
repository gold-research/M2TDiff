import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConv(nn.Module):
    def __init__(self, dim, k):
        super().__init__(); self.k=k; self.lin=nn.Linear(dim,dim)
    def forward(self,x):
        # x [B,N,C], dynamic cosine kNN graph
        xn=F.normalize(x,dim=-1)
        sim=xn@xn.transpose(-1,-2)
        idx=sim.topk(min(self.k,x.shape[1]),dim=-1).indices
        nbr=torch.gather(x.unsqueeze(1).expand(-1,x.shape[1],-1,-1),2,
                         idx.unsqueeze(-1).expand(-1,-1,-1,x.shape[-1]))
        return self.lin(nbr.mean(2))

class MGTE(nn.Module):
    def __init__(self, in_dim=2048, dim=256, ks=(8,16)):
        super().__init__()
        self.proj=nn.Conv2d(in_dim,dim,1)
        self.attn=nn.MultiheadAttention(dim,8,batch_first=True)
        self.graphs=nn.ModuleList([DynamicGraphConv(dim,k) for k in ks])
        self.norm=nn.LayerNorm(dim)
    def forward(self, feats):
        # feats [B,T,C,H,W]
        B,T,C,H,W=feats.shape
        x=self.proj(feats.flatten(0,1)).flatten(2).transpose(1,2).reshape(B,T*H*W,-1)
        a,_=self.attn(x,x,x)
        gs=[g(x) for g in self.graphs]
        return self.norm(a + sum(gs)/len(gs))
