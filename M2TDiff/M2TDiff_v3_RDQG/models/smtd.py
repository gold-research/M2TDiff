import torch
import torch.nn as nn

class QueryAwareMoE(nn.Module):
    def __init__(self, dim, experts=4):
        super().__init__()
        self.gate=nn.Linear(dim,experts)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim)) for _ in range(experts)])
    def forward(self,x):
        logits=self.gate(x)
        ids=logits.argmax(-1)
        out=torch.zeros_like(x)
        for i,e in enumerate(self.experts):
            mask=ids==i
            if mask.any(): out[mask]=e(x[mask])
        return x+out

class SMTDLayer(nn.Module):
    def __init__(self, dim, experts):
        super().__init__()
        self.self_attn=nn.MultiheadAttention(dim,8,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(dim,8,batch_first=True)
        self.moe=QueryAwareMoE(dim,experts)
        self.n1,self.n2=nn.LayerNorm(dim),nn.LayerNorm(dim)
    def forward(self,z,memory):
        x,_=self.self_attn(z,z,z); z=self.n1(z+x)
        x,_=self.cross_attn(z,memory,memory); z=self.n2(z+x)
        return self.moe(z)

class SMTD(nn.Module):
    def __init__(self,dim=256,layers=6,experts=4):
        super().__init__(); self.layers=nn.ModuleList([SMTDLayer(dim,experts) for _ in range(layers)])
    def forward(self,z,memory):
        for l in self.layers: z=l(z,memory)
        return z
