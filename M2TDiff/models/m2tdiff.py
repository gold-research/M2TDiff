import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .rdqg import RDQG
from .mgte import MGTE
from .smtd import SMTD

class M2TDiff(nn.Module):
    def __init__(self,num_classes=30,num_queries=300,hidden_dim=256,num_frames=4,diffusion_steps=4,trajectories=4,graph_k=(8,16),num_experts=4,decoder_layers=6):
        super().__init__()
        self.num_queries=num_queries
        self.backbone=ResNetBackbone()
        self.rdqg=RDQG(hidden_dim,diffusion_steps,trajectories)
        self.mgte=MGTE(2048,hidden_dim,graph_k)
        self.smtd=SMTD(hidden_dim,decoder_layers,num_experts)
        self.class_head=nn.Linear(hidden_dim,num_classes+1)
        self.box_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,4))
    def make_noisy_boxes(self,B,device):
        q=torch.rand(B,self.num_queries,4,device=device)
        q[...,2:]=torch.maximum(q[...,:2]+0.05,q[...,2:])
        return q.clamp(0,1)
    def forward(self,frames,boxes=None):
        # frames [B,T,3,H,W]
        B,T,C,H,W=frames.shape
        f=self.backbone(frames.flatten(0,1))
        fs=f.view(B,T,*f.shape[1:])
        if boxes is None: boxes=self.make_noisy_boxes(B,frames.device)
        z=self.rdqg(f.view(B*T,*f.shape[1:])[:B],boxes)
        memory=self.mgte(fs)
        dec=self.smtd(z,memory)
        return {"pred_logits":self.class_head(dec),"pred_boxes":self.box_head(dec).sigmoid()}
