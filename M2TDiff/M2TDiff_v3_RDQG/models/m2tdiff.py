import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .rdqg import RDQG
from .mgte import MGTE
from .smtd import SMTD

class M2TDiff(nn.Module):
    def __init__(self,num_classes=30,num_queries=100,hidden_dim=256,num_frames=5,
                 diffusion_steps=4,trajectories=5,graph_k=(1,10),
                 num_experts=4,decoder_layers=6,**kwargs):
        super().__init__()
        self.num_queries=num_queries
        self.num_classes=num_classes
        self.backbone=ResNetBackbone()
        self.rdqg=RDQG(2048,hidden_dim,diffusion_steps,trajectories,num_classes)
        self.mgte=MGTE(2048,hidden_dim,graph_k)
        self.smtd=SMTD(hidden_dim,decoder_layers,num_experts)
        self.class_head=nn.Linear(hidden_dim,num_classes+1)
        self.box_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,4))

    def forward(self,frames,boxes=None,targets=None):
        B,T,_,_,_=frames.shape
        f=self.backbone(frames.flatten(0,1))
        fs=f.view(B,T,*f.shape[1:])
        center=fs[:,T//2]
        z,aux=self.rdqg(center,targets=targets,num_queries=self.num_queries)
        memory=self.mgte(fs)
        dec=self.smtd(z,memory)
        out={"pred_logits":self.class_head(dec),"pred_boxes":self.box_head(dec).sigmoid()}
        out.update(aux)
        return out
