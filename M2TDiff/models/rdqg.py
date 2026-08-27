import torch
import torch.nn as nn
from torchvision.ops import roi_align

class IterativeOptimizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, z):
        a,_ = self.attn(z,z,z)
        return z + a + self.ffn(z+a)

class RDQG(nn.Module):
    # Paper-aligned interface: noisy boxes -> RoIAlign queries -> iterative denoising.
    def __init__(self, dim=256, steps=4, trajectories=4):
        super().__init__()
        self.steps, self.trajectories = steps, trajectories
        self.proj = nn.Conv2d(2048, dim, 1)
        self.optimizer = IterativeOptimizer(dim)

    def forward(self, feat, boxes):
        # boxes: [B,Q,4] normalized xyxy. During training caller supplies noisy boxes.
        B,Q,_ = boxes.shape
        f = self.proj(feat)
        H,W = f.shape[-2:]
        rois=[]
        for b in range(B):
            bb=boxes[b].clone()
            bb[:,[0,2]] *= W; bb[:,[1,3]] *= H
            rois.append(torch.cat([torch.full((Q,1), b, device=feat.device), bb],1))
        rois=torch.cat(rois,0)
        z=roi_align(f, rois, output_size=1, spatial_scale=1.0, aligned=True).flatten(1).view(B,Q,-1)
        for _ in range(self.steps):
            z=self.optimizer(z)
        return z
