import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from .utils.box_ops import box_cxcywh_to_xyxy

def sinusoidal_timestep_embedding(t, dim):
    half=dim//2
    freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1))
    x=t.float()[:,None]*freq[None]
    return torch.cat([x.sin(),x.cos()],-1)

class DynamicInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.param=nn.Linear(dim,dim*2)
        self.region_proj=nn.Linear(dim,dim)
        self.out=nn.Linear(dim,dim)
    def forward(self, query, region):
        scale,bias=self.param(query).chunk(2,-1)
        interaction=torch.sigmoid(scale)*self.region_proj(region)+bias
        return self.out(interaction)

class IterativeOptimizer(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norm1=nn.LayerNorm(dim)
        self.dynamic=DynamicInteraction(dim)
        self.norm2=nn.LayerNorm(dim)
        self.time_mlp=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.box_delta=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,4))
        self.cls_head=nn.Linear(dim,31)  # overwritten by RDQG if class count differs

    def forward(self,z,boxes,feat,t,num_classes):
        a,_=self.attn(z,z,z)
        z=self.norm1(z+a)
        B,Q,_=boxes.shape; H,W=feat.shape[-2:]
        xyxy=box_cxcywh_to_xyxy(boxes).clamp(0,1)
        rois=[]
        for b in range(B):
            bb=xyxy[b].clone()
            bb[:,[0,2]]*=W; bb[:,[1,3]]*=H
            rois.append(torch.cat([torch.full((Q,1),b,device=z.device),bb],-1))
        rois=torch.cat(rois)
        region=roi_align(feat,rois,output_size=1,spatial_scale=1.0,aligned=True).flatten(1).view(B,Q,-1)
        z=self.norm2(z+self.dynamic(z,region))
        te=sinusoidal_timestep_embedding(t, z.shape[-1])
        z=z+self.time_mlp(te).unsqueeze(1)
        box=(boxes+self.box_delta(z).tanh()*0.05).clamp(0,1)
        cls=self.cls_head(z)[...,:num_classes+1]
        return z,box,cls

class RDQG(nn.Module):
    """
    RDQG implementation following the paper-level pipeline:
    forward Gaussian box corruption -> RoIAlign initialization ->
    stochastic multi-trajectory iterative reverse optimization.
    """
    def __init__(self,in_dim=2048,dim=256,steps=4,trajectories=5,num_classes=30):
        super().__init__()
        self.steps,self.trajectories=steps,trajectories
        self.num_classes=num_classes
        self.proj=nn.Conv2d(in_dim,dim,1)
        self.optimizer=IterativeOptimizer(dim)
        self.optimizer.cls_head=nn.Linear(dim,num_classes+1)
        beta=torch.linspace(1e-4,0.02,steps)
        alpha=1-beta
        alpha_bar=torch.cumprod(alpha,0)
        self.register_buffer("alpha_bar",alpha_bar)

    def corrupt_boxes(self, gt_boxes, t):
        # DDPM-style box corruption.
        noise=torch.randn_like(gt_boxes)
        a=self.alpha_bar[t].view(-1,1,1).sqrt()
        b=(1-self.alpha_bar[t]).view(-1,1,1).sqrt()
        return (a*gt_boxes+b*noise).clamp(0,1)

    def init_queries(self, feat, boxes):
        B,Q,_=boxes.shape
        H,W=feat.shape[-2:]
        xyxy=box_cxcywh_to_xyxy(boxes).clamp(0,1)
        rois=[]
        for b in range(B):
            bb=xyxy[b].clone()
            bb[:,[0,2]]*=W; bb[:,[1,3]]*=H
            rois.append(torch.cat([torch.full((Q,1),b,device=feat.device),bb],1))
        rois=torch.cat(rois)
        return roi_align(feat,rois,output_size=1,spatial_scale=1.0,aligned=True).flatten(1).view(B,Q,-1)

    def run_trajectory(self, feat, start_boxes, stochastic=True):
        B=start_boxes.shape[0]
        z=self.init_queries(feat,start_boxes)
        boxes=start_boxes
        logprob=z.new_zeros(B)
        cls=None
        for step in reversed(range(self.steps)):
            t=torch.full((B,),step,device=z.device,dtype=torch.long)
            z,mean_boxes,cls=self.optimizer(z,boxes,feat,t,self.num_classes)
            if stochastic and step>0:
                sigma=(1-self.alpha_bar[step]).sqrt()*0.05
                noise=torch.randn_like(mean_boxes)
                boxes=(mean_boxes+sigma*noise).clamp(0,1)
                # Gaussian proxy log-prob for trajectory policy signal.
                logprob=logprob-0.5*(noise.flatten(1).pow(2).mean(1))
            else:
                boxes=mean_boxes
        return {"queries":z,"boxes":boxes,"logits":cls,"logprob":logprob}

    def _pad_gt(self, targets, num_queries, device):
        boxes=[]
        for t in targets:
            b=t["boxes"].to(device)
            if len(b)==0: b=torch.rand(1,4,device=device)
            if len(b)<num_queries:
                ids=torch.randint(len(b),(num_queries-len(b),),device=device)
                b=torch.cat([b,b[ids]],0)
            else:
                ids=torch.randperm(len(b),device=device)[:num_queries]
                b=b[ids]
            boxes.append(b)
        return torch.stack(boxes)

    @torch.no_grad()
    def reward(self, logits, boxes, targets):
        prob=logits.softmax(-1)[...,:self.num_classes].max(-1).values
        rewards=[]
        from .utils.box_ops import box_cxcywh_to_xyxy, box_iou
        for b,t in enumerate(targets):
            gt=t["boxes"]
            if len(gt)==0:
                rewards.append(prob[b].mean()*0)
                continue
            iou,_=box_iou(box_cxcywh_to_xyxy(boxes[b]),box_cxcywh_to_xyxy(gt))
            rewards.append((prob[b]*iou.max(1).values).mean())
        return torch.stack(rewards)

    def forward(self, feat, targets=None, num_queries=100):
        feat=self.proj(feat)
        B=feat.shape[0]
        if self.training and targets is not None:
            gt=self._pad_gt(targets,num_queries,feat.device)
            t=torch.full((B,),self.steps-1,device=feat.device,dtype=torch.long)
            noisy=self.corrupt_boxes(gt,t)
            traj=[]
            rewards=[]
            for _ in range(self.trajectories):
                out=self.run_trajectory(feat,noisy,stochastic=True)
                traj.append(out); rewards.append(self.reward(out["logits"],out["boxes"],targets))
            rewards=torch.stack(rewards,1) # [B,N]
            normalized=(rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True)+1e-6)
            logp=torch.stack([x["logprob"] for x in traj],1)
            # Contrastive positive/negative policy objective.
            pos=normalized.clamp(min=0)
            neg=(-normalized).clamp(min=0)
            contrastive=-(pos*logp).mean()+(neg*logp).mean()
            best=rewards.argmax(1)
            final_queries=torch.stack([traj[best[b]]["queries"][b] for b in range(B)])
            aux={"rdqg_loss":contrastive,"trajectory_rewards":rewards.detach()}
            return final_queries,aux
        # Inference: one stochastic-free reverse trajectory.
        start=torch.rand(B,num_queries,4,device=feat.device)
        out=self.run_trajectory(feat,start,stochastic=False)
        return out["queries"],{"rdqg_loss":feat.sum()*0}
