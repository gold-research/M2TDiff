import torch
import torch.nn as nn
import torch.nn.functional as F

class SetCriterion(nn.Module):
    # Lightweight placeholder matcher; replace with Hungarian matching for final reproduction.
    def forward(self, outputs, targets):
        logits,boxes=outputs["pred_logits"],outputs["pred_boxes"]
        B,Q,_=logits.shape
        loss_cls=0.; loss_box=0.
        for b,t in enumerate(targets):
            n=min(Q,len(t["labels"]))
            if n==0: continue
            loss_cls=loss_cls+F.cross_entropy(logits[b,:n],t["labels"][:n])
            loss_box=loss_box+F.l1_loss(boxes[b,:n],t["boxes"][:n])
        return {"loss":loss_cls+5*loss_box}
