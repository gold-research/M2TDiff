import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .matcher import HungarianMatcher

class SetCriterion(nn.Module):
    def __init__(self,num_classes,matcher=None,eos_coef=0.1,weight_dict=None,rdqg_coef=1.0):
        super().__init__()
        self.num_classes=num_classes; self.matcher=matcher or HungarianMatcher()
        self.weight_dict=weight_dict or {"loss_ce":1,"loss_bbox":5,"loss_giou":2}
        self.rdqg_coef=rdqg_coef
        empty=torch.ones(num_classes+1); empty[-1]=eos_coef
        self.register_buffer("empty_weight",empty)

    def forward(self,outputs,targets):
        indices=self.matcher(outputs,targets)
        logits=outputs["pred_logits"]; B,Q,_=logits.shape
        target_classes=torch.full((B,Q),self.num_classes,dtype=torch.long,device=logits.device)
        src=[]; tgt=[]
        for b,(i,j) in enumerate(indices):
            if len(i):
                target_classes[b,i]=targets[b]["labels"][j]
                src.append(outputs["pred_boxes"][b,i]); tgt.append(targets[b]["boxes"][j])
        loss_ce=F.cross_entropy(logits.transpose(1,2),target_classes,self.empty_weight)
        if src:
            src=torch.cat(src); tgt=torch.cat(tgt)
            loss_bbox=F.l1_loss(src,tgt,reduction="mean")
            giou=generalized_box_iou(box_cxcywh_to_xyxy(src),box_cxcywh_to_xyxy(tgt))
            loss_giou=(1-giou.diag()).mean()
        else:
            loss_bbox=logits.sum()*0; loss_giou=logits.sum()*0
        rdqg=outputs.get("rdqg_loss",logits.sum()*0)
        loss=(loss_ce+5*loss_bbox+2*loss_giou+self.rdqg_coef*rdqg)
        return {"loss":loss,"loss_ce":loss_ce,"loss_bbox":loss_bbox,"loss_giou":loss_giou,"loss_rdqg":rdqg}
