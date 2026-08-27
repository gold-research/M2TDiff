import torch
from models.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(torch.nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class,self.cost_bbox,self.cost_giou=cost_class,cost_bbox,cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as e:
            raise ImportError("Install scipy for Hungarian matching: pip install scipy") from e
        bs,num_queries=outputs["pred_logits"].shape[:2]
        out_prob=outputs["pred_logits"].softmax(-1)
        out_bbox=outputs["pred_boxes"]
        indices=[]
        for b in range(bs):
            tgt_ids=targets[b]["labels"]
            tgt_bbox=targets[b]["boxes"]
            if tgt_ids.numel()==0:
                indices.append((torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64)))
                continue
            cost_class=-out_prob[b][:,tgt_ids]
            cost_bbox=torch.cdist(out_bbox[b],tgt_bbox,p=1)
            cost_giou=-generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            C=self.cost_class*cost_class+self.cost_bbox*cost_bbox+self.cost_giou*cost_giou
            i,j=linear_sum_assignment(C.cpu())
            indices.append((torch.as_tensor(i,dtype=torch.int64),torch.as_tensor(j,dtype=torch.int64)))
        return indices
