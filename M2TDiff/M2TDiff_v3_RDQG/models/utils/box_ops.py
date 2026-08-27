import torch

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)

def box_xyxy_to_cxcywh(x):
    x0,y0,x1,y1=x.unbind(-1)
    return torch.stack([(x0+x1)/2,(y0+y1)/2,x1-x0,y1-y0],-1)

def box_area(boxes):
    return (boxes[...,2]-boxes[...,0]).clamp(min=0)*(boxes[...,3]-boxes[...,1]).clamp(min=0)

def box_iou(boxes1, boxes2):
    area1,area2=box_area(boxes1),box_area(boxes2)
    lt=torch.max(boxes1[:,None,:2],boxes2[None,:,:2])
    rb=torch.min(boxes1[:,None,2:],boxes2[None,:,2:])
    inter=(rb-lt).clamp(min=0).prod(-1)
    union=area1[:,None]+area2[None,:]-inter
    return inter/union.clamp(min=1e-6),union

def generalized_box_iou(boxes1, boxes2):
    iou,union=box_iou(boxes1,boxes2)
    lt=torch.min(boxes1[:,None,:2],boxes2[None,:,:2])
    rb=torch.max(boxes1[:,None,2:],boxes2[None,:,2:])
    area=(rb-lt).clamp(min=0).prod(-1)
    return iou-(area-union)/area.clamp(min=1e-6)
