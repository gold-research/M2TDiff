from pathlib import Path
import random, xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from models.utils.box_ops import box_xyxy_to_cxcywh

VID_CLASSES = [
"airplane","antelope","bear","bicycle","bird","bus","car","cattle","dog","domestic_cat",
"elephant","fox","giant_panda","hamster","horse","lion","lizard","monkey","motorcycle",
"rabbit","red_panda","sheep","snake","squirrel","tiger","train","turtle","watercraft",
"whale","zebra"]

class ImageNetVIDDataset(Dataset):
    def __init__(self, root, split="train", num_frames=5, transform=None):
        self.root=Path(root); self.split=split; self.num_frames=num_frames; self.transform=transform
        self.frames=[]
        img_root=self.root/"Data"/"VID"/split
        ann_root=self.root/"Annotations"/"VID"/split
        if not img_root.exists():
            raise FileNotFoundError(f"ImageNet VID path not found: {img_root}")
        for jpg in img_root.rglob("*.JPEG"):
            rel=jpg.relative_to(img_root)
            xml=(ann_root/rel).with_suffix(".xml")
            if xml.exists(): self.frames.append((jpg,xml))
        self.class_to_idx={c:i for i,c in enumerate(VID_CLASSES)}

    def __len__(self): return len(self.frames)

    def _sequence(self, idx):
        center,_=self.frames[idx]
        parent=center.parent
        seq=sorted(parent.glob("*.JPEG"))
        pos=seq.index(center)
        half=self.num_frames//2
        out=[]
        for off in range(-half,half+1):
            j=min(max(pos+off,0),len(seq)-1)
            out.append(seq[j])
        return out

    def _target(self, xml_path, size):
        root=ET.parse(xml_path).getroot()
        labels=[]; boxes=[]; W,H=size
        for obj in root.findall("object"):
            name=obj.findtext("name")
            if name not in self.class_to_idx: continue
            bb=obj.find("bndbox")
            x1=float(bb.findtext("xmin"))/W; y1=float(bb.findtext("ymin"))/H
            x2=float(bb.findtext("xmax"))/W; y2=float(bb.findtext("ymax"))/H
            labels.append(self.class_to_idx[name]); boxes.append([x1,y1,x2,y2])
        if boxes:
            boxes=torch.tensor(boxes,dtype=torch.float32)
            boxes=box_xyxy_to_cxcywh(boxes)
            labels=torch.tensor(labels,dtype=torch.long)
        else:
            boxes=torch.zeros((0,4),dtype=torch.float32)
            labels=torch.zeros((0,),dtype=torch.long)
        return {"labels":labels,"boxes":boxes}

    def __getitem__(self, idx):
        imgs=self._sequence(idx)
        tensors=[]
        for p in imgs:
            im=Image.open(p).convert("RGB")
            if self.transform: im=self.transform(im)
            else: im=TF.to_tensor(im)
            tensors.append(im)
        center_img=Image.open(self.frames[idx][0]).convert("RGB")
        target=self._target(self.frames[idx][1],center_img.size)
        return torch.stack(tensors),target

def collate_fn(batch):
    frames,targets=zip(*batch)
    return torch.stack(frames),list(targets)
