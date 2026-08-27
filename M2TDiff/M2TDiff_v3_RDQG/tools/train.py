import os, argparse, yaml, torch
from torch.utils.data import DataLoader
from models import M2TDiff
from datasets.imagenet_vid import ImageNetVIDDataset, collate_fn
from engine.criterion import SetCriterion
from engine.trainer import train_one_epoch, evaluate_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/m2tdiff_r101.yaml"); args=p.parse_args()
    cfg=yaml.safe_load(open(args.config))
    device=torch.device(cfg["train"].get("device","cuda") if torch.cuda.is_available() else "cpu")
    model=M2TDiff(**cfg["model"]).to(device)
    criterion=SetCriterion(cfg["model"]["num_classes"]).to(device)
    ds=ImageNetVIDDataset(cfg["data"]["root"],split="train",num_frames=cfg["model"]["num_frames"])
    loader=DataLoader(ds,batch_size=cfg["train"]["batch_size"],shuffle=True,
                      num_workers=cfg["data"]["num_workers"],collate_fn=collate_fn,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["lr"],weight_decay=cfg["train"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=cfg["train"].get("lr_drop",10),gamma=0.1)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    os.makedirs("outputs",exist_ok=True)
    best=float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        stats=train_one_epoch(model,criterion,loader,opt,device,scaler)
        scheduler.step()
        state={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"config":cfg}
        torch.save(state,"outputs/latest.pth")
        if stats["loss"]<best:
            best=stats["loss"]; torch.save(state,"outputs/best.pth")
        print(f"Epoch {epoch+1}: {stats}")
if __name__=="__main__": main()
