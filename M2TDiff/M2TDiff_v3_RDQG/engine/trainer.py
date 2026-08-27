import torch
from tqdm import tqdm

def train_one_epoch(model,criterion,loader,optimizer,device,scaler=None):
    model.train(); total=0.0
    for frames,targets in tqdm(loader,leave=False):
        frames=frames.to(device)
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            outputs=model(frames,targets=targets)
            losses=criterion(outputs,targets); loss=losses["loss"]
            loss.backward(); optimizer.step()
        else:
            with torch.autocast(device_type=device.type,enabled=True):
                outputs=model(frames,targets=targets)
                losses=criterion(outputs,targets); loss=losses["loss"]
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total+=loss.detach().item()
    return {"loss":total/max(len(loader),1)}

@torch.no_grad()
def evaluate_loss(model,criterion,loader,device):
    model.eval(); total=0.0
    for frames,targets in loader:
        frames=frames.to(device)
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        total+=criterion(model(frames,targets=targets),targets)["loss"].item()
    return {"loss":total/max(len(loader),1)}
