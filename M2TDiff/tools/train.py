import os, yaml, torch
from torch.utils.data import DataLoader
from models import M2TDiff
from engine.criterion import SetCriterion

def main():
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/m2tdiff_r101.yaml'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config))
    mcfg=cfg['model']; model=M2TDiff(**mcfg).to(cfg['train']['device'])
    criterion=SetCriterion().to(cfg['train']['device'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay'])
    print('Model initialized. Connect ImageNetVIDDataset, then start the epoch loop.')
    os.makedirs('outputs',exist_ok=True)
    torch.save({'model':model.state_dict(),'config':cfg},'outputs/initialized.pth')
if __name__=='__main__': main()
