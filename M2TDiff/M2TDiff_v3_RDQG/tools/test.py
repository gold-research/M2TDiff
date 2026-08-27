import yaml, torch
from models import M2TDiff
def main():
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/m2tdiff_r101.yaml'); p.add_argument('--checkpoint',required=True); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); model=M2TDiff(**cfg['model'])
    ckpt=torch.load(a.checkpoint,map_location='cpu'); model.load_state_dict(ckpt['model'],strict=False)
    model.eval(); print('Checkpoint loaded successfully.')
if __name__=='__main__': main()
