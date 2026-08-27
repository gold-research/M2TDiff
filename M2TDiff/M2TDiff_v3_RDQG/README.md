# M2TDiff

PyTorch implementation scaffold for **M2TDiff: Multi-Scale MoE-Enhanced Transformer Diffusion Network for Video Object Detection**.

## Implemented pipeline
`Video frames -> shared backbone -> {RDQG, MGTE} -> SMTD -> class/box heads`

## Quick start
```bash
pip install -r requirements.txt
python tools/train.py --config configs/m2tdiff_r101.yaml
python tools/test.py --config configs/m2tdiff_r101.yaml --checkpoint outputs/best.pth
```

The dataset adapter expects ImageNet-VID-style annotations and is intentionally isolated in `datasets/imagenet_vid.py`.
