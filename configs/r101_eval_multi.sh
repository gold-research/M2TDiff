#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/our_models/exps_multi/r101
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --epochs 7 \
    --eval \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --frames 30 \
    --plus_plus_ref_frames 10 \
    --rho 0.8 \
    --xi 0.7 \
    --nheads 4 \
    --resume ${EXP_DIR}/checkpoint0006.pth \
    --lr_drop_epochs 5 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_multi \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
