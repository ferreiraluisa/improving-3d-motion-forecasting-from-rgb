#!/usr/bin/env bash
# Usage:
#   bash extract_features.sh h36m
#   bash extract_features.sh 3dpw
#
# Fill in the paths below before running.

# ── Fill these in ────────────────────────────────────────────
EXTRACTOR="hmr2_crossattn"   # resnet_v2 | hmr2_vit_mean | hmr2_crossattn
CHECKPOINT="/path/to/checkpoint.ckpt"
OUT="/path/to/output/features.npy"

H36M_2D="/path/to/data_2d_h36m_gt.npz"
H36M_VIDEOS="/path/to/h36m/videos"

PW3D_ROOT="/path/to/3DPW"
# ─────────────────────────────────────────────────────────────

DATASET=$1

if [ "$DATASET" = "h36m" ]; then
    python feats.py \
        --dataset_2d_path "$H36M_2D" \
        --vid_path "$H36M_VIDEOS" \
        --ckpt "$CHECKPOINT" \
        --extractor "$EXTRACTOR" \
        --out "$OUT"

elif [ "$DATASET" = "3dpw" ]; then
    python feats_3dpw.py \
        --root "$PW3D_ROOT" \
        --ckpt "$CHECKPOINT" \
        --extractor "$EXTRACTOR" \
        --out "$OUT"

else
    echo "Usage: bash extract_features.sh h36m|3dpw"
fi