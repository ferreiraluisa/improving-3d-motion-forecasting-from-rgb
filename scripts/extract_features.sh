#!/usr/bin/env bash
# Usage:
#   bash scripts/extract_features.sh h36m   (from the repository root)
#   bash scripts/extract_features.sh 3dpw
#
# Fill in the paths below before running.

# ── Fill these in ────────────────────────────────────────────
EXTRACTOR="hmr2_crossattn"   # resnet_v2 | hmr2_vit_mean | hmr2_crossattn | wham
CHECKPOINT="/path/to/checkpoint.ckpt"   # wham: leave empty to use WHAM's default HMR2 ckpt
OUT="/path/to/output/features.npy"
WHAM_ROOT="$HOME/WHAM"                 # only for EXTRACTOR=wham (ViTPose detector)

H36M_2D="/path/to/data_2d_h36m_gt.npz"
H36M_VIDEOS="/path/to/h36m/videos"

PW3D_ROOT="/path/to/3DPW"
# ─────────────────────────────────────────────────────────────

export PYTHONPATH=.   # the code imports models.* and dataset.*

DATASET=$1

EXTRA=()
[ -n "$CHECKPOINT" ] && EXTRA+=(--ckpt "$CHECKPOINT")
[ "$EXTRACTOR" = "wham" ] && EXTRA+=(--wham-root "$WHAM_ROOT")

if [ "$DATASET" = "h36m" ]; then
    python dataset/feats.py \
        --dataset_2d_path "$H36M_2D" \
        --vid_path "$H36M_VIDEOS" \
        --extractor "$EXTRACTOR" \
        --out "$OUT" \
        "${EXTRA[@]}"

elif [ "$DATASET" = "3dpw" ]; then
    python dataset/feats_3dpw.py \
        --root "$PW3D_ROOT" \
        --extractor "$EXTRACTOR" \
        --out "$OUT" \
        "${EXTRA[@]}"

else
    echo "Usage: bash extract_features.sh h36m|3dpw"
fi