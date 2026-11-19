#!/usr/bin/env bash
# Train the WHAM skeleton adapter and rewrite the WHAM features with it.
#
#   H36M: COCO-17 -> H36M-17   (features stay 563-d)
#   3DPW: COCO-17 -> SMPL-24   (features go from 563-d to 584-d)
#
# Usage:
#   bash wham_adapter.sh h36m
#   bash wham_adapter.sh 3dpw
#
# Fill in the paths below before running.

# ── Fill these in ────────────────────────────────────────────
# H36M
H36M_3D="/path/to/data_3d_h36m.npz"
H36M_2D="/path/to/data_2d_h36m_gt.npz"
H36M_WHAM_FEATS="/path/to/wham_feats_h36m.npy"          # from extract_features.sh
H36M_OUT="/path/to/wham_feats_h36m_fixed.npy"           # new features

# 3DPW
PW3D_WHAM_FEATS="/path/to/3dpw_wham.npy"                # from extract_features.sh
PW3D_OUT="/path/to/3dpw_wham_smpl.npy"                  # new features

CKPT_DIR="checkpoints"
EPOCHS=30
# ─────────────────────────────────────────────────────────────

DATASET=$1
set -e   # stop if training fails, so we never convert with a missing checkpoint

if [ "$DATASET" = "h36m" ]; then
    # 1) train the adapter
    python train_h36m_adapter.py \
        --dataset_path "$H36M_3D" \
        --dataset_2d_path "$H36M_2D" \
        --features_path "$H36M_WHAM_FEATS" \
        --ckpt "$CKPT_DIR/h36_adapter" \
        --epochs "$EPOCHS" \
        --fit_linear

    # 2) rewrite the features with the trained adapter
    python wham_h36.py \
        --in "$H36M_WHAM_FEATS" \
        --out "$H36M_OUT" \
        --ckpt "$CKPT_DIR/h36_adapter_best.pth"

elif [ "$DATASET" = "3dpw" ]; then
    # extract_features.sh writes the ground truth next to the features as *_meta.npy
    PW3D_META="${PW3D_WHAM_FEATS%.npy}_meta.npy"

    # 1) train the adapter
    python train_smpl_adapter.py \
        --feats_path "$PW3D_WHAM_FEATS" \
        --meta_path "$PW3D_META" \
        --ckpt "$CKPT_DIR/smpl_adapter_3dpw" \
        --epochs "$EPOCHS" \
        --fit_linear

    # 2) rewrite the features with the trained adapter
    python wham_3dpw.py \
        --in "$PW3D_WHAM_FEATS" \
        --out "$PW3D_OUT" \
        --ckpt "$CKPT_DIR/smpl_adapter_3dpw_best.pth"

else
    echo "Usage: bash wham_adapter.sh h36m|3dpw"
fi