#!/usr/bin/env bash
# Train or test the PHD baseline.
#
# Usage:
#   bash train_test_phd.sh h36m train    # phase 1, then phase 2
#   bash train_test_phd.sh h36m test     # test the phase-2 model
#   bash train_test_phd.sh 3dpw train
#   bash train_test_phd.sh 3dpw test
#
# Fill in the paths below before running.

# ── Fill these in ────────────────────────────────────────────
NUM_GPUS=1

# H36M
H36M_3D="/path/to/data_3d_h36m.npz"
H36M_2D="/path/to/data_2d_h36m_gt.npz"
H36M_FEATS="/path/to/h36m_features.npy"        # from extract_features.sh

# 3DPW
PW3D_FEATS="/path/to/3dpw_features.npy"        # from extract_features.sh
PW3D_META="/path/to/3dpw_features_meta.npy"    # written next to the features

# Where checkpoints are saved
PHASE1_DIR="checkpoints/phd_phase1"
PHASE2_DIR="checkpoints/phd_phase2"
# ─────────────────────────────────────────────────────────────

DATASET=$1
MODE=$2
set -e   # stop if phase 1 fails, so phase 2 never starts without its checkpoint

RUN="torchrun --nproc_per_node=$NUM_GPUS"

if [ "$DATASET" = "h36m" ]; then
    DATA_ARGS="--dataset h36m
               --dataset_path $H36M_3D
               --dataset_2d_path $H36M_2D
               --features_path $H36M_FEATS
               --subjects_train S1,S5,S6,S7,S8
               --subjects_val S11
               --subjects_test S9"
elif [ "$DATASET" = "3dpw" ]; then
    DATA_ARGS="--dataset 3dpw
               --dataset_3dpw_path $PW3D_META
               --dataset_3dpw_feats $PW3D_FEATS"
else
    echo "Usage: bash train_test_phd.sh h36m|3dpw train|test"
    exit 1
fi

if [ "$MODE" = "train" ]; then
    # Phase 1: f_movie + f_3D (reconstruction)
    $RUN train_phd_ddp.py $DATA_ARGS \
        --mode train \
        --checkpoint_dir "$PHASE1_DIR"

    # Phase 2: f_movie + f_AR (forecasting), f_3D frozen
    $RUN train_phd_ddp2.py $DATA_ARGS \
        --mode train \
        --phase1_ckpt "$PHASE1_DIR/best_phd_model_phase1.pth" \
        --checkpoint_dir "$PHASE2_DIR"

elif [ "$MODE" = "test" ]; then
    $RUN train_phd_ddp2.py $DATA_ARGS \
        --mode test \
        --phase1_ckpt "$PHASE1_DIR/best_phd_model_phase1.pth" \
        --checkpoint_dir "$PHASE2_DIR"

else
    echo "Usage: bash train_test_phd.sh h36m|3dpw train|test"
    exit 1
fi