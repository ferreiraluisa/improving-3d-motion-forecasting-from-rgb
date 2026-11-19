#!/bin/bash
# Train and test the PHD baseline on Human3.6M.
#   Phase 1: train f_movie + f_3D (3D pose reconstruction)
#   Phase 2: freeze f_3D, train f_movie + f_AR (forecasting), then test
#
# Usage: bash train_phd_example.sh

# Data paths
DATA_3D=/media/data/ferreiral1/data_3d_h36m.npz
DATA_2D=/media/data/ferreiral1/data_2d_h36m_gt.npz

# Feature extractor: WHAM (563-dim) or HMR ResNet (2048-dim)
NAME=wham
FEATURES=/media/data/ferreiral1/WHAM/feats_h36m.npy
LATENT_DIM=563
# NAME=hmr_resnet
# FEATURES=/media/data/ferreiral1/hmr_resnet/feats.npy
# LATENT_DIM=2048

CKPT_DIR=../checkpoints
PHASE1_CKPT=best_phd_model_phase1_${NAME}.pth
PHASE2_CKPT=best_phase2_${NAME}.pth

# Environment (NCCL_P2P_DISABLE: PCIe P2P is broken on cvg-srv08)
export PYTHONPATH=. NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4 TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/thesis

# Arguments shared by both phases
COMMON=(
    --latent_dim "$LATENT_DIM"
    --dataset_path "$DATA_3D" --dataset_2d_path "$DATA_2D" --features_path "$FEATURES"
    --subjects_train 'S1,S6,S7,S8,S9' --subjects_val 'S11' --subjects_test 'S5'
    --config_path config/config.yaml
    --checkpoint_dir "$CKPT_DIR"
    --downsample 2 --stride 2 --epochs 50
)

# Phase 2 arguments (used for training and testing)
PHASE2=(
    --phase1_ckpt "$CKPT_DIR/$PHASE1_CKPT" --checkpoint_save_name "$PHASE2_CKPT"
    --work_dir ./work_dir_phase2
    --max_pred_steps 25 --curriculum_epochs 15
    --batch_size 256 --lr 2e-4 --workers 6
)

echo "Started: $(date)"
nvidia-smi -L

echo "=== Phase 1: train ==="
torchrun --standalone --nproc_per_node=1 src/train_phd_ddp.py "${COMMON[@]}" \
    --checkpoint_save_name "$PHASE1_CKPT" \
    --batch_size 64 --lr 1e-4 --warmup_epochs 6 --workers 4 \
    --mode train

echo "=== Phase 2: train ==="
torchrun --standalone --nproc_per_node=1 src/train_phd_ddp2.py "${COMMON[@]}" "${PHASE2[@]}" --mode train

echo "=== Phase 2: test ==="
torchrun --standalone --nproc_per_node=1 src/train_phd_ddp2.py "${COMMON[@]}" "${PHASE2[@]}" --mode test

echo "Finished: $(date)"