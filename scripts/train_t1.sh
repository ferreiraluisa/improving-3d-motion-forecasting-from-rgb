#!/bin/bash
# -------------- NCCL workaround (broken PCIe P2P on cvg-srv08) ---------
export NCCL_P2P_DISABLE=1
export PYTHONPATH=.
export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=3

export CUDA_VISIBLE_DEVICES=3,4,5,6
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/vit
# -------------- Logging ------------------------------------------------
echo "Run started at:"
date
nvidia-smi -L
# -------------- Launch -------------------------------------------------
torchrun --standalone --nproc_per_node=4 --master_port=29517 \
src/train_phd_ddp.py \
--latent_dim 1280 \
--dataset_path /media/data/ferreiral1/data_3d_h36m.npz \
--dataset_2d_path /media/data/ferreiral1/data_2d_h36m_gt.npz \
--features_path /media/data/ferreiral1/hmr2_vit_mean/feats.npy \
--checkpoint_dir ../checkpoints \
--checkpoint_save_name 'best_phd_model_phase1_hmr2_vit_mean.pth' \
--subjects_train 'S1,S6,S7,S8,S9' \
--subjects_val 'S11' \
--subjects_test 'S5' \
--batch_size 512 \
--config_path config/MB_train_h36m_long.yaml \
--downsample 2 \
--workers 4 \
--stride 2 \
--mode test \
--lr 1e-4 \
--warmup_epochs 6 \
--epochs 50 \
