#!/bin/bash

export PYTHONPATH=.
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/vit

export CUDA_VISIBLE_DEVICES=7
 #-------------- Logging ------------------------------------------------
echo "Run started at:"
date

# -------------- Launch ------------------------------------------------
torchrun --standalone --nproc_per_node=1 \
    src/train_phd_ddp2.py \
        --latent_dim 1280 \
        --dataset_path /media/data/ferreiral1/data_3d_h36m.npz \
        --dataset_2d_path /media/data/ferreiral1/data_2d_h36m_gt.npz \
        --features_path /media/data/ferreiral1/hmr2_vit_mean/feats.npy \
        --phase1_ckpt ../checkpoints/best_phd_model_phase1_hmr2_vit_mean.pth \
        --checkpoint_dir ../checkpoints \
        --checkpoint_save_name 'best_phase2_hmr2_vit_mean.pth' \
        --work_dir ./work_dir_phase2 \
        --subjects_train 'S1,S6,S7,S8,S9' \
        --subjects_val 'S11' \
        --subjects_test 'S5' \
        --config_path config/MB_train_h36m_long.yaml \
        --downsample 2 \
        --stride 2 \
        --mode test \
        --epochs 50 \
        --max_pred_steps 25 \
        --curriculum_epochs 15 \
        --batch_size 1024 \
        --lr 2e-4 \
        --workers 6 \

