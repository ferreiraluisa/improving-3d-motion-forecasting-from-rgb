#!/bin/bash

export PYTHONPATH=.
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/vit

export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES="1"
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Run started at:"
date

torchrun --standalone --nproc_per_node=1 \
        src/train_motionst.py \
            --wandb_run_name "wham ctx only" \
            --d_model 128 \
            --feat_dim 563 \
            --wham \
            --context_only \
            --n_heads 4 \
            --num_layers 4 \
            --dim_ff 256 \
            --dataset_path  /media/data/ferreiral1/data_3d_h36m.npz \
            --dataset_2d_path /media/data/ferreiral1/data_2d_h36m_gt.npz \
            --features_path /media/data/ferreiral1/WHAM/feats_h36m.npy \
            --checkpoint_dir checkpoints \
            --checkpoint_save_name 'wham_h36m_ctx.pth' \
            --subjects_train 'S1,S6,S7,S8,S9' \
            --subjects_val 'S11' \
            --subjects_test 'S5' \
            --batch_size 256 \
            --downsample 2 \
            --workers 16 \
            --stride 2 \
            --lr 3e-4 \
            --mode train \
            --epochs 50