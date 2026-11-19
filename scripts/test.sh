#!/bin/bash

export PYTHONPATH=.
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/vit

echo "Run started at:"
date

torchrun --standalone --nproc_per_node=1 \
        src/train_motionst.py \
            --feat_dim 1024 \
            --d_model 128 \
            --n_heads 4 \
            --num_layers 4 \
            --dim_ff 256 \
            --dataset_path  /media/data/ferreiral1/data_3d_h36m.npz \
            --dataset_2d_path /media/data/ferreiral1/data_2d_h36m_gt.npz \
            --resnet_features_path /media/data/ferreiral1/hmr2_decoder/hmr2_features.npy \
            --checkpoint_dir ../checkpoints \
            --checkpoint_save_name 'best_st_hmr2_decoder.pth' \
            --subjects_train 'S1,S6,S7,S8,S11' \
            --subjects_val 'S5' \
            --subjects_test 'S9' \
            --batch_size 128 \
            --config_path config/MB_train_h36m_long.yaml \
            --downsample 2 \
            --workers 2 \
            --stride 2 \
            --mode test \
            --lr 1e-4 \
            --epochs 50 \
            --lambda_future 1.0 \
            --num_layers 6