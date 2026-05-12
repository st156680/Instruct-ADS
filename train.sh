#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=hk-project-pai00077
#SBATCH --job-name=llava_anomaly
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# salloc -p dev_accelerated --gres=gpu:4 -t 60 --account=hk-project-pai00077

# ============================================================
# Single GPU training (simpler, good for debugging)
# ============================================================
# python train.py \
#     --data_path data/anomaly_dataset/mvtec_zero_shot_seg1turn.json \
#     --image_root data/ \
#     --output_dir ./checkpoints \
#     --num_epochs 3 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 2e-5 \
#     --freeze_vision \
#     --seg_training

# ============================================================
# Multi-GPU training with accelerate (4x GPU)
# ============================================================
accelerate launch \
    train.py \
    --data_path data/anomaly_dataset/mvtec_zero_shot_seg1turn.json \
    --image_root data/ \
    --output_dir ./checkpoints \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --mask_size 384 \
    --freeze_vision \
    --use_lora \
    --seg_training
