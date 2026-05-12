#!/bin/bash
# Minimal training run to test segmentation projection

accelerate launch \
    train.py \
    --data_path data/anomaly_dataset/mini_5.json \
    --image_root data/ \
    --output_dir ./checkpoints_mini \
    --num_epochs 30 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --mask_size 384 \
    --freeze_vision \
    --use_lora \
    --seg_training
