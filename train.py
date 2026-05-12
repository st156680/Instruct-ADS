"""
Training script for LLaVA OneVision 1.5 anomaly detection.

Usage:
    # Single GPU:
    python train.py --data_path data/train.json --image_root /path/to/images

    # Multi-GPU with accelerate:
    accelerate launch --num_processes 4 --mixed_precision bf16 \
        train.py --data_path data/train.json --image_root /path/to/images
"""

import argparse
import os
import sys
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import AnomalyDetectionDataset
from data.collator import AnomalyDataCollator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rank0_log(msg: str, level: str = "info"):
    """Only log from rank 0 (or non-distributed). Avoids duplicate logs from all GPUs."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        getattr(logger, level)(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLaVA-OV 1.5 for anomaly detection")
    parser.add_argument("--model_path", type=str, default="./model",
                        help="Path to the pretrained model directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training JSON file")
    parser.add_argument("--image_root", type=str, default="",
                        help="Root directory for image paths in the dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--mask_size", type=int, default=384,
                        help="Target size for segmentation masks (H=W)")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Freeze the vision encoder during training")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for training to reduce memory usage")
    parser.add_argument("--seg_training", action="store_true",
                        help="Enable segmentation loss (requires masks in dataset)")
    parser.add_argument("--seg_token", type=str, default="[SEG_DEFECT]",
                        help="Segmentation defect token")
    parser.add_argument("--seg_normal_token", type=str, default="[SEG_NORMAL]",
                        help="Segmentation normal token")
    return parser.parse_args()


class AnomalyTrainer(Trainer):
    """
    Custom Trainer that handles the gt_segmentation_masks field.
    
    The default Trainer would drop unknown columns. We override compute_loss
    to extract the masks and pass them to the model's forward method.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop segmentation masks (not a standard model input)
        gt_masks = inputs.pop("gt_segmentation_masks", None)

        # Forward pass with segmentation masks
        outputs = model(
            **inputs,
            gt_segmentation_masks=gt_masks,
            output_hidden_states=True,
        )

        loss = outputs.loss

        if loss is None:
            raise ValueError("Model returned None loss. Check that labels are provided.")

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    # Clear stale HuggingFace module cache so our modified model code is used
    import shutil
    hf_cache_modules = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/model")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0 and os.path.exists(hf_cache_modules):
        shutil.rmtree(hf_cache_modules, ignore_errors=True)
        rank0_log("Cleared stale HF module cache")

    rank0_log(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Initialize custom segmentation projector with weights from the visual merger
    if hasattr(model.model, "seg_projector"):
        model.model.seg_projector.load_state_dict(model.model.visual.merger.state_dict())
        rank0_log("Initialized custom seg_projector with visual.merger weights")

    # --- Add SEG tokens ---
    rank0_log("Adding segmentation tokens to vocabulary...")
    num_added = processor.tokenizer.add_tokens(
        [args.seg_token, args.seg_normal_token],
        special_tokens=True,
    )
    rank0_log(f"Added {num_added} new tokens")

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Store SEG token IDs in config for use during forward pass
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(args.seg_token)
    seg_normal_token_id = processor.tokenizer.convert_tokens_to_ids(args.seg_normal_token)
    model.config.seg_token_idx = seg_token_id
    model.config.seg_normal_token_idx = seg_normal_token_id
    rank0_log(f"SEG token IDs: defect={seg_token_id}, normal={seg_normal_token_id}")
        
    # Update vocab_size in config so loss_function uses the correct size
    new_vocab_size = len(processor.tokenizer)
    model.config.vocab_size = new_vocab_size
    model.config.text_config.vocab_size = new_vocab_size

    # Store SEG token IDs in config for use during forward pass
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(args.seg_token)
    seg_normal_token_id = processor.tokenizer.convert_tokens_to_ids(args.seg_normal_token)
    model.config.seg_token_idx = seg_token_id
    model.config.seg_normal_token_idx = seg_normal_token_id
    rank0_log(f"SEG token IDs: defect={seg_token_id}, normal={seg_normal_token_id}")
    rank0_log(f"Vocab size updated: {new_vocab_size}")

    # --- Freeze vision encoder (optional) ---
    if args.freeze_vision:
        rank0_log("Freezing vision encoder...")
        for param in model.model.visual.parameters():
            param.requires_grad = False
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        rank0_log(f"Trainable: {trainable/1e9:.2f}B / Total: {total/1e9:.2f}B")

    # --- Enable gradient checkpointing for memory savings ---
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # --- Setup LoRA (optional) ---
    if args.use_lora:
        rank0_log("Applying LoRA...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Required for gradient checkpointing + LoRA
        model.enable_input_require_grads()
        
        # Determine target modules. Qwen2 uses q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        modules_to_save = ["lm_head", "embed_tokens"] # Important for new vocab tokens
        if hasattr(model.model, "seg_projector"):
            modules_to_save.append("seg_projector")
            rank0_log("Added seg_projector to LoRA modules_to_save")

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=modules_to_save
        )
        
        # When freezing vision, ensure vision modules are NOT targeted by LoRA if they match name patterns,
        # but the above target modules are typical for the LLM. 
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters after LoRA
        model.print_trainable_parameters()
    else:
        rank0_log("Full fine-tuning (No LoRA)")

    # --- Create dataset ---
    rank0_log(f"Loading dataset from {args.data_path}...")
    train_dataset = AnomalyDetectionDataset(
        json_path=args.data_path,
        processor=processor,
        image_root=args.image_root,
        seg_token=args.seg_token,
        seg_normal_token=args.seg_normal_token,
    )
    rank0_log(f"Dataset size: {len(train_dataset)} samples")

    # --- Data collator ---
    collator = AnomalyDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        mask_size=(args.mask_size, args.mask_size),
    )

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important: keep gt_segmentation_masks
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_pin_memory=True,
    )

    # --- Trainer ---
    trainer = AnomalyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # --- Train ---
    rank0_log("Starting training...")
    trainer.train()

    # --- Save final model ---
    rank0_log(f"Saving model to {args.output_dir}/final...")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    # Copy modeling code so the saved checkpoint can be loaded with trust_remote_code
    import shutil
    for py_file in ["modeling_llavaonevision1_5.py", "configuration_llavaonevision1_5.py"]:
        src = os.path.join(args.model_path, py_file)
        dst = os.path.join(final_dir, py_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            rank0_log(f"Copied {py_file} to checkpoint")

    rank0_log("Training complete!")


if __name__ == "__main__":
    main()
