"""
Custom data collator for anomaly detection training.

Handles variable-length sequences (padding), variable-size pixel_values
(concatenation), and segmentation masks.
"""

from typing import List, Dict, Any
import torch
import torch.nn.functional as F


class AnomalyDataCollator:
    """
    Collates individual dataset samples into batches.
    
    Handles:
    - Padding input_ids, attention_mask, labels to the max sequence length in the batch
    - Concatenating pixel_values along dim=0 (patches from different images)
    - Stacking image_grid_thw
    - Resizing and stacking segmentation masks to a common size
    
    Args:
        pad_token_id: Token ID used for padding input_ids. Default: 151643 (<|endoftext|>).
        mask_size: Target size (H, W) to resize all segmentation masks to. Default: (384, 384).
    """
    def __init__(self, pad_token_id: int = 151643, mask_size: tuple = (384, 384)):
        self.pad_token_id = pad_token_id
        self.mask_size = mask_size

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        
        # --- Pad text tensors (input_ids, attention_mask, labels) ---
        max_len = max(f["input_ids"].shape[0] for f in features)
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len
            
            # Left-padding is standard for causal LM generation,
            # but right-padding is fine for training
            input_ids_list.append(
                F.pad(f["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            attention_mask_list.append(
                F.pad(f["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                F.pad(f["labels"], (0, pad_len), value=-100)
            )
        
        # --- Concatenate pixel_values ---
        # pixel_values from different images may have different numbers of patches
        # but the same patch dimension. We concatenate along dim=0.
        # The model uses image_grid_thw to know which patches belong to which image.
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        
        # --- Stack image_grid_thw ---
        image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        
        # --- Stack segmentation masks at original resolution ---
        # Pad to max size in batch (no fixed resize, matches V1.0 behavior)
        masks = [f["gt_segmentation_mask"] for f in features]
        max_h = max(m.shape[-2] for m in masks)
        max_w = max(m.shape[-1] for m in masks)

        padded_masks = []
        for mask in masks:
            h, w = mask.shape[-2], mask.shape[-1]
            padded = F.pad(mask.float(), (0, max_w - w, 0, max_h - h), value=0.0)
            padded_masks.append(padded)

        gt_segmentation_masks = torch.stack(padded_masks, dim=0)  # [B, H, W]

        batch = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": torch.stack(labels_list),
            "gt_segmentation_masks": gt_segmentation_masks,
        }

        return batch

