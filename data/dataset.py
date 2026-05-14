"""
Dataset class for anomaly detection training with LLaVA OneVision 1.5.

Supports the existing MVTec JSON format from LLaVA-NeXT:
{
    "id": "mvtec_00000000",
    "image": "anomaly_dataset/mvtec_anomaly_detection/capsule/test/poke/004.png",
    "conversations": [
        {"from": "human", "value": "Is there any anomaly on the capsule? If yes, segment it."},
        {"from": "gpt", "value": "Yes, there is an anomaly... [SEG_DEFECT] [SEG_NORMAL]."}
    ],
    "mask": "anomaly_dataset/mvtec_anomaly_detection/capsule/ground_truth/poke/004_mask.png",
    "metadata": { "anomaly": true, ... }
}
"""

import json
import os
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class AnomalyDetectionDataset(Dataset):
    """
    Loads anomaly detection data: images, conversation text, and segmentation masks.
    Uses the Qwen2.5VL processor for image preprocessing and tokenization.

    Args:
        json_path: Path to the JSON dataset file.
        processor: The Qwen2.5VL processor (handles both image and text processing).
        image_root: Root directory for image paths (prepended to relative paths in JSON).
        seg_token: The segmentation defect token string.
        seg_normal_token: The segmentation normal token string.
    """
    def __init__(
        self,
        json_path: str,
        processor,
        image_root: str = "",
        seg_token: str = "[SEG_DEFECT]",
        seg_normal_token: str = "[SEG_NORMAL]",
    ):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.processor = processor
        self.image_root = image_root
        self.seg_token = seg_token
        self.seg_normal_token = seg_normal_token

        # Get special token IDs for label masking
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.data)

    def _build_messages(self, item: Dict) -> List[Dict]:
        """Convert the JSON conversation format to Qwen2.5VL chat format."""
        conversations = item["conversations"]

        # Extract the human question
        human_text = conversations[0]["value"]
        # Remove <image> placeholder if present (not used in this format)
        human_text = human_text.replace("<image>\n", "").replace("<image>", "")

        # Get assistant response
        assistant_text = conversations[1]["value"]

        # Build messages in Qwen2.5VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": human_text},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ]
        return messages

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build labels for causal LM training.
        Masks everything before the assistant's response with -100.
        Only the assistant's actual response tokens contribute to the loss.
        """
        labels = input_ids.clone()

        # Find all <|im_start|> positions
        im_start_positions = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]

        if len(im_start_positions) >= 2:
            # The last <|im_start|> is the assistant's turn header
            assistant_start = im_start_positions[-1].item()

            # Encode "assistant\n" to count how many tokens to skip
            assistant_header_tokens = self.processor.tokenizer.encode(
                "assistant\n", add_special_tokens=False
            )
            mask_end = assistant_start + 1 + len(assistant_header_tokens)

            # Mask everything before the assistant response content
            labels[:mask_end] = -100

        # Mask the trailing <|im_end|>
        if labels[-1] == self.im_end_id:
            labels[-1] = -100

        # Mask any padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        return labels

    def _load_mask(self, mask_path: Optional[str], target_size: tuple) -> torch.Tensor:
        """
        Load and process segmentation mask.

        Args:
            mask_path: Path to mask image, or None for normal images.
            target_size: (width, height) of the original image.

        Returns:
            Binary mask tensor of shape [H, W] with values in {0, 1}.
        """
        if mask_path is None or mask_path == "":
            return torch.zeros(target_size[1], target_size[0], dtype=torch.float32)

        full_path = os.path.join(self.image_root, mask_path) if self.image_root else mask_path

        if not os.path.exists(full_path):
            print(f"Warning: Mask not found at {full_path}, using zero mask")
            return torch.zeros(target_size[1], target_size[0], dtype=torch.float32)

        mask = Image.open(full_path).convert("L")
        mask = mask.resize((target_size[0], target_size[1]), Image.NEAREST)
        mask = np.array(mask).astype(np.float32)

        # Binarize: any non-zero value is defective
        mask = (mask > 0).astype(np.float32)

        return torch.from_numpy(mask)

    def _get_mask_path(self, item: Dict) -> Optional[str]:
        """Extract mask path from item, checking both top-level and metadata."""
        mask = item.get("mask")
        if mask is not None:
            return mask
        metadata = item.get("metadata", {})
        return metadata.get("mask")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # 1. Load image
        image_path = os.path.join(self.image_root, item["image"]) if self.image_root else item["image"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size  # (width, height)

        # 2. Build chat messages
        messages = self._build_messages(item)

        # 3. Apply chat template (get the formatted string without tokenizing)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 4. Process with the Qwen2.5VL processor
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]          # [num_patches, C*patch_h*patch_w]
        image_grid_thw = inputs["image_grid_thw"]      # [1, 3]

        # 5. Build labels
        labels = self._build_labels(input_ids)

        # 6. Load segmentation mask
        mask_path = self._get_mask_path(item)
        gt_mask = self._load_mask(mask_path, image_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "gt_segmentation_mask": gt_mask,
        }
