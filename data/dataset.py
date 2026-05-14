"""
Dataset class for anomaly detection training with LLaVA OneVision 1.5.

This dataset makes image/mask preprocessing deterministic:

- image is resized to a fixed square training resolution
- mask is resized to the same fixed square resolution with nearest-neighbor interpolation
- mask is binarized to {0, 1}
- processor receives the already-resized image
- gt_segmentation_mask has shape [train_resolution, train_resolution]

Recommended default for the current model setup:
    train_resolution = 1008

Why 1008?
    1008 / patch_size 14 = 72 visual patches
    72 / spatial_merge_size 2 = 36 segmentation/anomaly grid

Expected flow:
    image:              [1008, 1008]
    mask:               [1008, 1008]
    image_grid_thw:      [1, 72, 72]  ideally
    anomaly_map:         [36, 36]
    upsampled logits:    [1008, 1008]
"""

import json
import os
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class AnomalyDetectionDataset(Dataset):
    """
    Loads anomaly detection data: images, conversation text, and segmentation masks.

    Supports JSON items like:

    {
        "id": "mvtec_00000000",
        "image": "anomaly_dataset/mvtec_anomaly_detection/capsule/test/poke/004.png",
        "conversations": [
            {
                "from": "human",
                "value": "Is there any anomaly on the capsule? If yes, segment it."
            },
            {
                "from": "gpt",
                "value": "Yes, there is an anomaly... [SEG_DEFECT] [SEG_NORMAL]."
            }
        ],
        "mask": "anomaly_dataset/mvtec_anomaly_detection/capsule/ground_truth/poke/004_mask.png",
        "metadata": {
            "anomaly": true
        }
    }

    Args:
        json_path:
            Path to the JSON dataset file.

        processor:
            Multimodal processor used for chat template, tokenization and image processing.

        image_root:
            Root directory prepended to relative image/mask paths.

        seg_token:
            Segmentation defect token string.

        seg_normal_token:
            Segmentation normal token string.

        train_resolution:
            Fixed square resolution for both image and mask.
            Recommended: 1008 for your current 36x36 anomaly-map setup.

        debug_first_n:
            Print debug info for the first N samples.
    """

    def __init__(
        self,
        json_path: str,
        processor,
        image_root: str = "",
        seg_token: str = "[SEG_DEFECT]",
        seg_normal_token: str = "[SEG_NORMAL]",
        train_resolution: int = 1008,
        debug_first_n: int = 5,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.image_root = image_root
        self.seg_token = seg_token
        self.seg_normal_token = seg_normal_token
        self.train_resolution = int(train_resolution)
        self.debug_first_n = int(debug_first_n)

        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        if self.im_start_id is None or self.im_start_id < 0:
            print("Warning: Could not resolve token id for <|im_start|>")

        if self.im_end_id is None or self.im_end_id < 0:
            print("Warning: Could not resolve token id for <|im_end|>")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None or path == "":
            return None

        if os.path.isabs(path):
            return path

        return os.path.join(self.image_root, path) if self.image_root else path

    def _build_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert the JSON conversation format to the processor chat format.
        """
        conversations = item["conversations"]

        human_text = conversations[0]["value"]
        human_text = human_text.replace("<image>\n", "").replace("<image>", "")

        assistant_text = conversations[1]["value"]

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

        Masks everything before the assistant response with -100.
        Only assistant response tokens contribute to LM loss.
        """
        labels = input_ids.clone()

        im_start_positions = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]

        if len(im_start_positions) >= 2:
            assistant_start = im_start_positions[-1].item()

            assistant_header_tokens = self.processor.tokenizer.encode(
                "assistant\n",
                add_special_tokens=False,
            )

            mask_end = assistant_start + 1 + len(assistant_header_tokens)
            labels[:mask_end] = -100

        if labels.numel() > 0 and labels[-1].item() == self.im_end_id:
            labels[-1] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        return labels

    def _get_mask_path(self, item: Dict[str, Any]) -> Optional[str]:
        """
        Extract mask path from top-level item or metadata.
        """
        mask_path = item.get("mask")
        if mask_path is not None:
            return mask_path

        metadata = item.get("metadata", {})
        return metadata.get("mask")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize RGB image to fixed square training resolution.

        Since your images are usually around 1000x1000, resizing to 1008x1008
        should introduce only minor scaling while making the resolution divisible
        by patch_size * spatial_merge_size = 14 * 2 = 28.
        """
        return image.resize(
            (self.train_resolution, self.train_resolution),
            Image.Resampling.BICUBIC,
        )

    def _load_mask(self, mask_path: Optional[str]) -> torch.Tensor:
        """
        Load, resize and binarize segmentation mask.

        Returns:
            Tensor of shape [train_resolution, train_resolution],
            dtype float32, values in {0, 1}.
        """
        if mask_path is None or mask_path == "":
            return torch.zeros(
                self.train_resolution,
                self.train_resolution,
                dtype=torch.float32,
            )

        full_path = self._resolve_path(mask_path)

        if full_path is None or not os.path.exists(full_path):
            print(f"Warning: Mask not found at {full_path}, using zero mask")
            return torch.zeros(
                self.train_resolution,
                self.train_resolution,
                dtype=torch.float32,
            )

        mask = Image.open(full_path).convert("L")

        mask = mask.resize(
            (self.train_resolution, self.train_resolution),
            Image.Resampling.NEAREST,
        )

        mask_np = np.array(mask).astype(np.float32)

        # Any non-zero value is considered anomalous/defective.
        mask_np = (mask_np > 0).astype(np.float32)

        return torch.from_numpy(mask_np)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        image_path = self._resolve_path(item["image"])
        if image_path is None or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")
        raw_image_size = raw_image.size  # (width, height)

        image = self._resize_image(raw_image)
        resized_image_size = image.size  # should be (train_resolution, train_resolution)

        messages = self._build_messages(item)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        labels = self._build_labels(input_ids)

        mask_path = self._get_mask_path(item)
        gt_mask = self._load_mask(mask_path)

        if idx < self.debug_first_n:
            grid_t, grid_h, grid_w = image_grid_thw[0].tolist()

            patch_size = getattr(self.processor.image_processor, "patch_size", 14)
            estimated_processor_h = int(grid_h * patch_size)
            estimated_processor_w = int(grid_w * patch_size)

            print(
                "[dataset-debug] "
                f"idx={idx} "
                f"image_path={image_path} "
                f"mask_path={mask_path} "
                f"raw_image_size={raw_image_size} "
                f"resized_image_size={resized_image_size} "
                f"train_resolution={self.train_resolution} "
                f"image_grid_thw={image_grid_thw.tolist()} "
                f"patch_size={patch_size} "
                f"estimated_processor_size=({estimated_processor_w}, {estimated_processor_h}) "
                f"pixel_values_shape={tuple(pixel_values.shape)} "
                f"input_ids_shape={tuple(input_ids.shape)} "
                f"attention_mask_shape={tuple(attention_mask.shape)} "
                f"labels_shape={tuple(labels.shape)} "
                f"gt_mask_shape={tuple(gt_mask.shape)} "
                f"gt_min={gt_mask.min().item():.4f} "
                f"gt_max={gt_mask.max().item():.4f} "
                f"gt_mean={gt_mask.mean().item():.8f} "
                f"gt_sum={gt_mask.sum().item():.2f}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "gt_segmentation_mask": gt_mask,
        }