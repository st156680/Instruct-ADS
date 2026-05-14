"""
Custom data collator for anomaly detection training.

Handles:
- padding variable-length text tensors
- concatenating variable-length pixel_values
- concatenating image_grid_thw
- resizing/stacking segmentation masks to a fixed training resolution

Expected sample fields from dataset.py:
    input_ids:              [seq_len]
    attention_mask:         [seq_len]
    labels:                 [seq_len]
    pixel_values:           [num_patches, patch_dim]
    image_grid_thw:         [1, 3]
    gt_segmentation_mask:   [H, W]

Recommended mask_size for the current setup:
    (1008, 1008)

Why 1008?
    1008 / patch_size 14 = 72 visual patches
    72 / spatial_merge_size 2 = 36 anomaly-map cells
"""

from typing import List, Dict, Any, Tuple, Union, Optional

import torch
import torch.nn.functional as F


class AnomalyDataCollator:
    """
    Collates individual dataset samples into batches.

    Args:
        pad_token_id:
            Token ID used for padding input_ids.

        mask_size:
            Target segmentation mask size as int or tuple.
            If int, converted to (mask_size, mask_size).
            Recommended: 1008.

        debug_first_n:
            Print debug information for the first N collator calls.
    """

    def __init__(
        self,
        pad_token_id: int = 151643,
        mask_size: Union[int, Tuple[int, int]] = (1008, 1008),
        debug_first_n: int = 5,
    ):
        self.pad_token_id = int(pad_token_id)

        if isinstance(mask_size, int):
            self.mask_size = (int(mask_size), int(mask_size))
        else:
            self.mask_size = (int(mask_size[0]), int(mask_size[1]))

        self.debug_first_n = int(debug_first_n)
        self._num_calls = 0

    def _pad_1d(
        self,
        tensor: torch.Tensor,
        target_len: int,
        value: int,
    ) -> torch.Tensor:
        """
        Right-pad a 1D tensor to target_len.
        """
        pad_len = target_len - tensor.shape[0]
        if pad_len < 0:
            raise ValueError(
                f"Cannot pad tensor of length {tensor.shape[0]} to shorter length {target_len}"
            )

        return F.pad(tensor, (0, pad_len), value=value)

    def _resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Resize a binary mask to self.mask_size using nearest-neighbor interpolation.

        Args:
            mask: [H, W]

        Returns:
            [mask_h, mask_w], float32, values in {0, 1}
        """
        if mask.dim() != 2:
            raise ValueError(
                f"Expected gt_segmentation_mask with shape [H, W], got {tuple(mask.shape)}"
            )

        mask = mask.float()

        if tuple(mask.shape[-2:]) != self.mask_size:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            mask = F.interpolate(
                mask,
                size=self.mask_size,
                mode="nearest",
            )

            mask = mask.squeeze(0).squeeze(0)  # [mask_h, mask_w]

        # Keep masks strictly binary in case interpolation/source introduced odd values.
        mask = (mask > 0.5).float()

        return mask

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) == 0:
            raise ValueError("AnomalyDataCollator received an empty feature list")

        self._num_calls += 1

        # ---------------------------------------------------------------------
        # Text padding
        # ---------------------------------------------------------------------
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for f in features:
            input_ids = f["input_ids"].long()
            attention_mask = f["attention_mask"].long()
            labels = f["labels"].long()

            input_ids_list.append(
                self._pad_1d(
                    input_ids,
                    target_len=max_len,
                    value=self.pad_token_id,
                )
            )

            attention_mask_list.append(
                self._pad_1d(
                    attention_mask,
                    target_len=max_len,
                    value=0,
                )
            )

            labels_list.append(
                self._pad_1d(
                    labels,
                    target_len=max_len,
                    value=-100,
                )
            )

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        labels = torch.stack(labels_list, dim=0)

        # ---------------------------------------------------------------------
        # Vision tensors
        # ---------------------------------------------------------------------
        pixel_values = torch.cat(
            [f["pixel_values"] for f in features],
            dim=0,
        )

        image_grid_thw = torch.cat(
            [f["image_grid_thw"].long() for f in features],
            dim=0,
        )

        # Optional consistency check:
        # pixel_values rows should equal sum(t * h * w) over image_grid_thw.
        expected_num_patches = int(torch.prod(image_grid_thw, dim=1).sum().item())
        actual_num_patches = int(pixel_values.shape[0])

        if actual_num_patches != expected_num_patches:
            raise ValueError(
                "pixel_values/image_grid_thw mismatch: "
                f"pixel_values.shape[0]={actual_num_patches}, "
                f"sum(prod(image_grid_thw))={expected_num_patches}, "
                f"image_grid_thw={image_grid_thw.tolist()}"
            )

        # ---------------------------------------------------------------------
        # Segmentation masks
        # ---------------------------------------------------------------------
        masks = []

        for f in features:
            mask = f["gt_segmentation_mask"]
            mask = self._resize_mask(mask)
            masks.append(mask)

        gt_segmentation_masks = torch.stack(masks, dim=0)  # [B, H, W]

        # ---------------------------------------------------------------------
        # Debug logging
        # ---------------------------------------------------------------------
        if self._num_calls <= self.debug_first_n:
            mask_min = gt_segmentation_masks.min().item()
            mask_max = gt_segmentation_masks.max().item()
            mask_mean = gt_segmentation_masks.float().mean().item()
            mask_sum = gt_segmentation_masks.float().sum().item()

            print(
                "[collator-debug] "
                f"call={self._num_calls} "
                f"batch_size={len(features)} "
                f"max_seq_len={max_len} "
                f"input_ids_shape={tuple(input_ids.shape)} "
                f"attention_mask_shape={tuple(attention_mask.shape)} "
                f"labels_shape={tuple(labels.shape)} "
                f"pixel_values_shape={tuple(pixel_values.shape)} "
                f"image_grid_thw_shape={tuple(image_grid_thw.shape)} "
                f"image_grid_thw={image_grid_thw.tolist()} "
                f"expected_num_patches={expected_num_patches} "
                f"gt_segmentation_masks_shape={tuple(gt_segmentation_masks.shape)} "
                f"mask_size={self.mask_size} "
                f"mask_min={mask_min:.4f} "
                f"mask_max={mask_max:.4f} "
                f"mask_mean={mask_mean:.8f} "
                f"mask_sum={mask_sum:.2f}"
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
            "gt_segmentation_masks": gt_segmentation_masks,
        }

        return batch