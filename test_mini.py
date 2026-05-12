import json
import torch
from PIL import Image
import os
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt


def main():
    print("Loading processor...")
    # Load processor from the checkpoint directory to get the custom tokens
    processor = AutoProcessor.from_pretrained(
        "./checkpoints_mini/final", trust_remote_code=True, use_fast=False
    )

    print("Loading base model...")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "./model", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    print("Resizing token embeddings to match tokenizer...")
    model.resize_token_embeddings(len(processor.tokenizer))

    # Initialize the seg_projector if it's there
    if hasattr(model.model, "seg_projector"):
        model.model.seg_projector.load_state_dict(
            model.model.visual.merger.state_dict()
        )
        print("Initialized seg_projector with visual.merger weights")

    print("Loading adapter weights...")
    # Fix PEFT KeyError bug by temporarily removing seg_projector from modules_to_save in adapter config
    config_path = "./checkpoints_mini/final/adapter_config.json"
    with open(config_path, "r") as f:
        adapter_cfg = json.load(f)
    if "seg_projector" in adapter_cfg.get("modules_to_save", []):
        adapter_cfg["modules_to_save"].remove("seg_projector")
        with open(config_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)

    model = PeftModel.from_pretrained(
        model, "./checkpoints_mini/final", ensure_weight_tying=True
    )

    print("Loading seg_projector weights from adapter manually...")
    import safetensors.torch

    adapter_weights = safetensors.torch.load_file(
        "./checkpoints_mini/final/adapter_model.safetensors"
    )
    seg_proj_weights = {
        k.replace("base_model.model.model.seg_projector.", ""): v
        for k, v in adapter_weights.items()
        if "seg_projector" in k
    }
    if (
        hasattr(model.model, "seg_projector")
        and getattr(model.model, "seg_projector", None) is not None
    ):
        model.base_model.model.model.seg_projector.load_state_dict(seg_proj_weights)
        print("Loaded fine-tuned seg_projector weights")

    model.config.seg_token_idx = processor.tokenizer.convert_tokens_to_ids(
        "[SEG_DEFECT]"
    )
    model.config.seg_normal_token_idx = processor.tokenizer.convert_tokens_to_ids(
        "[SEG_NORMAL]"
    )
    print(
        f"SEG token IDs: defect={model.config.seg_token_idx}, normal={model.config.seg_normal_token_idx}"
    )

    model.eval()

    # Load data
    with open("data/anomaly_dataset/mini_5.json", "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples for testing")

    for i, item in enumerate(data):
        print(f"\n--- Sample {i + 1} ---")
        image_path = os.path.join("data", item["image"])
        image = Image.open(image_path).convert("RGB")

        # Use the question from the dataset
        question = (
            item["conversations"][0]["value"]
            .replace("<image>\\n", "")
            .replace("<image>", "")
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], padding=False, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"Question: {question}")

        # Generate text + anomaly map in one call
        output, anomaly_map = model.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], output.sequences)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False
        )[0]

        print(f"Prediction: {output_text.strip()}")
        print(f"Ground Truth: {item['conversations'][1]['value']}")

        # anomaly_map is already here — no second call needed
        if anomaly_map is not None:
            defect_map = (
                anomaly_map[0, 1, :, :].unsqueeze(0).unsqueeze(0)
            )  # [1, 1, H, W]

            orig_w, orig_h = image.size
            upsampled_map = F.interpolate(
                defect_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ).squeeze()

            heatmap = upsampled_map.cpu().float().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            plt.title("Predicted Anomaly Map")
            plt.axis("off")

            os.makedirs("results", exist_ok=True)
            save_path = f"results/sample_{i + 1}_heatmap.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved anomaly map to {save_path}")
        else:
            print("No anomaly map generated (SEG tokens likely not outputted).")


if __name__ == "__main__":
    main()
