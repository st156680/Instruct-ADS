# test_load.py
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./model", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("./model", trust_remote_code=True)
print("Loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
