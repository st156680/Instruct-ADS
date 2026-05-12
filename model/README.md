---
base_model:
- Qwen/Qwen3-8B-Base
- DeepGlint-AI/rice-vit-large-patch14-560
datasets:
- lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M
- lmms-lab/LLaVA-OneVision-1.5-Insturct-Data
- HuggingFaceM4/FineVision
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
---

<div align="center">

<h1>LLaVA-OneVision-1.5: Fully Open-Source State-of-the-Art VLM Model</h1>


<p>
  <a href="https://huggingface.co/papers/2509.23661">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arXiv&logoColor=white">
  </a>
  <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5">
    <img alt="Code" src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M">
    <img alt="Mid-Training Dataset" src="https://img.shields.io/badge/Mid--Training%20Dataset-f59e0b?style=for-the-badge&logo=huggingface&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data">
    <img alt="Instruct Dataset" src="https://img.shields.io/badge/Instruct%20Dataset-3fb950?style=for-the-badge&logo=huggingface&logoColor=white">
  </a>
  <a href="https://huggingface.co/spaces/lmms-lab/LLaVA-OneVision-1.5">
    <img alt="Demo" src="https://img.shields.io/badge/Demo-1f6feb?style=for-the-badge&logo=huggingface&logoColor=white">
  </a>
  <a href="https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/tensorboard">
    <img alt="TensorBoard" src="https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  </a>
</p>

</div>



## Introduction

LLaVA-OneVision-1.5 is a fully open-source family of large multimodal models (LMMs) built to democratize multimodal training. Trained on native‑resolution images, it delivers state‑of‑the‑art performance at substantially lower cost. The project also releases high‑quality pretraining and SFT data, a complete and efficient training framework with recipes and configs, and comprehensive logs to support transparent, reproducible research.
#### **Superior Performance**
  - The model leads on multiple multimodal benchmarks and generally surpasses Qwen2.5-VL.
  - Training on native-resolution images significantly improves its visual understanding.

#### **High-Quality Data at Scale**
  - The pretraining corpus comprises large-scale, concept-balanced, diverse, and high-quality captions curated with strict filtering and quality control.
  - The instruction-tuning dataset is comprehensive and covers a wide range of tasks.

#### **Ultra-Efficient Training Framework**
  - The end-to-end training cost is about $16,000 on A100 GPUs at roughly $0.60 per GPU-hour.
  - The system is built on Megatron-LM with support for MoE, FP8, and long-sequence parallelism, and the codebase is optimized for cost-effective scaling.

#### **Fully Open Framework**
  - The project releases high-quality pretraining and SFT datasets along with the complete training framework, configurations, and recipes.
  - It also provides detailed training logs and metrics to enable reproducibility and community adoption.


## Models

| Model                    | HF Link                                                                                      | Training Log |
|--------------------------|--------------------------------------------------------------------------------------------------------|-------------|
| LLaVA-OV-1.5-4B-Instruct | [🤗 HF / 4B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct)                | [📈 Tensorboard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/tensorboard) |
| LLaVA-OV-1.5-8B-Instruct | [🤗 HF / 8B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)                | [📈 Tensorboard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/tensorboard) |


## Datasets

| Description        | Link                                                                                                   | Status      |
|--------------------|--------------------------------------------------------------------------------------------------------|-------------|
| LLaVA-OneVision-1.5-Mid-Training-85M   | [🤗HF / Mid-Training 85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M) | Uploading…  |
| LLaVA-OneVision-1.5-Instruct           | [🤗HF / Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)        | Available  |


## Evaluation Results
All evaluations were conducted using [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

|                                  | **LLaVA-OV-1.5-8B** | **Qwen2.5 VL 7B** |
|:----------------------------------|:---------------:|:-------------:|
| MMMU (Validation)                 |    **55.44**    |     51.33     |
| MMMU-Pro (Standard)               |    **37.40**    |     36.30     |
| MMMU-Pro (Vision)                 |      25.15      |   **32.83**   |
| MMBench (English; Test)           |    **84.14**    |     83.40     |
| MMBench (Chinese; Test)           |      81.00      |   **81.61**   |
| MME-RealWorld (English)           |    **62.31**    |     57.33     |
| MME-RealWorld (Chinese)           |    **56.11**    |     51.50     |
| AI2D (With Mask)                  |    **84.16**    |     82.58     |
| AI2D (Without Mask)               |    **94.11**    |     93.36     |
| CV-Bench                          |    **80.82**    |     79.95     |
| VL-RewardBench                    |      45.90      |   **49.65**   |
| V*                                |    **78.01**    |     76.96     |
| PixmoCount                        |      62.19      |   **63.33**   |
| CountBench                        |    **88.19**    |     86.35     |
| ChartQA                           |    **86.48**    |     84.08     |
| CharXiv (Direct Questions)        |    **74.10**    |     69.80     |
| DocVQA (Test)                     |    **95.00**    |     94.93     |
| InfoVQA (Test)                    |      78.42      |   **81.67**   |
| WeMath                            |    **33.62**    |     33.33     |
| MathVista (Mini)                  |    **69.57**    |     68.60     |
| MathVision                        |    **25.56**    |     22.37     |
| MMStar                            |    **67.72**    |     62.54     |
| SEED-Bench (Image)                |      77.32      |   **77.53**   |
| ScienceQA                         |    **94.98**    |     88.75     |
| SEED-Bench 2-Plus                 |      69.21      |   **70.93**   |
| OCRBench                          |      82.90      |   **84.20**   |
| RealWorldQA                       |      68.10      |   **68.50**   |

### Using 🤗  Transformers to Chat
Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_vl_utils`:

```python
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## Citation

If you find *LLaVA-OneVision-1.5* useful in your research, please consider to cite the following related papers:

```
@misc{an2025llavaonevision15fullyopenframework,
      title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training}, 
      author={Xiang An and Yin Xie and Kaicheng Yang and Wenkang Zhang and Xiuwei Zhao and Zheng Cheng and Yirui Wang and Songcen Xu and Changrui Chen and Chunsheng Wu and Huajie Tan and Chunyuan Li and Jing Yang and Jie Yu and Xiyao Wang and Bin Qin and Yumeng Wang and Zizhen Yan and Ziyong Feng and Ziwei Liu and Bo Li and Jiankang Deng},
      year={2025},
      eprint={2509.23661},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.23661}, 
}

@inproceedings{xie2025region,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong and Miles, Roy and Elezi, Ismail and Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}

@article{lillava,
  title={LLaVA-OneVision: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Zhang, Peiyuan and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={Transactions on Machine Learning Research}
  year={2024}
}
```


## Acknowledgement

We extend our sincere gratitude to **AIAK team of the** [**Baige AI computing platform**](https://cloud.baidu.com/product/aihc.html) **from Baidu AI Cloud** for providing the exceptional training framework. The outstanding capabilities of AIAK-Training-LLM and AIAK-Megatron have significantly accelerated our training process with remarkable efficiency. These cutting-edge frameworks have been instrumental in achieving our research goals. `To get full AIAK support, you can contact Baidu Cloud.` 

We acknowledge the support of [Synvo AI](https://synvo.ai/) for contributing to the partial data annotation in this work, and also thank the maintainers and contributors of the following open-source projects, whose work greatly inspired and supported our research:

- LLaVA: Large Language-and-Vision Assistant — [LLaVA](https://github.com/haotian-liu/LLaVA)
- LLaVA-NeXT: Next-generation multi-modal assistant — [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- lmms-eval: A standardized evaluation framework for Large Multimodal Models — [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- Megatron-LM: Efficient, scalable training for large language models — [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Qwen2.5-VL: Strong vision-language foundation model — [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- InternVL: Open-source large-scale vision-language foundation model — [InternVL](https://github.com/OpenGVLab/InternVL)
- Qwen3: Next-generation Qwen LLM — [Qwen](https://github.com/QwenLM/Qwen)
- MetaCLIP: Scalable contrastive pretraining — [MetaCLIP](https://github.com/facebookresearch/MetaCLIP)
- FineVision: Open Data Is All You Need — [FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision)