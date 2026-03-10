<div align="center">

# It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models.

[Jaeha Choi](https://github.com/allchiever)<sup>1\*</sup>, [Jin Won Lee](https://github.com/jinleevv)<sup>2\*</sup>, Siwoo You<sup>1</sup>, Jangho Lee<sup>1†</sup>

<sup>\*</sup>Equal Contribution. <sup>†</sup>Corresponding Author.

<sup>1</sup>Incheon National University, Incheon, Republic of Korea  
 <sup>2</sup>McGill University, Montreal, Canada

### **CVPR 2026 Findings** 🔥

[![CVPR 2026 Findings](https://img.shields.io/badge/CVPR-2026%20Findings-8b0000.svg)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.08011-b31b1b.svg)](https://arxiv.org/abs/2603.08011)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)
[![Project Page](https://img.shields.io/badge/Project%20Page-Link-blue)](https://it-s-time-to-get-it-right.github.io/)

</div>

---

## Official GitHub

- **Project Page**: [https://it-s-time-to-get-it-right.github.io/](https://it-s-time-to-get-it-right.github.io/)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/jaeha-choi/TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)
- **GitHub Repository**: [https://github.com/allchiever/It-s-Time-to-Get-It-Right](https://github.com/allchiever/It-s-Time-to-Get-It-Right)

> **Dataset Notice**: This dataset is collected from publicly available data corpora. The copyright and redistribution conditions of the dataset do not belong to the authors of this project. Please refer to the respective source data corpora and the license information in the `annotations.json` file for details on usage, attribution, and redistribution terms.

---

## Quick Start

TickTockVQA supports training with multiple Vision-Language Models. Use the scripts below for quick LoRA fine-tuning.

### Supported Models

| Model                | Script                   | Base Model                                 |
| -------------------- | ------------------------ | ------------------------------------------ |
| **Llama 3.2 Vision** | `scripts/train_llama.py` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| **Gemma 3**          | `scripts/train_gemma.py` | `google/gemma-3-4b-it`                     |
| **Qwen2-VL**         | `scripts/train_qwen.py`  | `Qwen/Qwen2.5-VL-7B-Instruct`              |

### Prerequisites

```bash
pip install torch transformers peft deepspeed accelerate
```

### Training

**1. Llama 3.2 Vision**

```bash
python scripts/train_llama.py \
    --data_path /path/to/annotations.json \
    --image_folder /path/to/images \
    --output_dir output/llama_ticktockvqa
```

**2. Gemma 3**

```bash
python scripts/train_gemma.py \
    --data_path /path/to/annotations.json \
    --image_folder /path/to/images \
    --output_dir output/gemma_ticktockvqa
```

**3. Qwen2-VL**

```bash
python scripts/train_qwen.py \
    --data_path /path/to/annotations.json \
    --image_folder /path/to/images \
    --output_dir output/qwen_ticktockvqa
```

### Dataset

Download the TickTockVQA dataset from [Hugging Face](https://huggingface.co/datasets/jaeha-choi/TickTockVQA) or use the upload script:

```bash
python scripts/upload_to_huggingface.py --dataset_path ./data_pool/TickTockVQA_Official
```

---

## Citation

If you find our work useful, please cite:

```bibtex
@article{choi2026clockreasoning,
  title   = {It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models},
  author  = {Choi, Jaeha and Lee, Jin Won and You, Siwoo and Lee, Jangho},
  journal = {arXiv preprint arXiv:2603.08011},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.08011}
}
```

If you find our work useful, we would appreciate a **Hugging Face** 🤗 like and a **GitHub** ⭐ Star. Thank you!

---

## Links

- **arXiv**: [https://arxiv.org/abs/2603.08011](https://arxiv.org/abs/2603.08011)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/jaeha-choi/TickTockVQA](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)
- **Project Page**: [https://it-s-time-to-get-it-right.github.io/](https://it-s-time-to-get-it-right.github.io/)

---

## Star History


<a href="https://star-history.com/#allchiever/It-s-Time-to-Get-It-Right&Date"><img src="https://api.star-history.com/svg?repos=allchiever/It-s-Time-to-Get-It-Right&type=Date" width="400" alt="Star History Chart" /></a>
