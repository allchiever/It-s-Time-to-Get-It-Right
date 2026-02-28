# It's Time to Get It Right. CVPR Findings Official Github

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/jaeha-choi/TickTockVQA)
[![Project Page](https://img.shields.io/badge/Project%20Page-Link-blue)](https://YOUR_PROJECT_PAGE_URL)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=allchiever/It-s-Time-to-Get-It-Right&type=Date)](https://star-history.com/#allchiever/It-s-Time-to-Get-It-Right&Date)

> The star history chart above is automatically updated via [Star History API](https://www.star-history.com/). No manual refresh needed.

---

## Quick Start

TickTockVQA supports training with multiple Vision-Language Models. Use the scripts below for quick LoRA fine-tuning.

### Supported Models

| Model | Script | Base Model |
|-------|--------|------------|
| **Llama 3.2 Vision** | `scripts/train_llama.py` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| **Gemma 3** | `scripts/train_gemma.py` | `google/gemma-3-4b-it` |
| **Qwen2-VL** | `scripts/train_qwen.py` | `Qwen/Qwen2.5-VL-7B-Instruct` |

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

```bibtex
@inproceedings{},
  title={},
  author={},
  booktitle={CVPR},
  year={}
}
```

---

## Links

- **arXiv**: [Add your paper link]
- **Hugging Face Dataset**: [Add your dataset link]
- **Project Page**: [Add your project page link]
