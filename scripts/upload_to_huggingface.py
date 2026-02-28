#!/usr/bin/env python3
"""
Upload TickTockVQA dataset to Hugging Face Hub.
Requires: pip install huggingface_hub datasets
Login: huggingface-cli login
"""

import argparse
import io
import json
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
    from datasets import Dataset, DatasetDict, Image
except ImportError:
    print("Install: pip install huggingface_hub datasets")
    raise


def load_annotations(annotations_path: Path) -> list:
    with open(annotations_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Upload TickTockVQA to Hugging Face")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to TickTockVQA_Official folder (contains annotations.json)")
    parser.add_argument("--repo_id", default="YOUR_USERNAME/TickTockVQA",
                        help="Hugging Face repo ID (e.g., username/TickTockVQA)")
    parser.add_argument("--github_url", default="https://github.com/allchiever/It-s-Time-to-Get-It-Right",
                        help="GitHub repository URL")
    parser.add_argument("--project_page_url", default="https://it-s-time-to-get-it-right.github.io/",
                        help="Project page URL")
    parser.add_argument("--arxiv_url", default="",
                        help="arXiv paper URL (e.g., https://arxiv.org/abs/XXXX.XXXXX)")
    parser.add_argument("--include_images", action="store_true",
                        help="Include images in upload (requires images folder)")
    parser.add_argument("--images_folder", type=str, default=None,
                        help="Path to images folder (default: dataset_path/../images)")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    annotations_file = dataset_path / "annotations.json"
    
    if not annotations_file.exists():
        print(f"[ERROR] annotations.json not found at {annotations_file}")
        return 1
    
    print(f"[INFO] Loading annotations from {annotations_file}")
    data = load_annotations(annotations_file)
    
    if args.include_images:
        # Build dataset with images (slower, larger)
        images_folder = Path(args.images_folder) if args.images_folder else dataset_path.parent / "images"
        if not images_folder.exists():
            print(f"[WARNING] Images folder not found: {images_folder}")
            print("  Uploading annotations only.")
            args.include_images = False
    
    if not args.include_images:
        # Upload annotations only (fast)
        api = HfApi()
        try:
            create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"[INFO] Repo may already exist: {e}")
        
        api.upload_file(
            path_or_fileobj=str(annotations_file),
            path_in_repo="annotations.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        
        # Build README with optional links
        badges = []
        if args.arxiv_url:
            badges.append(f"[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]({args.arxiv_url})")
        badges.append(f"[![GitHub](https://img.shields.io/badge/GitHub-Code-181717?logo=github)]({args.github_url})")
        if args.project_page_url:
            badges.append(f"[![Project](https://img.shields.io/badge/Project%20Page-Link-0066cc.svg)]({args.project_page_url})")
        badges.append(f"[![Hugging Face](https://img.shields.io/badge/🤗%20Dataset-View-yellow?logo=huggingface)](https://huggingface.co/datasets/{args.repo_id})")
        badge_line = " ".join(badges)

        readme = f"""---
license: cc-by-4.0
task_categories:
- visual-question-answering
tags:
- ticktockvqa
- clock
- vqa
- analog-clock
- visual-question-answering
---

# 🕐 TickTockVQA

**Analog clock reading dataset** from *It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models* (CVPR 2026 Findings). Contains 12,483 images with time labels for training and evaluating VQA models on analog clock understanding.

{badge_line}

---

## 📖 Overview

TickTockVQA is a benchmark dataset for reading analog clocks from images. Each sample includes an image and the ground-truth time displayed on the most prominent clock face.

| Split | Samples |
|-------|---------|
| Train | 7,236 |
| Test  | 5,247 |

**Sources:** OpenImages, COCO, ClockMovies, VisualGenome, CC12M, ImageNet, SBU

---

## 📁 Files

| File | Description |
|------|--------------|
| `annotations.json` | Image annotations with `image_name`, `image_path`, `time_string`, `hour`, `minute`, license info |
| `dataset_statistics.json` | Dataset statistics (source distribution, splits, license distribution) |

---

## 🚀 Quick Start

```python
from datasets import load_dataset
dataset = load_dataset("{args.repo_id}")
```

Or download annotations only:
```bash
huggingface-cli download {args.repo_id} annotations.json --local-dir ./data
```

---

## 📚 Citation

```bibtex
@inproceedings{{ticktockvqa2026}},
  title={{It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models}},
  author={{}},
  booktitle={{CVPR}},
  year={{2026}}
}}
```

---

## 🔗 Links

- **GitHub**: [{args.github_url}]({args.github_url})
"""
        if args.project_page_url:
            readme += f"- **Project Page**: [{args.project_page_url}]({args.project_page_url})\n"
        if args.arxiv_url:
            readme += f"- **arXiv**: [{args.arxiv_url}]({args.arxiv_url})\n"
        readme += "\n"
        api.upload_file(
            path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"[SUCCESS] Uploaded to https://huggingface.co/datasets/{args.repo_id}")
        return 0
    
    # With images: create Dataset and push
    image_paths = []
    labels = []
    for item in data:
        img_path = images_folder / item.get("image_path", item.get("image_name", ""))
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(item.get("time_string", ""))
    
    dataset = Dataset.from_dict({
        "image": image_paths,
        "label": labels,
    })
    dataset = dataset.cast_column("image", Image())
    dataset.push_to_hub(args.repo_id)
    print(f"[SUCCESS] Uploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
