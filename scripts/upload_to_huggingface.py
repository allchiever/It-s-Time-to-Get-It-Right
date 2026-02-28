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
        
        readme = f"""---
license: cc-by-4.0
task_categories:
- visual-question-answering
tags:
- ticktockvqa
- clock
- vqa
---

# TickTockVQA

Analog clock reading dataset. See [Project Page](https://YOUR_PROJECT_PAGE_URL) for details.

## Files

- `annotations.json`: Image annotations with time labels
"""
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
