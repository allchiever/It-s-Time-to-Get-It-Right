#!/usr/bin/env python3
"""
Quick Start: TickTockVQA LoRA training with Qwen2-VL
Supports: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_vlm_path() -> Path:
    """Find Qwen VLM finetune code path."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    vlm_path = os.environ.get("QWEN_VLM_PATH")
    if vlm_path and Path(vlm_path).exists():
        return Path(vlm_path)
    
    default = repo_root.parent / "VLMs" / "Qwen2-VL-Finetune"
    if default.exists():
        return default
    
    alt = repo_root / "VLMs" / "Qwen2-VL-Finetune"
    if alt.exists():
        return alt
    
    return default


def main():
    parser = argparse.ArgumentParser(description="TickTockVQA LoRA training - Qwen2-VL")
    parser.add_argument("--data_path", required=True, help="Path to training annotations JSON")
    parser.add_argument("--image_folder", required=True, help="Path to image folder")
    parser.add_argument("--output_dir", default="output/qwen_ticktockvqa", help="Output directory")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen/Qwen2.5-VL-7B-Instruct or Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_per_device", type=int, default=8)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--vlm_path", type=str, default=None)
    args = parser.parse_args()
    
    vlm_path = Path(args.vlm_path) if args.vlm_path else get_vlm_path()
    if not vlm_path.exists():
        print(f"[ERROR] Qwen VLM code not found at {vlm_path}")
        print("  Set QWEN_VLM_PATH env var or use --vlm_path")
        sys.exit(1)
    
    train_script = vlm_path / "src" / "train" / "train_sft.py"
    if not train_script.exists():
        print(f"[ERROR] train_sft.py not found at {train_script}")
        sys.exit(1)
    
    num_gpus = args.num_gpus
    if num_gpus is None:
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        num_gpus = len([x for x in cuda_dev.split(",") if x.strip()]) if cuda_dev else 1
    
    grad_accum = max(1, args.global_batch_size // (args.batch_per_device * num_gpus))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vlm_path / 'src'}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        "deepspeed",
        "--num_gpus", str(num_gpus),
        str(train_script),
        "--use_liger", "True",
        "--lora_enable", "True",
        "--use_dora", "False",
        "--lora_namespan_exclude", "['lm_head', 'embed_tokens']",
        "--lora_rank", "64",
        "--lora_alpha", "64",
        "--lora_dropout", "0.05",
        "--num_lora_modules", "-1",
        "--deepspeed", str(vlm_path / "scripts" / "zero3.json"),
        "--model_id", args.model_name,
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--output_dir", args.output_dir,
        "--remove_unused_columns", "False",
        "--freeze_vision_tower", "False",
        "--freeze_llm", "True",
        "--freeze_merger", "False",
        "--bf16", "True",
        "--disable_flash_attn2", "True",
        "--num_train_epochs", str(args.num_epochs),
        "--per_device_train_batch_size", str(args.batch_per_device),
        "--gradient_accumulation_steps", str(grad_accum),
        "--image_min_pixels", str(256 * 28 * 28),
        "--image_max_pixels", str(1280 * 28 * 28),
        "--learning_rate", "1e-4",
        "--merger_lr", "1e-5",
        "--vision_lr", "2e-6",
        "--weight_decay", "0.1",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--gradient_checkpointing", "True",
        "--report_to", "tensorboard",
        "--lazy_preprocess", "True",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--save_total_limit", "10",
        "--dataloader_num_workers", "4",
    ]
    
    print(f"[INFO] VLM path: {vlm_path}")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
