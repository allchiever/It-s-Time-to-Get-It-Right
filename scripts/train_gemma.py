#!/usr/bin/env python3
"""
Quick Start: TickTockVQA LoRA training with Gemma 3
Supports: google/gemma-3-4b-it
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_vlm_path() -> Path:
    """Find Gemma VLM finetune code path."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    
    vlm_path = os.environ.get("GEMMA_VLM_PATH")
    if vlm_path and Path(vlm_path).exists():
        return Path(vlm_path)
    
    default = repo_root.parent / "VLMs" / "Gemma3-Finetune"
    if default.exists():
        return default
    
    alt = repo_root / "VLMs" / "Gemma3-Finetune"
    if alt.exists():
        return alt
    
    return default


def main():
    parser = argparse.ArgumentParser(description="TickTockVQA LoRA training - Gemma 3")
    parser.add_argument("--data_path", required=True, help="Path to training annotations JSON")
    parser.add_argument("--image_folder", required=True, help="Path to image folder")
    parser.add_argument("--output_dir", default="output/gemma_ticktockvqa", help="Output directory")
    parser.add_argument("--model_name", default="google/gemma-3-4b-it")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_per_device", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--vlm_path", type=str, default=None)
    args = parser.parse_args()
    
    vlm_path = Path(args.vlm_path) if args.vlm_path else get_vlm_path()
    if not vlm_path.exists():
        print(f"[ERROR] Gemma VLM code not found at {vlm_path}")
        print("  Set GEMMA_VLM_PATH env var or use --vlm_path")
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
        "--lora_enable", "True",
        "--vision_lora", "False",
        "--use_dora", "False",
        "--lora_rank", "64",
        "--lora_alpha", "64",
        "--lora_dropout", "0.05",
        "--lora_namespan_exclude", "['lm_head', 'embed_tokens']",
        "--num_lora_modules", "-1",
        "--use_liger", "True",
        "--deepspeed", str(vlm_path / "scripts" / "zero3.json"),
        "--model_id", args.model_name,
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--output_dir", args.output_dir,
        "--disable_flash_attn2", "True",
        "--freeze_projector", "False",
        "--freeze_vision_tower", "False",
        "--freeze_llm", "True",
        "--bf16", "True",
        "--num_train_epochs", str(args.num_epochs),
        "--per_device_train_batch_size", str(args.batch_per_device),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", "1e-4",
        "--projector_lr", "1e-5",
        "--vision_lr", "2e-6",
        "--weight_decay", "0.1",
        "--warmup_ratio", "0.03",
        "--adam_beta2", "0.95",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--gradient_checkpointing", "True",
        "--report_to", "tensorboard",
        "--lazy_preprocess", "True",
        "--save_steps", "500",
        "--save_total_limit", "10",
        "--dataloader_num_workers", "4",
    ]
    
    print(f"[INFO] VLM path: {vlm_path}")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
