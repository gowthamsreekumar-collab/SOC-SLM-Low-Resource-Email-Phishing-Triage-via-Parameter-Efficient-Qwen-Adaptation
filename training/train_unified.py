"""
Unified CLI Training Script for Qwen2.5 and Qwen1.5 SOC Incident Triage.
Supports LoRA (FP16/BF16) and QLoRA (4-bit NF4) fine-tuning on consumer GPUs and cloud VMs (Colab T4).

Usage:
    python training/train_unified.py --model_id Qwen/Qwen2.5-1.5B-Instruct --method qlora --epochs 3
    python training/train_unified.py --model_id Qwen/Qwen1.5-1.8B-Chat --method lora --epochs 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.trainer_manager import (
    SUPPORTED_MODELS,
    ALL_TARGET_MODULES,
    DEFAULT_TARGET_MODULES,
    run_training_job,
)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5 / Qwen1.5 with LoRA or QLoRA for SOC Email Triage")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Hugging Face model backbone identifier",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="qlora",
        choices=["lora", "qlora", "LoRA", "QLoRA"],
        help="Fine-tuning method: lora (FP16) or qlora (4-bit NF4)",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="AdamW initial learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence context length")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save adapter weights")
    parser.add_argument("--simulated", action="store_true", help="Run simulated training for rapid verification")

    args = parser.parse_args()

    norm_method = "QLoRA (4-bit NF4 Quantization)" if args.method.lower() == "qlora" else "LoRA (FP16 Base Adapter)"
    output_dir = Path(args.output_dir) if args.output_dir else None

    print("=" * 70)
    print("  SOC TRIAGE ASSISTANT - MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Model Backbone       : {args.model_id}")
    print(f"Training Paradigm    : {norm_method}")
    print(f"LoRA Rank (r)        : {args.lora_r}")
    print(f"LoRA Alpha           : {args.lora_alpha}")
    print(f"Learning Rate        : {args.learning_rate}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch Size (Effective): {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Max Sequence Length  : {args.max_seq_length}")
    print("=" * 70)

    def print_progress(data: dict):
        if data.get("status") == "training":
            print(f"[{data.get('percent'):5.1f}%] {data.get('message')}")
        elif data.get("status") == "completed":
            print(f"\n{data.get('message')}")

    results = run_training_job(
        model_id=args.model_id,
        method=norm_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        output_adapter_dir=output_dir,
        progress_callback=print_progress,
        force_simulation=args.simulated,
    )

    print("\nTraining summary:")
    print(f"- Total Time        : {results['total_train_time_sec']}s")
    print(f"- Final Train Loss  : {results['final_train_loss']}")
    print(f"- Final Eval Loss   : {results['final_eval_loss']}")
    print(f"- Peak VRAM         : {results['peak_gpu_mb']} MB")
    print(f"- Adapter Directory : {results['adapter_dir']}")
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
