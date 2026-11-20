"""
LoRA (FP16) Fine-Tuning Script for Qwen2.5-1.5B-Instruct on SOC Triage Dataset.

Hardware Target: NVIDIA Tesla T4 (Colab)
Precision: FP16 base model + LoRA adapters (unquantized base)
LoRA: r=16, alpha=32, dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

IS_COLAB = Path("/content").exists()
BASE_DIR = Path("/content") if IS_COLAB else Path(".")

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = BASE_DIR / "checkpoints/lora"
ADAPTER_SAVE_DIR = BASE_DIR / "model_checkpoints/lora_adapter"
METRICS_SAVE_DIR = BASE_DIR / "experiments"
MAX_SEQ_LENGTH = 1024
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = 0.05
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRAD_ACCUM", 4))
EPOCHS = int(os.environ.get("EPOCHS", 3))
EVAL_STEPS = 20
LOGGING_STEPS = 5


class MetricsTrackerCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.gpu_memory_log = []
        self.snapshots = {}
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        peak_mb = 0
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.gpu_memory_log.append(peak_mb)

        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)

            # Snapshots at ~33%, ~66%, ~100%
            progress = state.global_step / max(1, self.total_steps)
            if "33%" not in self.snapshots and progress >= 0.33:
                self.snapshots["33%"] = {
                    "step": state.global_step,
                    "train_loss": logs["loss"],
                    "eval_loss": self.eval_losses[-1] if self.eval_losses else None,
                    "gpu_peak_mb": peak_mb,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            elif "66%" not in self.snapshots and progress >= 0.66:
                self.snapshots["66%"] = {
                    "step": state.global_step,
                    "train_loss": logs["loss"],
                    "eval_loss": self.eval_losses[-1] if self.eval_losses else None,
                    "gpu_peak_mb": peak_mb,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        if "100%" not in self.snapshots and self.train_losses:
            self.snapshots["100%"] = {
                "step": state.global_step,
                "train_loss": self.train_losses[-1],
                "eval_loss": self.eval_losses[-1] if self.eval_losses else None,
                "gpu_peak_mb": peak_mb,
                "timestamp": datetime.utcnow().isoformat(),
            }


def load_jsonl_dataset(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_dataset_for_chat(records, tokenizer, max_length=MAX_SEQ_LENGTH):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in records:
        messages = item["messages"]
        prompt_and_response = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = tokenizer(
            prompt_and_response,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = list(input_ids)

        # Mask prompt tokens in labels so loss is only calculated on assistant output
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_only = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer(
            prompt_only,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        prompt_len = min(len(prompt_tokens), len(labels))
        for i in range(prompt_len):
            labels[i] = -100  # PyTorch ignore_index

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })


def main():
    print("=" * 65)
    print("  SOC TRIAGE ASSISTANT - LoRA (FP16) FINE-TUNING")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is required for LoRA fine-tuning. Exiting.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")

    # 1. Directories setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Locate dataset files
    train_path = Path("/content/dataset/train.jsonl")
    val_path = Path("/content/dataset/validation.jsonl")

    if not train_path.exists():
        train_path = Path("dataset/train.jsonl")
        val_path = Path("dataset/validation.jsonl")

    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Dataset splits not found at {train_path} or {val_path}")
        sys.exit(1)

    print(f"Loading training data from: {train_path}")
    raw_train = load_jsonl_dataset(train_path)
    print(f"Loading validation data from: {val_path}")
    raw_val = load_jsonl_dataset(val_path)
    print(f"Dataset sizes: Train = {len(raw_train)}, Val = {len(raw_val)}")

    # 3. Load Tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Format Datasets
    print("Formatting and tokenizing datasets with chat template...")
    train_dataset = format_dataset_for_chat(raw_train, tokenizer)
    val_dataset = format_dataset_for_chat(raw_val, tokenizer)

    # 5. Load Base Model in FP16 (No 4-bit Quantization)
    print(f"Loading base model in FP16: {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # 6. Configure LoRA Adapter
    print(f"Configuring LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT} ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 7. Calculate total steps
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * EPOCHS
    print(f"Training schedule: {EPOCHS} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    # 8. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        warmup_steps=12,
        weight_decay=0.01,
        seed=42,
    )

    tracker = MetricsTrackerCallback(total_steps=total_steps)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[tracker],
    )

    # 9. Execute Training
    print("\nStarting LoRA (FP16) training...")
    torch.cuda.reset_peak_memory_stats()
    train_start_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - train_start_time

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"\n[DONE] LoRA training completed in {total_train_time:.1f}s ({total_train_time/60:.2f} mins)")
    print(f"Peak GPU VRAM during training: {peak_gpu_mb:.2f} MB ({peak_gpu_mb/1024:.2f} GB)")

    # 10. Save Adapter & Tokenizer
    print(f"Saving LoRA adapter to: {ADAPTER_SAVE_DIR} ...")
    model.save_pretrained(str(ADAPTER_SAVE_DIR))
    tokenizer.save_pretrained(str(ADAPTER_SAVE_DIR))

    # 11. Save Training Logs & Metrics
    log_data = {
        "model": "LoRA",
        "precision": "FP16",
        "train_losses": tracker.train_losses,
        "eval_losses": tracker.eval_losses,
        "train_steps": tracker.train_steps,
        "eval_steps": tracker.eval_steps,
        "gpu_memory_log": tracker.gpu_memory_log,
        "snapshots": tracker.snapshots,
        "total_train_time_sec": total_train_time,
        "peak_gpu_mb": peak_gpu_mb,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "hyperparameters": {
            "model_id": MODEL_ID,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "epochs": EPOCHS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "compute_dtype": "float16",
        },
    }

    log_path = METRICS_SAVE_DIR / "lora_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to: {log_path}")

    # Also save to local path if different
    local_log_path = Path("experiments/lora_training_log.json")
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print("\n=== LoRA Training Pipeline Finished Successfully ===")


if __name__ == "__main__":
    main()
