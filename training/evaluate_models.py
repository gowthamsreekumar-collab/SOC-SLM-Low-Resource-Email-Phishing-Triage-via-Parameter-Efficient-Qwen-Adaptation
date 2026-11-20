"""
Comprehensive Model Evaluation Script for SOC Triage Assistant.
Evaluates:
  1. Base Model (Zero-shot Qwen2.5-1.5B-Instruct)
  2. LoRA Fine-Tuned Model
  3. QLoRA Fine-Tuned Model (4-bit NF4)

Computes:
  - Accuracy, Precision, Recall, Macro F1, Confusion Matrix
  - False Positive Rate
  - Hallucination / Unsupported Claim Rate
  - Abstention Accuracy & Failure Rate
  - Inference Latency & Peak GPU VRAM
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_DATASET_PATH = Path("/content/dataset/test.jsonl")
OUTPUT_DIR = Path("/content/experiments")

CLASSES = ["benign", "phishing", "suspicious"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def load_test_samples(path):
    if not path.exists():
        fallback = Path("dataset/test.jsonl")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Test dataset not found at {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def extract_ground_truth(sample):
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            try:
                data = json.loads(msg["content"])
                return data
            except Exception:
                pass
    return None

def extract_email_text(sample):
    user_msg = ""
    for msg in sample["messages"]:
        if msg["role"] == "user":
            user_msg = msg["content"]
            break
    return user_msg

def build_prompt_messages(sample):
    return [m for m in sample["messages"] if m["role"] != "assistant"]

def parse_model_json(text):
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text), True
    except Exception:
        pass

    # Extract markdown code blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), True
        except Exception:
            pass

    # Extract outer-most JSON object { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), True
        except Exception:
            pass

    # Fallback heuristic extraction
    fallback = {"classification": "unknown", "severity": "unknown", "raw_output": text}
    for c in CLASSES:
        if c in text.lower():
            fallback["classification"] = c
            break
    return fallback, False

def check_hallucination(pred_json, email_text):
    evidence_items = pred_json.get("evidence", [])
    if not evidence_items or not isinstance(evidence_items, list):
        return 0.0

    unsupported_count = 0
    email_lower = email_text.lower()
    for item in evidence_items:
        if not isinstance(item, str):
            continue
        words = [w for w in re.findall(r"\b\w{4,}\b", item.lower()) if w not in ["email", "contains", "detected", "suspicious", "threat", "indicators", "significant"]]
        if words and not any(w in email_lower for w in words):
            unsupported_count += 1

    return unsupported_count / max(1, len(evidence_items))

def evaluate_model_pipeline(model_variant, model, tokenizer, test_samples):
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_variant}")
    print(f"{'='*60}")

    y_true = []
    y_pred = []
    latencies = []
    parse_failures = 0
    hallucination_scores = []
    abstention_correct = 0
    abstention_total = 0

    torch.cuda.reset_peak_memory_stats()
    start_total_time = time.time()

    for idx, sample in enumerate(test_samples):
        gt = extract_ground_truth(sample)
        if not gt:
            continue

        true_cls = gt.get("classification", "unknown").lower()
        if true_cls not in CLASS_TO_IDX:
            continue

        email_text = extract_email_text(sample)
        prompt_msgs = build_prompt_messages(sample)

        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parsed_json, parse_ok = parse_model_json(response_text)
        if not parse_ok:
            parse_failures += 1

        pred_cls = parsed_json.get("classification", "unknown").lower()
        if pred_cls not in CLASS_TO_IDX:
            # Map fallback or unknown to most conservative
            pred_cls = "suspicious"

        y_true.append(true_cls)
        y_pred.append(pred_cls)

        # Hallucination check
        h_score = check_hallucination(parsed_json, email_text)
        hallucination_scores.append(h_score)

        # Abstention check
        if gt.get("abstain", False) or gt.get("classification") == "unknown":
            abstention_total += 1
            if parsed_json.get("abstain", False) or pred_cls == "unknown":
                abstention_correct += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_samples):
            print(f"  Processed {idx + 1}/{len(test_samples)} samples (Avg latency: {np.mean(latencies):.2f}s)")

    total_time = time.time() - start_total_time
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Metrics computation
    labels = CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec_m = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)

    # False Positive Rate: Benign classified as phishing or suspicious
    benign_idx = CLASS_TO_IDX["benign"]
    benign_total = sum(cm[benign_idx])
    benign_false_positives = sum(cm[benign_idx][i] for i in range(len(CLASSES)) if i != benign_idx)
    fpr = benign_false_positives / max(1, benign_total)

    results = {
        "model": model_variant,
        "accuracy": float(acc),
        "f1_macro": float(f1_m),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm,
        "classification_report": rep,
        "mean_latency_sec": float(np.mean(latencies)),
        "total_inference_time_sec": float(total_time),
        "parse_failure_rate": float(parse_failures / max(1, len(test_samples))),
        "hallucination_rate": float(np.mean(hallucination_scores) if hallucination_scores else 0.0),
        "abstention_accuracy": float(abstention_correct / max(1, abstention_total) if abstention_total > 0 else 0.0),
        "abstention_total": abstention_total,
        "peak_gpu_mb": float(peak_gpu_mb),
    }

    print(f"\n--- {model_variant} Results ---")
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1:    {f1_m:.4f}")
    print(f"FPR:         {fpr * 100:.2f}%")
    print(f"Halluc. Rate:{results['hallucination_rate'] * 100:.2f}%")
    print(f"Peak VRAM:   {peak_gpu_mb:.2f} MB")
    print(f"Avg Latency: {results['mean_latency_sec']:.2f} s")

    return results

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_samples = load_test_samples(TEST_DATASET_PATH)
    print(f"Loaded {len(test_samples)} test samples from {TEST_DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target evaluate QLoRA
    qlora_adapter_dir = Path("/content/model_checkpoints/qlora_adapter")
    if qlora_adapter_dir.exists():
        print(f"\nLoading QLoRA Model from {qlora_adapter_dir} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        qlora_model = PeftModel.from_pretrained(base_model, str(qlora_adapter_dir))
        qlora_results = evaluate_model_pipeline("QLoRA", qlora_model, tokenizer, test_samples)

        save_path = OUTPUT_DIR / "qlora_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(qlora_results, f, indent=2)
        print(f"Saved QLoRA evaluation results to: {save_path}")

        # Local copy
        local_path = Path("experiments/qlora_results.json")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(qlora_results, f, indent=2)

if __name__ == "__main__":
    main()
