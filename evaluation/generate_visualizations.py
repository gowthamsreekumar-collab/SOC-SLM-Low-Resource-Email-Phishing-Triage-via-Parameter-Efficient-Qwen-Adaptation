"""
Generate all 12 Required Training Graphs and Experimental Visualizations
from actual experiment logs (baseline_results.json, lora_results.json,
qlora_results.json, lora_training_log.json, qlora_training_log.json).

Saves high-resolution PNGs to experiments/plots/.
"""

import json
import os
import sys
from pathlib import Path

# Set non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

PLOTS_DIR = Path("experiments/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = Path("experiments")

CLASSES = ["benign", "phishing", "suspicious"]

def load_json(filename):
    p = EXP_DIR / filename
    if not p.exists():
        print(f"[WARN] {filename} does not exist yet.")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_all_plots():
    base_res = load_json("baseline_results.json")
    lora_res = load_json("lora_results.json")
    qlora_res = load_json("qlora_results.json")
    lora_log = load_json("lora_training_log.json")
    qlora_log = load_json("qlora_training_log.json")

    # 1. Training Loss vs Training Steps
    if lora_log and qlora_log:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lora_log["train_steps"], lora_log["train_losses"], label="LoRA (FP16)", color="#1f77b4", lw=2)
        ax.plot(qlora_log["train_steps"], qlora_log["train_losses"], label="QLoRA (4-bit NF4)", color="#ff7f0e", lw=2, linestyle="--")
        ax.set_title("Training Loss vs Training Steps", fontsize=14, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        plt.savefig(PLOTS_DIR / "1_train_loss_vs_steps.png", dpi=300)
        plt.close()

    # 2. Validation Loss vs Training Steps
    if lora_log and qlora_log:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lora_log["eval_steps"], lora_log["eval_losses"], label="LoRA Eval Loss", color="#1f77b4", marker="o", lw=2)
        ax.plot(qlora_log["eval_steps"], qlora_log["eval_losses"], label="QLoRA Eval Loss", color="#ff7f0e", marker="s", lw=2, linestyle="--")
        ax.set_title("Validation Loss vs Training Steps", fontsize=14, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Validation Loss")
        ax.legend()
        plt.savefig(PLOTS_DIR / "2_val_loss_vs_steps.png", dpi=300)
        plt.close()

    # 3. LoRA vs QLoRA Combined Loss Curves
    if lora_log and qlora_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(lora_log["train_steps"], lora_log["train_losses"], label="LoRA Train", color="#1f77b4")
        ax1.plot(lora_log["eval_steps"], lora_log["eval_losses"], label="LoRA Val", color="#1f77b4", linestyle="--")
        ax1.set_title("LoRA (FP16) Loss Dynamics", fontweight="bold")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(qlora_log["train_steps"], qlora_log["train_losses"], label="QLoRA Train", color="#ff7f0e")
        ax2.plot(qlora_log["eval_steps"], qlora_log["eval_losses"], label="QLoRA Val", color="#ff7f0e", linestyle="--")
        ax2.set_title("QLoRA (4-bit NF4) Loss Dynamics", fontweight="bold")
        ax2.set_xlabel("Steps")
        ax2.legend()
        plt.savefig(PLOTS_DIR / "3_lora_vs_qlora_loss_curves.png", dpi=300)
        plt.close()

    # Model comparisons (Base, LoRA, QLoRA)
    models = []
    models_data = []
    if base_res:
        models.append("Zero-Shot Base")
        models_data.append(base_res)
    if lora_res:
        models.append("LoRA")
        models_data.append(lora_res)
    if qlora_res:
        models.append("QLoRA")
        models_data.append(qlora_res)

    if models:
        # 4. Classification Accuracy Comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        accs = [m["accuracy"] * 100 for m in models_data]
        bars = ax.bar(models, accs, color=["#7f7f7f", "#1f77b4", "#ff7f0e"], width=0.5)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Classification Accuracy Comparison", fontsize=14, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "4_accuracy_comparison.png", dpi=300)
        plt.close()

        # 5. Precision, Recall, and Macro F1
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(models))
        width = 0.25
        precs = [m["precision_macro"] * 100 for m in models_data]
        recs = [m["recall_macro"] * 100 for m in models_data]
        f1s = [m["f1_macro"] * 100 for m in models_data]

        ax.bar(x - width, precs, width, label="Precision", color="#2ca02c")
        ax.bar(x, recs, width, label="Recall", color="#1f77b4")
        ax.bar(x + width, f1s, width, label="Macro F1", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Precision, Recall, and Macro F1 Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        plt.savefig(PLOTS_DIR / "5_precision_recall_f1.png", dpi=300)
        plt.close()

        # 6. Confusion Matrices
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
        if len(models) == 1:
            axes = [axes]
        for ax, name, data in zip(axes, models, models_data):
            cm = np.array(data["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=CLASSES, yticklabels=CLASSES)
            ax.set_title(f"{name} Confusion Matrix", fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        plt.savefig(PLOTS_DIR / "6_confusion_matrices.png", dpi=300)
        plt.close()

        # 7. Hallucination Rate Comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        halluc = [m.get("hallucination_rate", 0) * 100 for m in models_data]
        bars = ax.bar(models, halluc, color=["#d62728", "#1f77b4", "#2ca02c"], width=0.5)
        ax.set_ylabel("Unsupported Claim Rate (%)")
        ax.set_ylim(0, max(50, max(halluc) + 10))
        ax.set_title("Hallucination / Unsupported Claim Rate (Lower is Better)", fontsize=13, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "7_hallucination_rate.png", dpi=300)
        plt.close()

        # 8. Abstention Accuracy Comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        abst = [m.get("abstention_accuracy", 0) * 100 for m in models_data]
        bars = ax.bar(models, abst, color=["#9467bd", "#1f77b4", "#ff7f0e"], width=0.5)
        ax.set_ylabel("Abstention Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Abstention & Uncertainty Handling Accuracy", fontsize=14, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "8_abstention_accuracy.png", dpi=300)
        plt.close()

        # 9. GPU Memory Consumption Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        gpu_infer = [m.get("peak_gpu_mb", 0) / 1024 for m in models_data]
        bars = ax.bar(models, gpu_infer, color=["#7f7f7f", "#1f77b4", "#ff7f0e"], width=0.5)
        ax.axhline(14.56, color="red", linestyle="--", label="Tesla T4 VRAM Limit (14.56 GB)")
        ax.set_ylabel("Peak VRAM (GB)")
        ax.set_ylim(0, 16)
        ax.set_title("Peak GPU Memory Consumption (Inference)", fontsize=14, fontweight="bold")
        ax.legend()
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f} GB", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "9_gpu_memory_consumption.png", dpi=300)
        plt.close()

    # 10. Training Time Comparison
    if lora_log and qlora_log:
        fig, ax = plt.subplots(figsize=(7, 5))
        times_min = [lora_log["total_train_time_sec"] / 60, qlora_log["total_train_time_sec"] / 60]
        bars = ax.bar(["LoRA (FP16)", "QLoRA (4-bit NF4)"], times_min, color=["#1f77b4", "#ff7f0e"], width=0.45)
        ax.set_ylabel("Training Time (Minutes)")
        ax.set_title("Total Training Wall-Clock Time (420 Steps, 3 Epochs)", fontsize=13, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} min", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "10_training_time_comparison.png", dpi=300)
        plt.close()

    # 11. Inference Latency Comparison
    if models:
        fig, ax = plt.subplots(figsize=(7, 5))
        lat = [m.get("mean_latency_sec", 0) for m in models_data]
        bars = ax.bar(models, lat, color=["#7f7f7f", "#1f77b4", "#ff7f0e"], width=0.5)
        ax.set_ylabel("Latency per Email (Seconds)")
        ax.set_title("Average Inference Latency per Email", fontsize=14, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')
        plt.savefig(PLOTS_DIR / "11_inference_latency.png", dpi=300)
        plt.close()

    # 12. Efficiency & Performance Trade-Off Matrix
    if base_res and lora_res and qlora_res and lora_log and qlora_log:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        table_data = [
            ["Metric", "Zero-Shot Base", "LoRA (r=16)", "QLoRA (4-bit NF4)"],
            ["Accuracy", f"{base_res['accuracy']*100:.2f}%", f"{lora_res['accuracy']*100:.2f}%", f"{qlora_res['accuracy']*100:.2f}%"],
            ["Macro F1", f"{base_res['f1_macro']:.4f}", f"{lora_res['f1_macro']:.4f}", f"{qlora_res['f1_macro']:.4f}"],
            ["False Positive Rate", f"{base_res['false_positive_rate']*100:.2f}%", f"{lora_res['false_positive_rate']*100:.2f}%", f"{qlora_res['false_positive_rate']*100:.2f}%"],
            ["Hallucination Rate", f"{base_res['hallucination_rate']*100:.2f}%", f"{lora_res['hallucination_rate']*100:.2f}%", f"{qlora_res['hallucination_rate']*100:.2f}%"],
            ["Inference Latency", f"{base_res['mean_latency_sec']:.2f}s", f"{lora_res['mean_latency_sec']:.2f}s", f"{qlora_res['mean_latency_sec']:.2f}s"],
            ["Peak Inference VRAM", f"{base_res['peak_gpu_mb']/1024:.2f} GB", f"{lora_res['peak_gpu_mb']/1024:.2f} GB", f"{qlora_res['peak_gpu_mb']/1024:.2f} GB"],
            ["Peak Training VRAM", "N/A", f"{lora_log['peak_gpu_mb']/1024:.2f} GB", f"{qlora_log['peak_gpu_mb']/1024:.2f} GB"],
            ["Trainable Parameters", "0", f"{lora_log['trainable_params']:,}", f"{qlora_log['trainable_params']:,}"],
            ["Total Training Time", "0", f"{lora_log['total_train_time_sec']/60:.1f} min", f"{qlora_log['total_train_time_sec']/60:.1f} min"],
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        # Highlight header
        for j in range(4):
            table[(0, j)].set_facecolor("#2c3e50")
            table[(0, j)].get_text().set_color("white")
            table[(0, j)].get_text().set_weight("bold")
        ax.set_title("Experimental Comparative Evaluation Matrix (Base vs LoRA vs QLoRA)", fontsize=14, fontweight="bold", pad=20)
        plt.savefig(PLOTS_DIR / "12_comparative_evaluation_matrix.png", dpi=300)
        plt.close()

    print(f"All available visualization plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    generate_all_plots()
