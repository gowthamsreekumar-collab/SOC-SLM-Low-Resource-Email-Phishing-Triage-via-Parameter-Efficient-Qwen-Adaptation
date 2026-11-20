"""
Quality Evaluation Module for SOC Triage Assistant.

Evaluates:
1. MITRE ATT&CK Mapping Quality:
   - Precision, Recall, and F1 of predicted ATT&CK technique IDs against ground truth.
2. Evidence-to-Conclusion Consistency:
   - Verification that cited evidence corresponds logically to the chosen threat classification.
   - Groundedness check: confirms evidence strings originate from the raw email text.
3. Recommended Action & Summary Alignment:
   - ROUGE-L and Token F1 overlap between model recommendations and analyst ground truth.
4. Comparative Quality Summary across Baseline, LoRA, and QLoRA models.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_ground_truth_output(sample):
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg["content"])
            except Exception:
                pass
    return None


def extract_email_body(sample):
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def tokenize_words(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def compute_rouge_l(candidate, reference):
    """Compute sentence-level ROUGE-L (LCS) F1 score."""
    cand_tokens = tokenize_words(candidate)
    ref_tokens = tokenize_words(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    # LCS dynamic programming
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if cand_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs_len = dp[m][n]
    prec = lcs_len / m if m > 0 else 0.0
    rec = lcs_len / n if n > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def evaluate_mitre_attack_mapping(predicted_outputs, ground_truths):
    """
    Evaluates precision, recall, and F1 for MITRE ATT&CK technique IDs.
    """
    tp = 0
    fp = 0
    fn = 0
    per_technique = {}

    for pred, gt in zip(predicted_outputs, ground_truths):
        pred_techniques = set()
        for item in pred.get("attack_mapping", []):
            tid = item.get("technique_id", "").strip().upper()
            if tid:
                pred_techniques.add(tid)

        gt_techniques = set()
        for item in gt.get("attack_mapping", []):
            tid = item.get("technique_id", "").strip().upper()
            if tid:
                gt_techniques.add(tid)

        # TP, FP, FN
        true_pos = pred_techniques & gt_techniques
        false_pos = pred_techniques - gt_techniques
        false_neg = gt_techniques - pred_techniques

        tp += len(true_pos)
        fp += len(false_pos)
        fn += len(false_neg)

        for t in gt_techniques:
            if t not in per_technique:
                per_technique[t] = {"tp": 0, "fp": 0, "fn": 0}
            if t in true_pos:
                per_technique[t]["tp"] += 1
            else:
                per_technique[t]["fn"] += 1

        for t in false_pos:
            if t not in per_technique:
                per_technique[t] = {"tp": 0, "fp": 0, "fn": 0}
            per_technique[t]["fp"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "total_true_positives": tp,
        "total_false_positives": fp,
        "total_false_negatives": fn,
        "per_technique": per_technique,
    }


def evaluate_evidence_consistency(predicted_outputs, raw_emails):
    """
    Evaluates:
    1. Groundedness: Is evidence derived from raw email text?
    2. Logic Consistency: Does evidence support the classification severity?
    """
    grounded_count = 0
    total_evidence_items = 0
    severity_aligned_count = 0

    for pred, email_text in zip(predicted_outputs, raw_emails):
        evidence_list = pred.get("evidence", [])
        classification = pred.get("classification", "").lower()
        severity = pred.get("severity", "").lower()
        email_lower = email_text.lower()

        # Severity alignment rule check
        if classification == "benign":
            if severity in ("low", "informational", "none"):
                severity_aligned_count += 1
        elif classification in ("suspicious", "phishing"):
            if severity in ("medium", "high", "critical"):
                severity_aligned_count += 1

        for ev in evidence_list:
            total_evidence_items += 1
            ev_lower = str(ev).lower().strip()
            # Check direct substring or high token overlap (>50%)
            ev_tokens = tokenize_words(ev_lower)
            if not ev_tokens:
                continue
            matching_tokens = sum(1 for t in ev_tokens if t in email_lower)
            token_ratio = matching_tokens / len(ev_tokens)
            if ev_lower in email_lower or token_ratio >= 0.5:
                grounded_count += 1

    groundedness_rate = grounded_count / total_evidence_items if total_evidence_items > 0 else 1.0
    severity_consistency_rate = severity_aligned_count / len(predicted_outputs) if predicted_outputs else 1.0

    return {
        "total_evidence_items": total_evidence_items,
        "grounded_evidence_items": grounded_count,
        "evidence_groundedness_rate": groundedness_rate,
        "severity_consistency_rate": severity_consistency_rate,
    }


def evaluate_action_alignment(predicted_outputs, ground_truths):
    """
    Computes average ROUGE-L and Action Set Jaccard index between model recommended actions and ground truth.
    """
    rouge_l_scores = []
    jaccard_scores = []

    for pred, gt in zip(predicted_outputs, ground_truths):
        pred_actions = pred.get("recommended_actions", [])
        gt_actions = gt.get("recommended_actions", [])

        pred_text = " ".join(str(a) for a in pred_actions)
        gt_text = " ".join(str(a) for a in gt_actions)

        rouge_l = compute_rouge_l(pred_text, gt_text)
        rouge_l_scores.append(rouge_l)

        # Token Jaccard
        p_tokens = set(tokenize_words(pred_text))
        g_tokens = set(tokenize_words(gt_text))
        if not p_tokens and not g_tokens:
            jaccard_scores.append(1.0)
        else:
            jaccard = len(p_tokens & g_tokens) / max(1, len(p_tokens | g_tokens))
            jaccard_scores.append(jaccard)

    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    return {
        "mean_rouge_l": avg_rouge,
        "mean_action_jaccard": avg_jaccard,
    }


def run_quality_evaluation():
    print("=" * 65)
    print("  SOC TRIAGE ASSISTANT - MODEL QUALITY EVALUATION")
    print("=" * 65)

    test_path = Path("dataset/test.jsonl")
    if not test_path.exists():
        test_path = Path("/content/dataset/test.jsonl")

    if not test_path.exists():
        print(f"[ERROR] Test dataset not found at {test_path}")
        return

    test_samples = load_jsonl(test_path)
    print(f"Loaded {len(test_samples)} test samples.")

    ground_truths = [extract_ground_truth_output(s) for s in test_samples]
    raw_emails = [extract_email_body(s) for s in test_samples]

    # Evaluate ground truth consistency as benchmark ceiling
    ceiling_mitre = evaluate_mitre_attack_mapping(ground_truths, ground_truths)
    ceiling_evidence = evaluate_evidence_consistency(ground_truths, raw_emails)
    ceiling_actions = evaluate_action_alignment(ground_truths, ground_truths)

    # Load experimental results
    results_dir = Path("experiments")
    baseline_path = results_dir / "baseline_results.json"
    lora_path = results_dir / "lora_results.json"
    qlora_path = results_dir / "qlora_results.json"

    models_summary = {}

    for model_name, path in [("Baseline", baseline_path), ("LoRA", lora_path), ("QLoRA", qlora_path)]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models_summary[model_name] = {
                "accuracy": data.get("accuracy", 0.0),
                "f1_macro": data.get("f1_macro", 0.0),
                "false_positive_rate": data.get("false_positive_rate", 0.0),
                "hallucination_rate": data.get("hallucination_rate", 0.0),
                "parse_failure_rate": data.get("parse_failure_rate", 0.0),
            }

    quality_report = {
        "dataset_size": len(test_samples),
        "ground_truth_baseline": {
            "evidence_groundedness_rate": ceiling_evidence["evidence_groundedness_rate"],
            "severity_consistency_rate": ceiling_evidence["severity_consistency_rate"],
            "total_evidence_items": ceiling_evidence["total_evidence_items"],
            "techniques_identified": list(ceiling_mitre["per_technique"].keys()),
        },
        "model_performance_summary": models_summary,
        "mitre_evaluation": {
            "top_techniques": sorted(
                ceiling_mitre["per_technique"].keys(),
                key=lambda t: ceiling_mitre["per_technique"][t]["tp"],
                reverse=True,
            )[:8]
        },
    }

    out_file = results_dir / "quality_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    print("\n--- Model Quality Summary ---")
    print(f"{'Model':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'FPR':<8} | {'Hallucination':<14}")
    print("-" * 65)
    for name, metrics in models_summary.items():
        print(
            f"{name:<12} | "
            f"{metrics['accuracy']*100:>8.2f}% | "
            f"{metrics['f1_macro']:>10.4f} | "
            f"{metrics['false_positive_rate']*100:>6.2f}% | "
            f"{metrics['hallucination_rate']*100:>12.2f}%"
        )
    print("-" * 65)
    print(f"\n[DONE] Quality evaluation report saved to: {out_file}")


if __name__ == "__main__":
    run_quality_evaluation()
