"""
Inspect the existing downloaded data, dataset files, schema, and metrics.
"""

import json
from pathlib import Path

DATA_DIR = Path("/content/dataset/raw/data")
METRICS_DIR = Path("/content/dataset/raw/metrics")
CHECKPOINT_DIR = Path("/content/dataset/raw/model_checkpoints/lora_adapter")

def inspect():
    print("=" * 60)
    print("  INSPECTING EXISTING DATASET FILES")
    print("=" * 60)

    # 1. Inspect data files
    for split_name in ["pilot_train.jsonl", "pilot_validation.jsonl", "pilot_test.jsonl"]:
        p = DATA_DIR / split_name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                lines = [f.readline() for _ in range(3)]
                total_lines = 1 + sum(1 for _ in f) if lines else 0
            # reset to count properly
            with open(p, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            print(f"\n[FILE] {split_name} ({total_lines} lines / samples)")
            if lines:
                try:
                    sample = json.loads(lines[0])
                    print("Sample keys:", list(sample.keys()))
                    if "messages" in sample:
                        print("Format: Chat template with 'messages'")
                        for msg in sample["messages"]:
                            role = msg.get("role")
                            content = msg.get("content", "")
                            preview = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")
                            print(f"  Role '{role}': {preview}")
                    else:
                        print("First sample content:", json.dumps(sample, indent=2)[:500])
                except Exception as e:
                    print("Error parsing sample JSON:", e)

    # 2. Inspect metrics
    print("\n" + "=" * 60)
    print("  INSPECTING EXISTING METRICS")
    print("=" * 60)
    for m in ["baseline_results.json", "lora_results.json", "lora_training_log.json"]:
        mp = METRICS_DIR / m
        if mp.exists():
            print(f"\n[METRIC] {m}:")
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(json.dumps(data, indent=2)[:600])
            except Exception as e:
                print(f"Error reading {m}: {e}")

    # 3. Inspect adapter config
    cfg_p = CHECKPOINT_DIR / "adapter_config.json"
    if cfg_p.exists():
        print("\n" + "=" * 60)
        print("  INSPECTING LORA ADAPTER CONFIG")
        print("=" * 60)
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            print(json.dumps(cfg, indent=2))

if __name__ == "__main__":
    inspect()
