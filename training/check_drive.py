"""
Check Google Drive mount status and create persistent project directories.
"""

import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_DRIVE_DIR = DRIVE_ROOT / "soc-triage-assistant"

SUBDIRS = [
    "datasets",
    "checkpoints/lora",
    "checkpoints/qlora",
    "adapters/lora",
    "adapters/qlora",
    "experiments/baseline",
    "experiments/lora",
    "experiments/qlora",
    "experiments/graphs",
]

def check_drive():
    print("=== Google Drive Mount Check ===")
    if not DRIVE_ROOT.exists():
        print("[STATUS] Google Drive is NOT mounted at /content/drive/MyDrive.")
        print("Note: To mount Drive in this Colab session, run the following in a Colab notebook cell:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return False

    print("[STATUS] Google Drive is mounted successfully!")
    print(f"Creating persistent directory structure at: {PROJECT_DRIVE_DIR}")

    for sub in SUBDIRS:
        target = PROJECT_DRIVE_DIR / sub
        target.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR READY] {target}")

    print("\nGoogle Drive persistent storage initialized successfully!")
    return True

if __name__ == "__main__":
    check_drive()
