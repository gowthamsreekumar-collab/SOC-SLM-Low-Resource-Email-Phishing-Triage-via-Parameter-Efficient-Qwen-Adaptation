"""
Download dataset files from Google Drive folder using gdown.
Target directory: /content/dataset/raw
"""

import os
import subprocess
import sys
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1C0qtGOQe-Ei5BgM0GewK4wfivIWiXc2e"
TARGET_DIR = Path("/content/dataset/raw")

def download_data():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {TARGET_DIR}")
    print(f"Downloading from Google Drive folder: {FOLDER_URL} ...")

    cmd = [
        sys.executable, "-m", "gdown",
        "--folder",
        "--remaining-ok",
        FOLDER_URL,
        "-O", str(TARGET_DIR)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

    print("\nContents of target directory:")
    for root, dirs, files in os.walk(TARGET_DIR):
        rel = os.path.relpath(root, TARGET_DIR)
        print(f"Directory: {rel}")
        for f in files:
            p = Path(root) / f
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    download_data()
