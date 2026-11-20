
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_datasets.py

Fetch all public SOC-triage related resources and store them under
raw_datasets/.  The script works both in Google Colab (recommended) and on a
local machine with internet access.

Folder layout after a successful run:

raw_datasets/
├ huggingface/
│   ├─ phishing_email_dataset/        ← zefang-liu / phishing-email-dataset
│   ├─ seven_phishing_email_datasets/ ← puyang2025 / seven-phishing-email-datasets
│   └─ enron_emails/                  ← corbt / enron-emails
├─ url_feeds/
│   ├─ phishtank_online_valid.csv
│   └─ urlhaus_online.csv
└─ mitre_attack/
    └─ enterprise-attack.json
"""

import os
import json
from pathlib import Path

# --------------------------------------------------------------
# 0 - Global configuration
# --------------------------------------------------------------
PROJECT_ROOT = Path.cwd()                     # wherever you run the script
RAW_ROOT = PROJECT_ROOT / "raw_datasets"

HF_ROOT    = RAW_ROOT / "huggingface"
URL_ROOT   = RAW_ROOT / "url_feeds"
MITRE_ROOT = RAW_ROOT / "mitre_attack"

for p in (HF_ROOT, URL_ROOT, MITRE_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# 1 - Hugging Face corpora
# --------------------------------------------------------------
def download_hf_dataset(repo_id: str, split: str = "train", dst: Path = None):
    """
    Downloads a HF repo with datasets.load_dataset and writes the whole
    split as a parquet file into dst.
    """
    from datasets import load_dataset

    print(f"🔽 Downloading HF repo `{repo_id}` …")
    ds = load_dataset(repo_id, split=split, cache_dir=str(dst))
    out_path = dst / f"{repo_id.replace('/', '_')}_{split}.parquet"
    ds.to_parquet(str(out_path))
    print(f"✅ Saved → {out_path}")
    return out_path


# --------------------------------------------------------------
# 2 - URL-feed CSVs (PhishTank & URLhaus)
# --------------------------------------------------------------
def fetch_csv(url: str, dst_path: Path):
    import requests
    print(f"🔽 Fetching CSV from {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    dst_path.write_bytes(r.content)
    print(f"✅ Saved → {dst_path}")


# --------------------------------------------------------------
# 3 - MITRE ATT&CK STIX bundle (enterprise-attack)
# --------------------------------------------------------------
def download_mitre_stix(dst_dir: Path):
    import requests, zipfile, shutil

    zip_url = "https://github.com/mitre/cti/archive/refs/heads/master.zip"
    zip_path = dst_dir / "cti_master.zip"

    print(f"🔽 Downloading MITRE ATT&CK STIX bundle from {zip_url}")
    r = requests.get(zip_url, timeout=20)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [n for n in z.namelist() if n.endswith("enterprise-attack.json")]
        if not candidates:
            raise FileNotFoundError("enterprise-attack.json not found in the zip")
        src_name = candidates[0]
        z.extract(src_name, path=dst_dir)

    # Move to a clean location
    extracted = dst_dir / src_name
    final_path = dst_dir / "enterprise-attack.json"
    shutil.move(str(extracted), str(final_path))

    # Cleanup
    shutil.rmtree(dst_dir / "cti-master", ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    print(f"✅ MITRE ATT&CK STIX saved → {final_path}")
    return final_path


# --------------------------------------------------------------
# 4 - Main execution
# --------------------------------------------------------------
if __name__ == "__main__":
    # ----- 1 - HF datasets -----
    # a) Phishing-email-dataset (zefang-liu)
    download_hf_dataset(
        repo_id="zefang-liu/phishing-email-dataset",
        dst=HF_ROOT,
    )

    # b) Seven-phishing-email-datasets (puyang2025)
    download_hf_dataset(
        repo_id="puyang2025/seven-phishing-email-datasets",
        dst=HF_ROOT,
    )

    # c) Enron email corpus – use the real repo name:
    download_hf_dataset(
        repo_id="corbt/enron-emails",
        dst=HF_ROOT,
    )

    # ----- 2 - URL feeds -----
    fetch_csv(
        "https://data.phishtank.com/data/online-valid.csv",
        URL_ROOT / "phishtank_online_valid.csv",
    )
    fetch_csv(
        "https://urlhaus.abuse.ch/downloads/csv_online/",
        URL_ROOT / "urlhaus_online.csv",
    )

    # ----- 3 - MITRE ATT&CK -----
    download_mitre_stix(MITRE_ROOT)

    # ----- 4 - Summary & manifest -----
    print("\n=== All downloads completed ===")
    print(f"Root folder: {RAW_ROOT}")
    print("Sub-folders:")
    for sub in RAW_ROOT.iterdir():
        if sub.is_dir():
            print(f" • {sub.name} ({len(list(sub.iterdir()))} items)")

    manifest = {
        "datasets": {
            "phishing_email_dataset": str(HF_ROOT / "zefang-liu_phishing-email-dataset_train.parquet"),
            "seven_phishing_email_datasets": str(HF_ROOT / "puyang2025_seven-phishing-email-datasets_train.parquet"),
            "enron_emails": str(HF_ROOT / "corbt_enron-emails_train.parquet"),
        },
        "url_feeds": {
            "phishtank": str(URL_ROOT / "phishtank_online_valid.csv"),
            "urlhaus": str(URL_ROOT / "urlhaus_online.csv"),
        },
        "mitre_attack": str(MITRE_ROOT / "enterprise-attack.json"),
    }
    manifest_path = RAW_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"🗂️ Manifest written to {manifest_path}")
