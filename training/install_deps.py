"""
Bootstrap script to ensure all required ML dependencies are installed
on the remote Google Colab GPU runtime.

Usage:
    colab exec -s soc-triage -f training/install_deps.py --timeout 120
"""

import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "pydantic",
]

def check_and_install():
    print("=== Colab Remote Dependency Bootstrap ===")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            missing.append(pkg)
        else:
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, "__version__", "installed")
                print(f"[OK] {pkg}: {ver}")
            except Exception as e:
                print(f"[ERR] {pkg} found but failed import: {e}")
                missing.append(pkg)

    if missing:
        print(f"\nInstalling missing packages: {missing} ...")
        cmd = [sys.executable, "-m", "pip", "install", *missing]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Installation failed:\n{res.stderr}")
            sys.exit(1)
        print("Installation complete.")
    else:
        print("\nAll required packages are already installed.")

    # Sanity checks
    print("\n=== Running Sanity Checks ===")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("[OK] BitsAndBytesConfig (4-bit NF4) created successfully.")
    except Exception as e:
        print(f"[WARN] BitsAndBytesConfig check failed: {e}")

    try:
        import trl
        print(f"[OK] TRL loaded successfully (v{trl.__version__}).")
    except Exception as e:
        print(f"[ERR] TRL import failed: {e}")

    print("\nEnvironment bootstrap completed successfully!")

if __name__ == "__main__":
    check_and_install()
