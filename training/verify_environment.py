"""
Comprehensive environment verification script for SOC Triage Assistant.
Executes on the remote Colab GPU runtime.

Usage:
    colab exec -s soc-triage -f training/verify_environment.py --timeout 120
"""

import sys
from pathlib import Path

def verify_all():
    report = {}
    print("=" * 60)
    print("  SOC TRIAGE ASSISTANT - PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. System & Python
    print("\n[1/5] Checking Python Environment...")
    py_ver = sys.version.split()[0]
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {py_ver}")
    report["python"] = py_ver

    # 2. CUDA & GPU
    print("\n[2/5] Checking GPU & CUDA...")
    import torch
    cuda_avail = torch.cuda.is_available()
    report["cuda_available"] = cuda_avail
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        print(f"  CUDA Available: YES (PyTorch {torch.__version__}, CUDA {torch.version.cuda})")
        print(f"  Device: {gpu_name}")
        print(f"  Total VRAM: {vram_gb} GB")
        report["gpu_name"] = gpu_name
        report["vram_gb"] = vram_gb

        # Run a quick tensor computation on GPU
        a = torch.randn(512, 512, device="cuda")
        b = a @ a
        print(f"  GPU Matrix Ops Test: PASS (output norm: {b.norm().item():.2f})")
    else:
        print("  CUDA Available: NO")
        report["gpu_name"] = None

    # 3. Required ML Libraries
    print("\n[3/5] Verifying Core ML Packages...")
    pkgs = [
        "transformers",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "pydantic",
    ]
    pkg_status = {}
    for p in pkgs:
        try:
            m = __import__(p)
            ver = getattr(m, "__version__", "ok")
            print(f"  [PASS] {p:<15} v{ver}")
            pkg_status[p] = ver
        except Exception as e:
            print(f"  [FAIL] {p:<15} Error: {e}")
            pkg_status[p] = f"FAIL: {e}"
    report["packages"] = pkg_status

    # 4. Qwen2.5-1.5B Tokenizer Hugging Face Connectivity
    print("\n[4/5] Testing Hugging Face Model Connectivity (Tokenizer)...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        from transformers import AutoTokenizer
        print(f"  Fetching tokenizer config for: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab_size = len(tokenizer)
        print(f"  [PASS] Successfully loaded tokenizer! Vocab size: {vocab_size}")
        report["tokenizer_test"] = f"PASS (vocab: {vocab_size})"
    except Exception as e:
        print(f"  [FAIL] Tokenizer load failed: {e}")
        report["tokenizer_test"] = f"FAIL: {e}"

    # 5. Persistent Storage (Google Drive)
    print("\n[5/5] Checking Persistent Storage...")
    drive_path = Path("/content/drive/MyDrive/soc-triage-assistant")
    if Path("/content/drive/MyDrive").exists():
        print(f"  [PASS] Google Drive mounted at /content/drive/MyDrive")
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"  [PASS] Project directory ready: {drive_path}")
        report["drive_mounted"] = True
    else:
        print(f"  [INFO] Google Drive is currently NOT mounted.")
        print(f"         Local VM /content can be used, but Drive mount is recommended for persistence.")
        report["drive_mounted"] = False

    # Summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    all_crit_passed = (
        report["cuda_available"]
        and all(not v.startswith("FAIL") for v in report["packages"].values())
        and report["tokenizer_test"].startswith("PASS")
    )
    if all_crit_passed:
        print("  STATUS: ALL CORE CHECKS PASSED! Ready for Phase 2.")
    else:
        print("  STATUS: SOME CHECKS FAILED. Review output above.")
    print("=" * 60)

    return all_crit_passed

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
