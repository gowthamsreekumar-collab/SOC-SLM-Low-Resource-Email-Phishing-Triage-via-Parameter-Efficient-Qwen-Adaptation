import importlib.util
import sys

packages = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "pydantic",
    "sqlite3",
]

print("=== Remote Colab Python Environment ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\n=== Dependency Check ===")
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    if spec is not None:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed (no __version__)")
            print(f"[FOUND] {pkg}: {version}")
        except Exception as e:
            print(f"[ERROR importing {pkg}]: {e}")
    else:
        print(f"[MISSING] {pkg}")
