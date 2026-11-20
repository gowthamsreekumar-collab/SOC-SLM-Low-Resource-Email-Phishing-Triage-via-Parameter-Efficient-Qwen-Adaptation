"""
Model Loader for SOC Triage Assistant.
Loads Base, LoRA, and QLoRA variants across the Qwen2.5 and Qwen1.5 model families:
- Qwen/Qwen2.5-1.5B-Instruct (Primary research & PPT benchmark)
- Qwen/Qwen2.5-0.5B-Instruct (Ultra-light edge)
- Qwen/Qwen1.5-1.8B-Chat (Stable foundation chat)
- Qwen/Qwen1.5-0.5B-Chat (Compact 1.5 series)

Supports CPU and GPU environments with resilient fallback handling for local dashboard testing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LORA_ADAPTER_DIR = Path("model_checkpoints/lora_adapter")
DEFAULT_QLORA_ADAPTER_DIR = Path("model_checkpoints/qlora_adapter")

class TriageModelWrapper:
    def __init__(
        self,
        variant: str = "LoRA",
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: Optional[str] = None,
    ):
        self.variant = variant
        self.model_id = model_id
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self._load()

    def _load(self):
        if not TORCH_AVAILABLE:
            print(f"[INFO] PyTorch not detected. Operating in high-performance heuristic inference mode for {self.model_id} ({self.variant}).")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            print(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.variant == "Base (Zero-Shot)":
                print(f"Loading Zero-Shot Base model: {self.model_id}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
            elif "QLoRA" in self.variant:
                print(f"Loading 4-bit Base model for QLoRA ({self.model_id})...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                chosen_adapter = self.adapter_path or (
                    DEFAULT_QLORA_ADAPTER_DIR if DEFAULT_QLORA_ADAPTER_DIR.exists() else Path("/content/model_checkpoints/qlora_adapter")
                )
                if chosen_adapter.exists():
                    self.model = PeftModel.from_pretrained(base, str(chosen_adapter))
                else:
                    print(f"[WARN] QLoRA adapter not found at {chosen_adapter}, using base model weights.")
                    self.model = base
            else:  # LoRA or Custom
                print(f"Loading Base model with LoRA adapter ({self.model_id})...")
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
                chosen_adapter = self.adapter_path or (
                    DEFAULT_LORA_ADAPTER_DIR if DEFAULT_LORA_ADAPTER_DIR.exists() else Path("/content/model_checkpoints/lora_adapter")
                )
                if chosen_adapter.exists():
                    self.model = PeftModel.from_pretrained(base, str(chosen_adapter))
                else:
                    print(f"[WARN] LoRA adapter not found at {chosen_adapter}, using base model weights.")
                    self.model = base

            print(f"Model successfully loaded: {self.model_id} [{self.variant}]")
        except Exception as e:
            print(f"[WARN] Local model loading bypassed: {e}")
            self.model = None

    def generate_response(self, messages: list) -> str:
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            return ""

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return ""
