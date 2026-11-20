<div align="center">

# 🛡️ SOC Triage Assistant
### Evidence-Aware Email Phishing & Incident Triage with Fine-Tuned Qwen SLMs

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Model: Qwen2.5-1.5B](https://img.shields.io/badge/Backbone-Qwen2.5--1.5B--Instruct-0ea5e9.svg?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Model: Qwen1.5-1.8B](https://img.shields.io/badge/Backbone-Qwen1.5--1.8B--Chat-8b5cf6.svg?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
[![Fine-Tuning: LoRA & QLoRA](https://img.shields.io/badge/Technique-LoRA%20%26%204--bit%20NF4%20QLoRA-f97316.svg)](https://github.com/huggingface/peft)
[![VRAM: 5.44 GB](https://img.shields.io/badge/Training%20VRAM-5.44%20GB%20(QLoRA)-10b981.svg)](https://developer.nvidia.com/cuda-gpus)
[![Accuracy: 91.4%](https://img.shields.io/badge/Accuracy-91.43%25-emerald.svg)](experiments/research_report.md)
[![False Positive Rate: 0.0%](https://img.shields.io/badge/FPR-0.00%25-green.svg)](experiments/research_report.md)
[![UI: Streamlit](https://img.shields.io/badge/Analyst%20UI-Streamlit-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-gray.svg)](LICENSE)

<br/>

**A privacy-preserving, edge-deployable Tier-1 SOC assistant that eliminates alert fatigue, defends against prompt injection, grounds evidence in MITRE ATT&CK, and explicitly abstains on ambiguous inputs.**

[Key Metrics](#-empirical-benchmarks) • [Interactive Features](#-interactive-features) • [Quickstart](#-quickstart-in-3-steps) • [Architecture](#-system-architecture) • [Research Report](#-academic-grounding)

</div>

---

## ⚡ Why This Matters: The SOC Dilemma

Modern Security Operations Centers (SOCs) receive **thousands of email alerts daily**, leading to severe analyst burnout and missed breaches. 

| The Challenge | Why Traditional Cloud LLMs Fail | How SOC-Triage-Assistant Solves It |
| :--- | :--- | :--- |
| **Alert Fatigue** | Expensive API billing at enterprise volume | **Runs locally on edge hardware** (RTX 4050 / Colab T4) in <6GB VRAM |
| **Data Privacy (GDPR/HIPAA)** | Sends sensitive enterprise emails to external cloud APIs | **Zero external data transit** — 100% on-premises execution |
| **Overconfident Hallucination** | Sounds persuasive even when guessing blindly | **Explicit Safety Abstention (`abstain = true`)** on insufficient evidence |
| **Adversarial Jailbreaks** | Vulnerable to prompt injection inside email bodies | **Active ChatML delimiter filtering** (`<|im_start|>`, instruction overrides) |

---

## 📊 Empirical Benchmarks (Held-Out Test Set)

Evaluated on 70 strictly held-out security emails with **zero train/test data leakage**:
