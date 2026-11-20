"""
Generate a publication-grade, high-resolution 16:9 social preview thumbnail
for GitHub, LinkedIn, and presentations.
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 16:9 aspect ratio at high resolution (1920 x 1080)
DPI = 150
WIDTH_PX = 1920
HEIGHT_PX = 1080
FIG_W = WIDTH_PX / DPI
FIG_H = HEIGHT_PX / DPI

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor("#070b14")
ax.set_facecolor("#070b14")
ax.set_xlim(0, 100)
ax.set_ylim(0, 56.25)  # 16:9 scale
ax.axis("off")

# 1. Subtle cyber grid lines in background
for x in range(0, 105, 5):
    ax.axvline(x, color="#0f172a", lw=0.8, alpha=0.6, zorder=1)
for y in np.arange(0, 60, 2.8125):
    ax.axhline(y, color="#0f172a", lw=0.8, alpha=0.6, zorder=1)

# Radial glowing ambient circles in background
glow_circle_left = patches.Circle((20, 28), 28, color="#0369a1", alpha=0.12, zorder=2)
glow_circle_center = patches.Circle((50, 30), 22, color="#4338ca", alpha=0.15, zorder=2)
glow_circle_right = patches.Circle((80, 26), 25, color="#0d9488", alpha=0.10, zorder=2)
ax.add_patch(glow_circle_left)
ax.add_patch(glow_circle_center)
ax.add_patch(glow_circle_right)

# 2. Glowing Header Tag
ax.text(
    50, 48.5,
    "A I   x   C Y B E R S E C U R I T Y   O P E R A T I O N S",
    color="#38bdf8",
    fontsize=15,
    fontweight="bold",
    ha="center",
    va="center",
    family="sans-serif",
    zorder=10,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#0284c7", lw=1.5, alpha=0.9)
)

# 3. Main Hero Title
ax.text(
    50, 40.5,
    "SOC TRIAGE ASSISTANT",
    color="#ffffff",
    fontsize=38,
    fontweight="heavy",
    ha="center",
    va="center",
    family="sans-serif",
    zorder=10
)

# 4. Subtitle / Architecture Tagline
ax.text(
    50, 34.0,
    "Evidence-Aware Phishing Triage with Fine-Tuned Qwen2.5 SLMs",
    color="#94a3b8",
    fontsize=18,
    fontweight="semibold",
    ha="center",
    va="center",
    family="sans-serif",
    zorder=10
)

# 5. Core Feature Badges (Pills)
badges = [
    ("Qwen2.5-1.5B Backbone", "#0284c7"),
    ("LoRA & 4-bit QLoRA", "#f97316"),
    ("91.4% Accuracy (0% FPR)", "#10b981"),
    ("5.44 GB VRAM Edge Fit", "#8b5cf6"),
    ("MITRE ATT&CK Grounded", "#ec4899"),
]
start_x = 10.0
box_w = 15.5
for i, (b_text, b_color) in enumerate(badges):
    bx = start_x + (i * 16.5)
    rect = patches.FancyBboxPatch(
        (bx, 24.5), box_w, 4.2,
        boxstyle="round,pad=0.3",
        facecolor="#0f172a",
        edgecolor=b_color,
        lw=1.8,
        alpha=0.95,
        zorder=10
    )
    ax.add_patch(rect)
    ax.text(
        bx + box_w/2, 26.6,
        b_text,
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=11
    )

# 6. Lower Showcase Cards (Three Columns)
# Card 1: Threat Radar
c1 = patches.FancyBboxPatch((8, 6), 25, 15, boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#ef4444", lw=1.5, alpha=0.85, zorder=8)
ax.add_patch(c1)
ax.text(20.5, 18.5, "THREAT RADAR DEFENSE", color="#f87171", fontsize=12, fontweight="bold", ha="center", zorder=9)
ax.text(20.5, 14.5, "THREAT SCORE: 92/100", color="#ffffff", fontsize=15, fontweight="bold", ha="center", zorder=9)
ax.text(20.5, 11.0, "Spearphishing Link (T1566.002)\nPrompt Injection Neutralized (100%)", color="#94a3b8", fontsize=9.5, ha="center", va="center", zorder=9)
ax.text(20.5, 7.8, "Forensic Evidence Grounded", color="#38bdf8", fontsize=9, fontweight="bold", ha="center", zorder=9)

# Card 2: Benchmarks & VRAM
c2 = patches.FancyBboxPatch((37.5, 6), 25, 15, boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#10b981", lw=1.5, alpha=0.85, zorder=8)
ax.add_patch(c2)
ax.text(50, 18.5, "EMPIRICAL BENCHMARKS", color="#34d399", fontsize=12, fontweight="bold", ha="center", zorder=9)
ax.text(50, 14.5, "91.43% Triage Accuracy", color="#ffffff", fontsize=15, fontweight="bold", ha="center", zorder=9)
ax.text(50, 11.0, "Zero-Shot 31.4% -> LoRA 91.4% (+60%)\nQLoRA: 5.44 GB Training VRAM (-56.7%)", color="#94a3b8", fontsize=9.5, ha="center", va="center", zorder=9)
ax.text(50, 7.8, "0.00% False Positive Rate", color="#10b981", fontsize=9, fontweight="bold", ha="center", zorder=9)

# Card 3: Interactive UI & Duel
c3 = patches.FancyBboxPatch((67, 6), 25, 15, boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", lw=1.5, alpha=0.85, zorder=8)
ax.add_patch(c3)
ax.text(79.5, 18.5, "INTERACTIVE WORKBENCH", color="#38bdf8", fontsize=12, fontweight="bold", ha="center", zorder=9)
ax.text(79.5, 14.5, "Streamlit Analyst UI", color="#ffffff", fontsize=15, fontweight="bold", ha="center", zorder=9)
ax.text(79.5, 11.0, "3-Way Model Duel (Base vs LoRA vs QLoRA)\nAdversarial Red-Team Generator\nIn-Browser Fine-Tuning Studio", color="#94a3b8", fontsize=9.5, ha="center", va="center", zorder=9)
ax.text(79.5, 7.8, "Human-in-the-Loop Review", color="#a855f7", fontsize=9, fontweight="bold", ha="center", zorder=9)

# Save high-res PNG
output_path = Path("thumbnail.png")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(output_path, dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Thumbnail generated successfully: {output_path.resolve()}")
