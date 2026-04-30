"""Locate where the refusal feature 'first writes' into the Gemma 3 1B
residual stream.

Hypothesis (mechanistic guess for F36): block 3 is computationally
load-bearing for refusal because it's where the refusal feature first
becomes meaningfully separable in the residual stream. If true, we
should see one or more of:

  1. Sharp rise in ‖direction[L]‖ around layer 3-4 (separation magnitude
     starts to grow there).
  2. Cosine similarity of direction[L] to its peak-norm direction
     becomes high starting around layer 3-4 (the direction stabilizes
     into its final orientation early).
  3. Per-class mean magnitudes (‖mean_harmful[L]‖, ‖mean_harmless[L]‖)
     diverge starting around that depth.

Output:
  - results/gemma3_first_write.png  (3-panel figure: norm growth, cosine
    stability, per-class mean magnitudes)
  - prints peak-rise layer + summary
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.style import apply, COLORS, label_for
apply()

PT_PATH = ROOT / "activations" / "gemma3_1b.pt"
OUT     = ROOT / "results" / "gemma3_first_write.png"

payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
dirs       = payload["refusal_direction"]      # (L+1, hidden)
mean_harm  = payload["mean_harmful"]           # (L+1, hidden)
mean_safe  = payload["mean_harmless"]          # (L+1, hidden)
layer_types = payload["layer_types"]            # length L (per-block)

n_layers = dirs.shape[0]
print(f"Gemma 3 1B: {n_layers} hidden states (= 1 embedding + {n_layers-1} blocks)")
print(f"Full attention at transformer-block indices: "
      f"{[i for i, t in enumerate(layer_types) if t == 'full_attention']}")

norms_dir  = dirs.norm(dim=-1).numpy()
norms_h    = mean_harm.norm(dim=-1).numpy()
norms_s    = mean_safe.norm(dim=-1).numpy()
ratio      = norms_dir / (norms_h + 1e-9)  # how much of the harmful magnitude is class-discriminative

# cosine similarity of direction[L] to direction at peak-norm layer
peak = int(norms_dir.argmax())
peak_unit = dirs[peak] / (dirs[peak].norm() + 1e-9)
cos_to_peak = (dirs @ peak_unit) / (dirs.norm(dim=-1) + 1e-9)
cos_to_peak = cos_to_peak.numpy()

# Find first layer where ‖direction‖ exceeds 10% of its peak — the "emergence" layer
threshold = 0.10 * norms_dir[peak]
emergence_layer = int(next((L for L in range(n_layers) if norms_dir[L] >= threshold), peak))
print(f"Peak ‖d‖ = {norms_dir[peak]:.2f} at hidden_state index L={peak}")
print(f"First L where ‖d‖ ≥ 10% of peak: L={emergence_layer}")

# 3-panel figure
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
xs = list(range(n_layers))
gemma_color = COLORS["gemma3_1b"]
emergence_color = COLORS["accent_works"]

# Panel A: per-class mean magnitudes + direction magnitude
ax = axes[0]
ax.plot(xs, norms_h, label=r"$\|\mathrm{mean\ harmful}[L]\|$",
        color=COLORS["accent_broken"], marker="o", markersize=4)
ax.plot(xs, norms_s, label=r"$\|\mathrm{mean\ harmless}[L]\|$",
        color=COLORS["accent_works"], marker="s", markersize=4)
ax.plot(xs, norms_dir, label=r"$\|\mathrm{direction}[L]\|$",
        color=gemma_color, marker="^", markersize=4, linewidth=2.0)
ax.axvline(3, color="black", linestyle="--", alpha=0.5, label="block 3 (fragile)")
ax.set_ylabel("L2 norm  (fp32)")
ax.set_title("Where does the refusal feature first emerge in Gemma 3 1B?")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=8)

# Panel B: linear-scale zoom on the early-layer rise
ax = axes[1]
zoom_n = min(10, n_layers)
ax.plot(xs[:zoom_n], norms_dir[:zoom_n], color=gemma_color,
        marker="^", markersize=5, linewidth=2.0,
        label=r"$\|\mathrm{direction}[L]\|$  (linear, layers 0–9)")
ax.axvline(3, color="black", linestyle="--", alpha=0.5)
ax.axhline(threshold, color=emergence_color, linestyle=":", alpha=0.6,
           label=f"10% of peak = {threshold:.1f}")
ax.set_ylabel(r"$\|\mathrm{direction}[L]\|$")
ax.legend(loc="upper left", fontsize=8)

# Panel C: cosine similarity to peak direction
ax = axes[2]
ax.plot(xs, cos_to_peak, color=COLORS["qwen25_3b"], marker="o", markersize=4,
        label=r"$\cos(\mathrm{direction}[L],\, \mathrm{direction}[\mathrm{peak}])$")
ax.axvline(3, color="black", linestyle="--", alpha=0.5)
ax.axhline(0, color="gray", alpha=0.5, linewidth=0.5)
ax.set_xlabel("hidden_state index  L  (= output of transformer block L-1)")
ax.set_ylabel("cosine to peak")
ax.set_ylim(-0.2, 1.05)
ax.legend(loc="lower right", fontsize=8)

# Annotate full-attention layers across all panels
for ax in axes:
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            ax.axvline(i + 1, color=COLORS["gemma3_1b"], alpha=0.15)

fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")

# Print the early-layer table for inspection
print("\n  L   ||d||      ||harm||    ||safe||    cos-to-peak   block-type")
for L in range(min(8, n_layers)):
    btype = (layer_types[L-1] if L >= 1 else "embedding")[:8]
    print(f"  {L:>2d}  {norms_dir[L]:>8.2f}  {norms_h[L]:>10.2f}  "
          f"{norms_s[L]:>10.2f}  {cos_to_peak[L]:>10.3f}  {btype}")
