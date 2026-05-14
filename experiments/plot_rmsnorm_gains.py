"""Plot F-4: post-norm RMSNorm gain magnitudes across the model lineup.

Source data: `results/rmsnorm_gains.csv`, which holds the mean|gain| values
reported in RESULTS.md F44 (derived from `inspect_rmsnorm_gains.py`, whose
raw `.log` output is gitignored). Re-run `experiments/inspect_rmsnorm_gains.py`
to regenerate the underlying numbers.

The story: Gemma 3's post-attention and post-feedforward RMSNorm gains are
20-30x larger than Qwen 2.5, Gemma 2, or Gemma 4. Those inflated gains amplify
per-block residual perturbations, which is the mechanism behind the cross-Gemma
single-layer-ablation asymmetry (F42/F44). Gemma 4 corrected the calibration.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.style import apply, COLORS, label_for
apply()

CSV = ROOT / "results" / "rmsnorm_gains.csv"

NORMS = [
    ("post_attention_layernorm", "post_attention_layernorm"),
    ("post_feedforward_layernorm", "post_feedforward_layernorm"),
]

rows = []
with CSV.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

models = [r["short_name"] for r in rows]
n_models = len(models)
group_w = 0.8
bar_w = group_w / n_models

fig, ax = plt.subplots(figsize=(8, 4.5))

for gi, (col, _label) in enumerate(NORMS):
    for mi, r in enumerate(rows):
        raw = r[col].strip()
        x = gi + (mi - (n_models - 1) / 2) * bar_w
        if raw == "":
            # Qwen has only 2 norm types — no post_feedforward_layernorm.
            ax.text(x, 0.6, "n/a", ha="center", va="bottom",
                    fontsize=7, color=COLORS["accent_refused"], rotation=90)
            continue
        val = float(raw)
        ax.bar(x, val, bar_w * 0.92, color=COLORS[r["short_name"]],
               label=label_for(r["short_name"]) if gi == 0 else None)
        ax.text(x, val + 0.6, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(range(len(NORMS)))
ax.set_xticklabels([n[1] for n in NORMS])
ax.set_ylabel("mean |gain|  (averaged over blocks)")
ax.set_title("Post-norm RMSNorm gain magnitudes across the model lineup")
ax.set_ylim(0, 38)
ax.legend(loc="upper left", title=None)

# Annotate the headline contrast.
ax.annotate("Gemma 3: 20-30x larger\nthan its predecessor or successor",
            xy=(1.18, 33.69), xytext=(1.3, 26),
            fontsize=8.5, color=COLORS["gemma3_1b"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gemma3_1b"], lw=1.2))

fig.tight_layout()
out = ROOT / "results" / "rmsnorm_gains.png"
fig.savefig(out)
print(f"wrote {out}")
