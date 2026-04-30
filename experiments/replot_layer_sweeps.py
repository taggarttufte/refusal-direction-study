"""Regenerate per-model layer-sweep plots from the saved CSVs using the new style."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.style import apply, COLORS, color_for, label_for
apply()

RES = ROOT / "results"


def load(short):
    rows = []
    with (RES / f"layer_sweep_{short}.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "layer":     int(r["layer"]),
                "asr":       float(r["asr"]),
                "norm_dir":  float(r["norm_dir"]),
                "n_base_ref": int(r["n_base_ref"]),
            })
    return rows


def plot_one(short):
    rows = load(short)
    n_layers = max(r["layer"] for r in rows) + 1
    n_base_ref = rows[1]["n_base_ref"]  # row 0 is the embedding (skipped); use row 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    L = [r["layer"] for r in rows]
    asr = [r["asr"] for r in rows]
    norm = [r["norm_dir"] for r in rows]

    ax.plot(L, asr, marker="o", color=color_for(short), label="ASR (flip rate)")
    ax.set_xlabel("layer index (0 = embedding)")
    ax.set_ylabel("ASR — flip rate among baseline-refused prompts",
                  color=color_for(short))
    ax.tick_params(axis="y", labelcolor=color_for(short))
    ax.set_ylim(-0.05, 1.05)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(L, norm, marker="s", color=COLORS["accent_refused"],
             linewidth=1.0, alpha=0.7, label=r"$\|\mathrm{direction}\|$")
    ax2.set_ylabel(r"$\|\mathrm{direction}[\ell]\|$",
                   color=COLORS["accent_refused"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent_refused"])
    ax2.grid(False)

    ax.set_title(f"Layer sweep: {label_for(short)}\n"
                 f"({n_base_ref}/10 prompts refused at baseline)")
    fig.tight_layout()
    out = RES / f"layer_sweep_{short}.png"
    fig.savefig(out)
    print(f"wrote {out}")
    plt.close(fig)


for short in ("qwen25_1p5b", "qwen25_3b"):
    plot_one(short)
