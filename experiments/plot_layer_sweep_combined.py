"""Combined layer-sweep plot: Qwen 1.5B + Qwen 3B on shared depth axis.

Reads results/layer_sweep_qwen25_1p5b.csv and results/layer_sweep_qwen25_3b.csv.
Plots ASR vs normalized depth so the two model sizes overlay despite different
layer counts.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"

sys.path.insert(0, str(ROOT))
from src.style import apply, color_for, label_for
apply()


def load(short):
    rows = []
    with (RES / f"layer_sweep_{short}.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "layer": int(r["layer"]),
                "asr": float(r["asr"]),
                "norm_dir": float(r["norm_dir"]),
            })
    return rows


def main():
    a = load("qwen25_1p5b")
    b = load("qwen25_3b")
    nA = max(r["layer"] for r in a) + 1
    nB = max(r["layer"] for r in b) + 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["layer"] / (nA - 1) for r in a],
            [r["asr"] for r in a],
            marker="o", color=color_for("qwen25_1p5b"),
            label=f"{label_for('qwen25_1p5b')} (L={nA-1})")
    ax.plot([r["layer"] / (nB - 1) for r in b],
            [r["asr"] for r in b],
            marker="o", color=color_for("qwen25_3b"),
            label=f"{label_for('qwen25_3b')} (L={nB-1})")
    ax.set_xlabel("normalized transformer depth (0 = embedding, 1 = final layer)")
    ax.set_ylabel("ASR — flip rate among baseline-refused prompts")
    ax.set_title("Refusal-direction ablation: ASR vs normalized depth")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = RES / "layer_sweep_qwen_size_comparison.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    plt.close(fig)

    # peak-layer summary
    print("\nPeaks:")
    for short, rows, n in [("Qwen 1.5B", a, nA), ("Qwen 3B", b, nB)]:
        peak = max(rows, key=lambda r: r["asr"])
        n_at_peak = sum(1 for r in rows if r["asr"] == peak["asr"])
        depth = peak["layer"] / (n - 1)
        print(f"  {short:9s}  peak ASR={peak['asr']:.0%}  "
              f"first peak at layer {peak['layer']} ({depth:.0%} depth)  "
              f"n_layers_at_peak={n_at_peak}")


if __name__ == "__main__":
    main()
