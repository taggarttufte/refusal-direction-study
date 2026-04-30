"""Morning-after analysis of saved refusal-direction artifacts.

Loads each .pt file in activations/, prints summary stats, and produces
a stack of plots in results/:

  1. refusal_direction_norms.png
        ||direction[l]|| as a function of layer, one line per model.
        Layer index normalized to [0, 1] so models with different depths
        overlay on the same axis. Annotates Gemma layers by attention type.

  2. cosine_similarity_to_peak.png
        cos(direction[l], direction[l*]) where l* is the model's argmax-norm
        layer. Tells you how "stable" the direction is along the depth.

  3. norms_table.csv
        per-model peak-norm layer + value, useful for downstream ablation.

Run: python analyze.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import sys
import torch

ROOT = Path(__file__).parent.parent
ACT = ROOT / "activations"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.style import apply, COLORS, MARKERS, color_for, label_for
apply()


def _load_all() -> list[dict]:
    files = sorted(ACT.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt files in {ACT}")
    data = []
    for f in files:
        # weights_only=False because we save plain tensors + metadata
        d = torch.load(f, map_location="cpu", weights_only=False)
        data.append(d)
    return data


def _norm_layer_axis(n_layers: int) -> torch.Tensor:
    return torch.linspace(0, 1, n_layers)


def plot_direction_norms(data: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in data:
        norms = torch.tensor(d["norms_direction"])
        x = _norm_layer_axis(len(norms))
        try:
            color = color_for(d["short"])
        except KeyError:
            color = None
        line, = ax.plot(x.numpy(), norms.numpy(),
                        label=f"{label_for(d['short'])}  (L={len(norms)-1})",
                        marker="o", markersize=3, color=color)
        # annotate full-attention layers in Gemma stacks with stars
        lt = d.get("layer_types")
        if lt:
            for i, t in enumerate(lt):
                if t == "full_attention":
                    ax.scatter([x[i + 1].item()], [norms[i + 1].item()],
                               color=line.get_color(),
                               s=80, marker="*",
                               edgecolor="black", linewidth=0.5,
                               zorder=10)
    ax.set_xlabel("normalized depth (0 = embedding, 1 = final layer)")
    ax.set_ylabel(r"$\|\mathrm{refusal\_direction}[\ell]\|$  (fp32 L2 norm)")
    ax.set_title("Refusal-direction magnitude across depth\n"
                 r"$\bigstar$ marks full-attention layers in Gemma stacks")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = OUT / "refusal_direction_norms.png"
    fig.savefig(p)
    print(f"  wrote {p}")
    plt.close(fig)


def plot_cosine_to_peak(data: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in data:
        dirs = d["refusal_direction"]  # (L+1, hidden)
        norms = dirs.norm(dim=-1)
        peak = int(norms.argmax().item())
        peak_dir = dirs[peak] / (norms[peak] + 1e-9)
        cos = (dirs @ peak_dir) / (norms + 1e-9)
        x = _norm_layer_axis(len(cos))
        try:
            color = color_for(d["short"])
        except KeyError:
            color = None
        ax.plot(x.numpy(), cos.numpy(),
                label=f"{label_for(d['short'])} (peak @ L={peak})",
                marker="o", markersize=3, color=color)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("normalized depth (0 = embedding, 1 = final layer)")
    ax.set_ylabel(r"$\cos(\mathrm{direction}[\ell],\, \mathrm{direction}[\mathrm{peak}])$")
    ax.set_title("Refusal-direction stability along depth\n"
                 "(1.0 = perfectly aligned with peak-norm layer)")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    p = OUT / "cosine_similarity_to_peak.png"
    fig.savefig(p)
    print(f"  wrote {p}")
    plt.close(fig)


def write_norms_table(data: list[dict]):
    p = OUT / "norms_table.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_prompts", "n_layers",
                    "peak_layer", "peak_norm",
                    "harmful_norm_at_peak", "harmless_norm_at_peak",
                    "peak_layer_type"])
        for d in data:
            norms = torch.tensor(d["norms_direction"])
            peak = int(norms.argmax().item())
            lt = d.get("layer_types")
            # layer_types[i] aligns with hidden_states[i+1]
            ltype = (lt[peak - 1] if lt and 0 < peak <= len(lt) else
                     "embedding" if peak == 0 else "n/a")
            w.writerow([
                d["short"], d["n"], len(norms),
                peak, f"{norms[peak].item():.4f}",
                f"{d['norms_harmful'][peak]:.4f}",
                f"{d['norms_harmless'][peak]:.4f}",
                ltype,
            ])
    print(f"  wrote {p}")


def main():
    data = _load_all()
    print(f"loaded {len(data)} model(s) from {ACT}")
    for d in data:
        norms = torch.tensor(d["norms_direction"])
        peak = int(norms.argmax().item())
        print(f"  {d['short']:14s}  L={len(norms):2d}  "
              f"peak@layer={peak:2d}  ||dir||_max={norms[peak]:.3f}  "
              f"layer_types={'yes' if d.get('layer_types') else 'no'}")

    print("\nproducing plots...")
    plot_direction_norms(data)
    plot_cosine_to_peak(data)
    write_norms_table(data)
    print("done.")


if __name__ == "__main__":
    main()
