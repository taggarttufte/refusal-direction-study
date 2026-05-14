"""Plot F-3: cross-Gemma single-layer ablation matrix (block x direction-source).

Parses the qualitative matrices from
`results/diagnose_depth_alignment_{gemma2_2b,gemma3_1b,gemma4_e2b}.log`
(produced by `diagnose_depth_alignment_matrix.py`) and renders a 3-panel
categorical grid, one panel per Gemma generation.

Each cell is a (application_block x direction_source) combination, colored
by outcome: BASE (no effect / baseline preserved), JBR (clean jailbreak),
BREAK (degenerate output), MIXED (split jailbreak/break), PART (partial /
unclassified). The runner picks blocks and direction sources at matched
fractional depths across models, so the three panels are comparable.

Story: Gemma 3 is the architectural outlier. Gemma 2 resists single-layer
ablation almost entirely (1/15 cells), Gemma 4 resists it completely
(0/15), but Gemma 3 is broadly affected (12/15 cells show an effect) and
broadly fragile.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.style import apply, label_for
apply()

MODELS = ["gemma2_2b", "gemma3_1b", "gemma4_e2b"]

# Outcome -> color. Semantic, matching src/style.py conventions:
# gray = nothing happened, green = clean jailbreak, red = broke the model,
# orange = split outcome, tan = partial / unclassified.
TAG_COLOR = {
    "BASE":  "#7f7f7f",
    "JBR":   "#2ca02c",
    "BREAK": "#d62728",
    "MIXED": "#ff9f1a",
    "PART":  "#d9c89e",
}
TAG_DESC = {
    "BASE":  "BASE - no effect",
    "JBR":   "JBR - clean jailbreak",
    "BREAK": "BREAK - degenerate",
    "MIXED": "MIXED - split JB/break",
    "PART":  "PART - partial",
}
# Light cell backgrounds want dark text; dark backgrounds want white.
DARK_TEXT_TAGS = {"PART"}


def parse_log(path: Path):
    """Return dict with blocks, srcs, tag grid, baseline string."""
    text = path.read_text(encoding="utf-8", errors="replace")

    app_blocks = [int(x) for x in re.search(
        r"Application blocks:\s*\[([^\]]+)\]", text).group(1).split(",")]
    src_layers = [int(x) for x in re.search(
        r"Direction sources:\s*\[([^\]]+)\]", text).group(1).split(",")]

    base_match = re.search(r"no hooks:\s*h_REF=\s*(\d+)/10\s+h_DEG=\s*(\d+)/10"
                           r"\s+h_JBC=\s*(\d+)/10", text)
    h_ref, h_deg, h_jbc = (int(g) for g in base_match.groups())

    # Qualitative matrix block: 5 rows of "  block   N    TAG  TAG  TAG"
    qual = text.split("QUALITATIVE MATRIX")[1]
    grid = {}
    for line in qual.splitlines():
        m = re.match(r"\s*block\s+(\d+)\s+(.+)", line)
        if not m:
            continue
        blk = int(m.group(1))
        tags = m.group(2).split()
        if len(tags) == len(src_layers):
            for src, tag in zip(src_layers, tags):
                grid[(blk, src)] = tag
    return {
        "blocks": app_blocks, "srcs": src_layers, "grid": grid,
        "baseline": f"baseline {h_ref}/10 refuse",
    }


def main():
    data = {m: parse_log(ROOT / "results" / f"diagnose_depth_alignment_{m}.log")
            for m in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    src_col_labels = ["early", "mid", "late"]

    for ax, model in zip(axes, MODELS):
        d = data[model]
        blocks, srcs, grid = d["blocks"], d["srcs"], d["grid"]
        n_row, n_col = len(blocks), len(srcs)

        for ri, blk in enumerate(blocks):
            for ci, src in enumerate(srcs):
                tag = grid[(blk, src)]
                # Row 0 at top: invert y.
                y = n_row - 1 - ri
                ax.add_patch(plt.Rectangle((ci, y), 1, 1,
                                           facecolor=TAG_COLOR[tag],
                                           edgecolor="white", linewidth=2))
                ax.text(ci + 0.5, y + 0.5, tag, ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="#222222" if tag in DARK_TEXT_TAGS else "white")

        ax.set_xlim(0, n_col)
        ax.set_ylim(0, n_row)
        ax.set_xticks([c + 0.5 for c in range(n_col)])
        ax.set_xticklabels([f"{lab}\nd[{src}]"
                            for lab, src in zip(src_col_labels, srcs)])
        ax.set_yticks([n_row - 1 - r + 0.5 for r in range(n_row)])
        ax.set_yticklabels([f"block {b}" for b in blocks])
        ax.set_title(f"{label_for(model)}\n{d['baseline']}", fontsize=10.5)
        ax.set_xlabel("direction source (by depth)")
        if model == MODELS[0]:
            ax.set_ylabel("application block (by depth)")
        ax.grid(False)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # Shared legend below.
    handles = [Patch(facecolor=TAG_COLOR[t], edgecolor="white", label=TAG_DESC[t])
               for t in ["BASE", "JBR", "BREAK", "MIXED", "PART"]]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Single-layer refusal-direction ablation across the Gemma "
                 "family (5x3: application block x direction source)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = ROOT / "results" / "cross_gemma_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
