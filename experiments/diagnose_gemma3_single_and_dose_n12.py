"""Single-layer + dose-threshold scan on Gemma 3 1B at N=12.

Two experiments in one model load:

EXPERIMENT 1 (single-layer N=12 re-verification of F33):
  Ablate at exactly ONE transformer block. Test 8 informative blocks
  spanning full + sliding × early/mid/late. Each block gets the 12-prompt
  diagnostic battery.

EXPERIMENT 2 (dose-threshold scan):
  Ablate at K randomly-chosen transformer blocks for K in {2, 3, 5, 8, 12}.
  Locate the dose level at which Gemma 3 starts breaking down. Combined
  with single-layer (K=1) and prior every-layer (K=26) data, this gives
  us a dose-vs-coherence curve.

Direction source: residual_stream layer 14 throughout (‖d‖ ~676).
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ablate import (
    register_ablation_hooks, register_ablation_hooks_at_indices,
    unregister_hooks,
)
from src.diag_prompts import load_diag_prompts, HARMFUL_INDICES, BENIGN_INDICES
from src.extract import format_prompt, load_model, unload
from src.refusal_classifier import is_degraded, is_refusal


MODEL_ID = "google/gemma-3-1b-it"
PT_PATH  = ROOT / "activations" / "gemma3_1b.pt"
SOURCE_LAYER = 14

# EXPERIMENT 1: single-layer test points (block_index, attn_type, label)
SINGLE_LAYER_TESTS = [
    (4,  "sliding", "early"),
    (5,  "full",    "early"),
    (11, "full",    "mid"),
    (12, "sliding", "mid"),
    (16, "sliding", "mid-late"),
    (17, "full",    "mid-late"),
    (22, "sliding", "late"),
    (23, "full",    "late"),
]

# EXPERIMENT 2: dose-threshold scan. For each K, sample K random block
# indices from [0, 26). Use a fixed seed for reproducibility.
DOSE_LEVELS = [2, 3, 5, 8, 12]
RNG_SEED = 42
N_BLOCKS = 26


@torch.no_grad()
def gen(model, tok, prompt, max_new=80):
    enc = format_prompt(tok, prompt)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    pt = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
    if full.startswith(pt):
        full = full[len(pt):]
    return full.strip()


def classify(text: str, category: str) -> str:
    if is_degraded(text):
        return "DEG"
    if is_refusal(text):
        return "REF"
    return "COR" if category == "benign" else "JBC"


def run_config(model, tok, prompts, hook_fn) -> list[dict]:
    handles = hook_fn()
    out_results = []
    try:
        for p in prompts:
            out = gen(model, tok, p["text"])
            out_results.append({
                "prompt_id": p["id"],
                "category": p["category"],
                "class": classify(out, p["category"]),
                "len": len(out),
                "preview": out[:60].replace("\n", " "),
            })
    finally:
        unregister_hooks(handles)
    return out_results


def summarize(label: str, rows: list[dict]) -> tuple[int, int, int, int, int, int, int]:
    h = [r for r in rows if r["category"] != "benign"]
    b = [r for r in rows if r["category"] == "benign"]
    h_ref = sum(r["class"] == "REF" for r in h)
    h_deg = sum(r["class"] == "DEG" for r in h)
    h_jbc = sum(r["class"] == "JBC" for r in h)
    b_cor = sum(r["class"] == "COR" for r in b)
    b_deg = sum(r["class"] == "DEG" for r in b)
    return h_ref, h_deg, h_jbc, len(h), b_cor, b_deg, len(b)


def main():
    print(f"loading {MODEL_ID}...")
    model, tok = load_model(MODEL_ID)
    payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    direction = payload["refusal_direction"][SOURCE_LAYER].clone()
    print(f"source layer = {SOURCE_LAYER}  ||direction|| = {direction.norm().item():.2f}")

    prompts = load_diag_prompts()
    print(f"prompts: {len(prompts)} (harmful: 10, benign: 2)\n")

    # ============================================================
    # EXPERIMENT 1: single-layer at N=12
    # ============================================================
    print("="*80)
    print(" EXPERIMENT 1: single-layer ablation, N=12")
    print("="*80)
    e1: list[tuple] = []
    for blk_idx, attn, depth_label in SINGLE_LAYER_TESTS:
        cfg_label = f"L=block-{blk_idx} ({attn} {depth_label})"
        print(f"running {cfg_label}...")
        rows = run_config(
            model, tok, prompts,
            lambda b=blk_idx: register_ablation_hooks_at_indices(
                model, direction, [b]
            ),
        )
        e1.append((cfg_label, rows))

    # ============================================================
    # EXPERIMENT 2: dose threshold
    # ============================================================
    print("\n" + "="*80)
    print(" EXPERIMENT 2: dose-threshold scan, N=12")
    print("="*80)
    rng = random.Random(RNG_SEED)
    e2: list[tuple] = []
    # baseline (K=0, no hooks) and ALL-26 control
    print(f"running baseline (K=0)...")
    rows_baseline = run_config(model, tok, prompts, lambda: [])
    e2.append(("K=0 (baseline)", [], rows_baseline))
    print(f"running K=26 (all)...")
    rows_all = run_config(
        model, tok, prompts,
        lambda: register_ablation_hooks(model, direction),
    )
    e2.append(("K=26 (all)", list(range(N_BLOCKS)), rows_all))
    for K in DOSE_LEVELS:
        chosen = sorted(rng.sample(range(N_BLOCKS), K))
        cfg_label = f"K={K:>2d} blocks={chosen}"
        print(f"running {cfg_label}...")
        rows = run_config(
            model, tok, prompts,
            lambda c=chosen: register_ablation_hooks_at_indices(
                model, direction, c
            ),
        )
        e2.append((cfg_label, chosen, rows))

    # ============================================================
    # REPORT
    # ============================================================
    print("\n" + "="*80)
    print(" EXPERIMENT 1 SUMMARY (single-layer)")
    print("="*80)
    print(f"  {'config':<32s}  {'h_REF':>5s} {'h_DEG':>5s} {'h_JBC':>5s} {'/':<2s} {'b_COR':>5s} {'b_DEG':>5s}")
    for label, rows in e1:
        h_ref, h_deg, h_jbc, hn, b_cor, b_deg, bn = summarize(label, rows)
        print(f"  {label:<32s}  {h_ref:>5d} {h_deg:>5d} {h_jbc:>5d}    {b_cor:>5d} {b_deg:>5d}")

    print("\n" + "="*80)
    print(" EXPERIMENT 2 SUMMARY (dose threshold)")
    print("="*80)
    print(f"  {'config':<48s}  {'h_REF':>5s} {'h_DEG':>5s} {'h_JBC':>5s} {'/':<2s} {'b_COR':>5s} {'b_DEG':>5s}")
    for label, _, rows in e2:
        h_ref, h_deg, h_jbc, hn, b_cor, b_deg, bn = summarize(label, rows)
        print(f"  {label:<48s}  {h_ref:>5d} {h_deg:>5d} {h_jbc:>5d}    {b_cor:>5d} {b_deg:>5d}")

    # GRID for experiment 1
    print("\n" + "="*80)
    print(" EXPERIMENT 1 GRID (rows = configs, cols = prompt id)")
    print("="*80)
    cat_row = "       cat: " + "  ".join(
        " H " if i in HARMFUL_INDICES else " B " for i in range(len(prompts))
    )
    print(cat_row)
    for label, rows in e1:
        cells = "  ".join(r["class"][:3] for r in rows)
        print(f"  {label:<32s}  {cells}")

    # GRID for experiment 2
    print("\n" + "="*80)
    print(" EXPERIMENT 2 GRID (rows = K-configs, cols = prompt id)")
    print("="*80)
    print(cat_row)
    for label, _, rows in e2:
        cells = "  ".join(r["class"][:3] for r in rows)
        # truncate long labels
        short = label if len(label) <= 32 else label[:29] + "..."
        print(f"  {short:<32s}  {cells}")

    unload(model)


if __name__ == "__main__":
    main()
