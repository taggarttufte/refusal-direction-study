"""Hypothesis (f) test + direction-vs-general-fragility discrimination, N=12.

EXPERIMENT 1 (direct test of hypothesis (f) from F36):
  Single-layer ablation at blocks 3 and 7 alone using the refusal direction.
  If both produce degenerate output, (f) is confirmed: those blocks are
  computationally fragile to projection-out ablation.

EXPERIMENT 2 (direction-specificity test):
  Same blocks (3, 7) ablated with a RANDOM unit vector scaled to the same
  magnitude as the refusal direction. Plus block 17 (known to tolerate
  the refusal direction) with random direction as a control.

Outcomes:
  - Refusal-direction breaks blocks 3,7 AND random-direction also breaks
    them: GENERAL fragility (blocks are sensitive to any in-stream
    intervention).
  - Refusal-direction breaks blocks 3,7 BUT random-direction does NOT:
    DIRECTION-SPECIFIC fragility (something about the refusal feature
    interacts with these blocks particularly).
  - Block 17 with random direction: confirms whether tolerant blocks
    are tolerant to ANY direction or just the refusal one.
"""
from __future__ import annotations

import io
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

from src.ablate import register_ablation_hooks_at_indices, unregister_hooks
from src.diag_prompts import load_diag_prompts, HARMFUL_INDICES, BENIGN_INDICES
from src.extract import format_prompt, load_model, unload
from src.refusal_classifier import is_degraded, is_refusal


MODEL_ID = "google/gemma-3-1b-it"
PT_PATH  = ROOT / "activations" / "gemma3_1b.pt"
SOURCE_LAYER = 14

# Reproducible random direction
RNG_SEED = 7


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


def make_random_direction(hidden_dim: int, target_norm: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_dim, generator=g, dtype=torch.float32)
    v = v / v.norm() * target_norm
    return v


def run_config(model, tok, prompts, direction, block_idx) -> list[dict]:
    handles = register_ablation_hooks_at_indices(model, direction, [block_idx])
    out_results = []
    try:
        for p in prompts:
            out = gen(model, tok, p["text"])
            out_results.append({
                "prompt_id": p["id"],
                "category": p["category"],
                "class": classify(out, p["category"]),
                "preview": out[:60].replace("\n", " "),
            })
    finally:
        unregister_hooks(handles)
    return out_results


def summarize(rows):
    h = [r for r in rows if r["category"] != "benign"]
    b = [r for r in rows if r["category"] == "benign"]
    h_ref = sum(r["class"] == "REF" for r in h)
    h_deg = sum(r["class"] == "DEG" for r in h)
    h_jbc = sum(r["class"] == "JBC" for r in h)
    b_cor = sum(r["class"] == "COR" for r in b)
    b_deg = sum(r["class"] == "DEG" for r in b)
    return h_ref, h_deg, h_jbc, b_cor, b_deg


def main():
    print(f"loading {MODEL_ID}...")
    model, tok = load_model(MODEL_ID)
    payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    refusal_direction = payload["refusal_direction"][SOURCE_LAYER].clone()
    target_norm = refusal_direction.norm().item()
    hidden_dim = refusal_direction.shape[0]
    random_direction = make_random_direction(hidden_dim, target_norm, RNG_SEED)
    print(f"refusal direction: ||d|| = {target_norm:.2f}, hidden_dim = {hidden_dim}")
    print(f"random direction:  ||d|| = {random_direction.norm().item():.2f} (matched)")

    prompts = load_diag_prompts()
    configs = [
        ("block 3 + refusal_dir", 3,  refusal_direction),
        ("block 3 + random_dir",  3,  random_direction),
        ("block 7 + refusal_dir", 7,  refusal_direction),
        ("block 7 + random_dir",  7,  random_direction),
        ("block 17 + refusal_dir (control: known-tolerant)", 17, refusal_direction),
        ("block 17 + random_dir  (control: tolerant?)", 17, random_direction),
    ]

    results = []
    for label, blk, direction in configs:
        print(f"running {label}...")
        rows = run_config(model, tok, prompts, direction, blk)
        results.append((label, rows))

    print("\n" + "="*84)
    print(" SUMMARY")
    print("="*84)
    print(f"  {'config':<55s}  {'h_REF':>5s} {'h_DEG':>5s} {'h_JBC':>5s} | {'b_COR':>5s} {'b_DEG':>5s}")
    for label, rows in results:
        h_ref, h_deg, h_jbc, b_cor, b_deg = summarize(rows)
        print(f"  {label:<55s}  {h_ref:>5d} {h_deg:>5d} {h_jbc:>5d}   {b_cor:>5d} {b_deg:>5d}")

    print("\n" + "="*84)
    print(" GRID  (rows = configs, cols = prompt id; H/B = harmful/benign)")
    print("="*84)
    cat_row = "  cat: " + " ".join(
        " H " if i in HARMFUL_INDICES else " B " for i in range(len(prompts))
    )
    print(cat_row)
    for label, rows in results:
        cells = " ".join(r["class"][:3] for r in rows)
        short = label if len(label) <= 50 else label[:47] + "..."
        print(f"  {short:<50s}  {cells}")

    print("\n" + "="*84)
    print(" SAMPLE OUTPUTS  (first 2 prompts per config)")
    print("="*84)
    for label, rows in results:
        print(f"\n  [{label}]")
        for r in rows[:2]:
            print(f"    P{r['prompt_id']} ({r['category']:>6s}, {r['class']}): {r['preview']!r}")

    unload(model)


if __name__ == "__main__":
    main()
