"""Source-layer / application-layer mismatch test on Gemma 3 1B.

Hypothesis: block 3's "fragility" in F36 is actually about applying a
late-layer direction (direction[14]) at an early block. The late
direction's orientation is essentially orthogonal to the local block-3
direction (cos = 0.029) — but it's a *learned* orientation, not random.
Projecting it out at block 3 might destroy collateral computation that
random projections leave alone (F38).

Test: at block 3 and block 17 separately, ablate using directions
extracted from different depths. If direction[4] (block 3's own
depth-matched direction) does NOT break block 3, the "fragility" is
really an off-target source-layer effect, not a block-3-specific
property.

Configurations (6):
  Block 3 (the previously-fragile one):
    - direction[4]  (depth-matched, ‖d‖≈8 — local class-separation)
    - direction[14] (mid-network, ‖d‖≈676 — our usual source, breaks block 3)
    - direction[25] (peak, ‖d‖≈4214 — very late direction)
  Block 17 (the tolerant one — known to admit ablation):
    - direction[18] (depth-matched, ≈the source we used in earlier tests)
    - direction[14] (cross-source, our usual)
    - direction[4]  (very early direction applied at late block)

All directions are normalized to unit length inside the hook (per
src/ablate.py:register_ablation_hooks_at_indices), so what we're varying
is the *orientation* of the projection-out, not its magnitude.

Predictions:
  (block 3, direction[4]):   NOT broken  -- hypothesis confirmed
  (block 3, direction[14]):  BROKEN      -- reproduces F36
  (block 3, direction[25]):  BROKEN      -- late directions break block 3
  (block 17, direction[18]): WORKS       -- coherent compliance (matches F31/F33)
  (block 17, direction[14]): WORKS       -- known from F33-N=12
  (block 17, direction[4]):  WORKS or COR -- low-magnitude / orthogonal direction
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

# (block_index, direction_source_layer, label)
CONFIGS = [
    (3,  4,  "block 3  + direction[4]   (depth-matched)"),
    (3,  14, "block 3  + direction[14]  (source, breaks)"),
    (3,  25, "block 3  + direction[25]  (peak)"),
    (17, 18, "block 17 + direction[18]  (depth-matched)"),
    (17, 14, "block 17 + direction[14]  (cross-source)"),
    (17, 4,  "block 17 + direction[4]   (very-early)"),
]


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


def summarize(rows):
    h = [r for r in rows if r["category"] != "benign"]
    b = [r for r in rows if r["category"] == "benign"]
    return (
        sum(r["class"] == "REF" for r in h),
        sum(r["class"] == "DEG" for r in h),
        sum(r["class"] == "JBC" for r in h),
        sum(r["class"] == "COR" for r in b),
        sum(r["class"] == "DEG" for r in b),
    )


def main():
    print(f"loading {MODEL_ID}...")
    model, tok = load_model(MODEL_ID)
    payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    dirs = payload["refusal_direction"]
    norms = dirs.norm(dim=-1)

    # show the direction norms / cosines for context
    print("\nDirection magnitudes / cosines (for context):")
    print(f"  ||dir[4]||  = {norms[4].item():>8.2f}")
    print(f"  ||dir[14]|| = {norms[14].item():>8.2f}")
    print(f"  ||dir[18]|| = {norms[18].item():>8.2f}")
    print(f"  ||dir[25]|| = {norms[25].item():>8.2f}")
    d14_unit = dirs[14] / (norms[14] + 1e-9)
    print(f"  cos(dir[4],  dir[14]) = "
          f"{((dirs[4] @ d14_unit) / (norms[4] + 1e-9)).item():>+6.3f}")
    print(f"  cos(dir[18], dir[14]) = "
          f"{((dirs[18] @ d14_unit) / (norms[18] + 1e-9)).item():>+6.3f}")
    print(f"  cos(dir[25], dir[14]) = "
          f"{((dirs[25] @ d14_unit) / (norms[25] + 1e-9)).item():>+6.3f}")
    print()

    prompts = load_diag_prompts()
    results = []
    for blk_idx, src_layer, label in CONFIGS:
        d = dirs[src_layer].clone()
        if d.norm().item() < 1e-6:
            print(f"SKIP {label}: ||d|| ≈ 0")
            continue
        print(f"running {label}...")
        handles = register_ablation_hooks_at_indices(model, d, [blk_idx])
        try:
            rows = []
            for p in prompts:
                out = gen(model, tok, p["text"])
                rows.append({
                    "prompt_id": p["id"],
                    "category": p["category"],
                    "class": classify(out, p["category"]),
                    "preview": out[:60].replace("\n", " "),
                })
        finally:
            unregister_hooks(handles)
        results.append((label, rows))

    print("\n" + "="*88)
    print(" SUMMARY")
    print("="*88)
    print(f"  {'config':<48s}  {'h_REF':>5s} {'h_DEG':>5s} {'h_JBC':>5s} | "
          f"{'b_COR':>5s} {'b_DEG':>5s}")
    for label, rows in results:
        h_ref, h_deg, h_jbc, b_cor, b_deg = summarize(rows)
        print(f"  {label:<48s}  {h_ref:>5d} {h_deg:>5d} {h_jbc:>5d}   "
              f"{b_cor:>5d} {b_deg:>5d}")

    print("\n" + "="*88)
    print(" GRID  (rows = configs, cols = prompt id)")
    print("="*88)
    cat_row = "  cat: " + " ".join(
        " H " if i in HARMFUL_INDICES else " B " for i in range(len(prompts))
    )
    print(cat_row)
    for label, rows in results:
        cells = " ".join(r["class"][:3] for r in rows)
        print(f"  {label:<48s}  {cells}")

    print("\n" + "="*88)
    print(" SAMPLE OUTPUTS  (first 2 prompts per config)")
    print("="*88)
    for label, rows in results:
        print(f"\n  [{label}]")
        for r in rows[:2]:
            print(f"    P{r['prompt_id']} ({r['category']:>6s}, "
                  f"{r['class']}): {r['preview']!r}")

    unload(model)


if __name__ == "__main__":
    main()
