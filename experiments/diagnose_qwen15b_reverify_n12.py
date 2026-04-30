"""F5 re-verification on Qwen 2.5 1.5B at N=12 with coherence checks.

The original F5 finding (100% ASR on Qwen 1.5B layer 14) was obtained at N=10
with substring-only refusal classification — no degeneracy filtering. Given
that we caught classifier-blind-spot issues on the Gemma family (F25/F30
where degenerate output was misclassified as compliance), we should re-verify
the headline Qwen result with the same coherence pipeline used for the
Gemma matrices.

Three conditions:
  1. baseline (no hooks)
  2. every-layer ablation with direction[14]   (the F5 protocol)
  3. single-layer ablation at block 13 with direction[14]
     (block 13 is the source block for direction[14])

Each condition runs the standard 12-prompt diagnostic battery and reports
REF / DEG / JBC / COR class counts.
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

from src.ablate import (
    register_ablation_hooks, register_ablation_hooks_at_indices,
    unregister_hooks,
)
from src.diag_prompts import load_diag_prompts, HARMFUL_INDICES, BENIGN_INDICES
from src.extract import format_prompt, load_model, unload
from src.refusal_classifier import is_degraded, is_refusal


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PT_PATH  = ROOT / "activations" / "qwen25_1p5b.pt"
SOURCE_LAYER = 14   # the F5 layer; direction[14] = output of block 13


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


def run_config(model, tok, prompts, hook_setup_fn) -> list[dict]:
    handles = hook_setup_fn()
    rows = []
    try:
        for p in prompts:
            out = gen(model, tok, p["text"])
            rows.append({
                "prompt_id": p["id"],
                "category": p["category"],
                "class": classify(out, p["category"]),
                "preview": out[:80].replace("\n", " "),
            })
    finally:
        unregister_hooks(handles)
    return rows


def main():
    print(f"loading {MODEL_ID}...")
    model, tok = load_model(MODEL_ID)
    payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    direction = payload["refusal_direction"][SOURCE_LAYER].clone()
    print(f"source layer = {SOURCE_LAYER}  ||direction|| = {direction.norm().item():.2f}")

    prompts = load_diag_prompts()
    print(f"prompts: {len(prompts)} (10 harmful, 2 benign)\n")

    print("running baseline (no hooks)...")
    rows_baseline = run_config(model, tok, prompts, lambda: [])

    print("running every-layer ablation with direction[14] (F5 protocol)...")
    rows_every = run_config(
        model, tok, prompts,
        lambda: register_ablation_hooks(model, direction),
    )

    print("running single-layer ablation at block 13...")
    rows_single = run_config(
        model, tok, prompts,
        lambda: register_ablation_hooks_at_indices(model, direction, [13]),
    )

    print("\n" + "="*84)
    print(" SUMMARY")
    print("="*84)
    print(f"  {'config':<48s}  {'h_REF':>5s} {'h_DEG':>5s} {'h_JBC':>5s} | "
          f"{'b_COR':>5s} {'b_DEG':>5s}")
    for label, rows in [
        ("baseline (no hooks)", rows_baseline),
        ("every-layer ablation, direction[14]", rows_every),
        ("single-layer ablation, block 13, direction[14]", rows_single),
    ]:
        h_ref, h_deg, h_jbc, b_cor, b_deg = summarize(rows)
        print(f"  {label:<48s}  {h_ref:>5d} {h_deg:>5d} {h_jbc:>5d}   "
              f"{b_cor:>5d} {b_deg:>5d}")

    print("\n" + "="*84)
    print(" PER-PROMPT GRID")
    print("="*84)
    print("  cat: " + " ".join(
        " H " if i in HARMFUL_INDICES else " B " for i in range(len(prompts))
    ))
    for label, rows in [
        ("baseline                              ", rows_baseline),
        ("every-layer ablation                  ", rows_every),
        ("single-layer ablation (block 13)      ", rows_single),
    ]:
        cells = " ".join(r["class"][:3] for r in rows)
        print(f"  {label:<40s}  {cells}")

    print("\n" + "="*84)
    print(" SAMPLE OUTPUTS  (first 3 prompts per config)")
    print("="*84)
    for label, rows in [
        ("baseline", rows_baseline),
        ("every-layer ablation", rows_every),
        ("single-layer (block 13)", rows_single),
    ]:
        print(f"\n  [{label}]")
        for r in rows[:3]:
            print(f"    P{r['prompt_id']} ({r['category']:>6s}, {r['class']}): "
                  f"{r['preview']!r}")

    unload(model)


if __name__ == "__main__":
    main()
