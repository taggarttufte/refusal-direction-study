"""Re-verify F35 (cumulative-dose hypothesis) at N=12.

Original F35 was based on N=3 prompts. This re-runs the same skip-blocks
configurations on the standard 12-prompt diagnostic set so we can report
per-condition rates instead of single-prompt anecdotes.

For each (config, prompt) pair, classify the output as:
  - REF (refused with substring match)
  - DEG (degenerate per is_degraded)
  - JB  (coherent harmful compliance — guess from substring overlap with
        the prompt's intent + neither REF nor DEG; this is heuristic and
        we'll spot-check by hand)
  - OTH (coherent but not real compliance — circular, prompt-echo, or
        non-responsive; falls into "neither REF nor DEG nor obvious JB")
  - For benign prompts: COR (correct/coherent), DEG, REF, OTH.

Reports a 6×12 grid (configs × prompts) plus per-config aggregate stats.
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
    register_ablation_hooks, register_ablation_hooks_excluding,
    register_ablation_hooks_at_indices, unregister_hooks,
)
from src.diag_prompts import load_diag_prompts, HARMFUL_INDICES, BENIGN_INDICES
from src.extract import format_prompt, load_model, unload
from src.refusal_classifier import is_degraded, is_refusal


MODEL_ID = "google/gemma-3-1b-it"
PT_PATH  = ROOT / "activations" / "gemma3_1b.pt"
SOURCE_LAYER = 14   # residual_stream index; ‖d‖ ~676 in this model

# (label, hook_fn_args). hook_fn_args is one of:
#   ("all",)            -> register at all blocks
#   ("none",)           -> baseline, no hooks
#   ("excl", [...])     -> hooks at all blocks EXCEPT these indices
#   ("only", [...])     -> hooks at ONLY these indices
CONFIGS = [
    ("baseline (no hooks)",          ("none",)),
    ("ABL all 26 (control)",         ("all",)),
    ("ABL skip {11,12}",             ("excl", [11, 12])),
    ("ABL skip {10,11,12,13}",       ("excl", [10, 11, 12, 13])),
    ("ABL skip {11}",                ("excl", [11])),
    ("ABL skip {12}",                ("excl", [12])),
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
    """Return one of: REF, DEG, COR, OTH (per category meaning)."""
    if is_degraded(text):
        return "DEG"
    if is_refusal(text):
        return "REF"
    # benign + not degraded + not refused = correct-or-other
    # harmful + not degraded + not refused = potential jailbreak (coherent
    # non-refusal — flag for spot-check)
    if category == "benign":
        return "COR"  # may need spot-check, but generally coherent + not refused = ok
    return "JBC"  # coherent non-refusal on a harmful prompt; "JaiBreak Candidate"


def apply_hooks(model, direction, mode, args):
    if mode == "none":
        return []
    if mode == "all":
        return register_ablation_hooks(model, direction)
    if mode == "excl":
        return register_ablation_hooks_excluding(model, direction, args)
    if mode == "only":
        return register_ablation_hooks_at_indices(model, direction, args)
    raise ValueError(f"unknown mode {mode!r}")


def main():
    print(f"loading {MODEL_ID}...")
    model, tok = load_model(MODEL_ID)
    payload = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    direction = payload["refusal_direction"][SOURCE_LAYER].clone()
    print(f"source layer = {SOURCE_LAYER}  ||direction|| = {direction.norm().item():.2f}")

    prompts = load_diag_prompts()
    print(f"prompts: {len(prompts)} (harmful: {len(HARMFUL_INDICES)}, "
          f"benign: {len(BENIGN_INDICES)})\n")

    # results[config_idx][prompt_idx] = {"label": ..., "text": ..., "out": ...}
    results: list[list[dict]] = []
    for ci, (label, hook_args) in enumerate(CONFIGS):
        print(f"running config: {label}")
        per_prompt = []
        for pi, p in enumerate(prompts):
            mode = hook_args[0]
            args = hook_args[1] if len(hook_args) > 1 else None
            handles = apply_hooks(model, direction, mode, args)
            try:
                out = gen(model, tok, p["text"])
            finally:
                unregister_hooks(handles)
            cls = classify(out, p["category"])
            per_prompt.append({
                "prompt_id": pi,
                "category": p["category"],
                "class": cls,
                "len": len(out),
                "preview": out[:80].replace("\n", " "),
            })
        results.append(per_prompt)

    # render grid
    print("\n" + "="*84)
    print(" GRID  (rows = configs, cols = prompt id)")
    print(" Class codes:  REF=refused  DEG=degenerate  COR=benign-coherent  JBC=jailbreak-candidate (coherent, not refused)")
    print("="*84)
    header = "  " + " ".join(f"P{i:>2d}" for i in range(len(prompts)))
    print(f"\n  cat  : " + " ".join(
        " H " if i in HARMFUL_INDICES else " B " for i in range(len(prompts))
    ))
    for ci, (label, _) in enumerate(CONFIGS):
        cells = "  ".join(r["class"][:3] for r in results[ci])
        print(f"  {label:<30s} " + cells)

    # per-config aggregate stats (over harmful prompts only — JBC = jailbroken)
    print("\n" + "="*84)
    print(" PER-CONFIG STATS  (harmful prompts only, n=10)")
    print("="*84)
    print(f"  {'config':<30s}  REF  DEG  JBC  (sum should be 10)")
    for ci, (label, _) in enumerate(CONFIGS):
        harm_results = [results[ci][i] for i in HARMFUL_INDICES]
        ref = sum(r["class"] == "REF" for r in harm_results)
        deg = sum(r["class"] == "DEG" for r in harm_results)
        jbc = sum(r["class"] == "JBC" for r in harm_results)
        print(f"  {label:<30s}  {ref:>3d}  {deg:>3d}  {jbc:>3d}")

    # per-config benign coherence
    print("\n" + "="*84)
    print(" BENIGN COHERENCE  (n=2)")
    print("="*84)
    print(f"  {'config':<30s}  COR  DEG  REF  OTH")
    for ci, (label, _) in enumerate(CONFIGS):
        ben_results = [results[ci][i] for i in BENIGN_INDICES]
        cor = sum(r["class"] == "COR" for r in ben_results)
        deg = sum(r["class"] == "DEG" for r in ben_results)
        ref = sum(r["class"] == "REF" for r in ben_results)
        oth = sum(r["class"] == "JBC" for r in ben_results)  # JBC code reused on benign
        print(f"  {label:<30s}  {cor:>3d}  {deg:>3d}  {ref:>3d}  {oth:>3d}")

    # interesting examples to inspect
    print("\n" + "="*84)
    print(" SAMPLE OUTPUTS  (one per (config, category) for spot-check)")
    print("="*84)
    for ci, (label, _) in enumerate(CONFIGS):
        # pick first JBC harmful + first DEG harmful + benign
        harm_results = [(i, results[ci][i]) for i in HARMFUL_INDICES]
        for cls in ("JBC", "DEG", "REF"):
            sample = next((r for _, r in harm_results if r["class"] == cls), None)
            if sample:
                p_text = prompts[sample["prompt_id"]]["text"]
                print(f"\n  [{label}]  {cls}  on P{sample['prompt_id']}")
                print(f"    prompt:  {p_text[:80]!r}")
                print(f"    output:  {sample['preview']!r}")
                break

    unload(model)


if __name__ == "__main__":
    main()
