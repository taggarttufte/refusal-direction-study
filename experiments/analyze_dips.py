"""Per-prompt flip identity at plateau, dip, and collapse layers.

For each model, re-run ablation at three diagnostic layers:
  - one plateau layer (100% ASR control)
  - one minor dip layer (90% ASR — single-flip variation)
  - one end-layer (collapse — multiple-flip variation)

For each prompt, report whether the baseline refused and whether ablation
flipped it. The grid output shows whether dips are caused by *the same*
prompt repeatedly resisting ablation (a consistent boundary case) or by
*different* prompts at different layers (prompt-layer interaction).
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
    load_direction, register_ablation_hooks, unregister_hooks,
)
from src.extract import format_prompt, load_model, unload
from src.refusal_classifier import is_refusal
from layer_sweep import HAND_AUTHORED_STRONG_REFUSERS, load_heldout_prompts


MAX_NEW_TOKENS = 80

# diagnostic layer triplets per model: (plateau, dip, collapse)
SPECS = [
    # (hf_id, short, pt_filename, [layers to test])
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen25_1p5b", "qwen25_1p5b.pt",
     [("plateau", 14), ("dip", 17), ("dip", 25), ("collapse", 28)]),
    ("Qwen/Qwen2.5-3B-Instruct",   "qwen25_3b",   "qwen25_3b.pt",
     [("plateau", 20), ("dip", 25), ("dip", 33), ("collapse", 36)]),
]


@torch.no_grad()
def gen(model, tok, prompt, max_new=MAX_NEW_TOKENS):
    enc = format_prompt(tok, prompt)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    pt = tok.decode(enc["input_ids"][0], skip_special_tokens=True)
    if full.startswith(pt):
        full = full[len(pt):]
    return full.strip()


def main():
    prompts = load_heldout_prompts()
    print(f"prompts: {len(prompts)}\n")
    # short labels for the grid columns
    labels = []
    for i, p in enumerate(prompts):
        head = p[:34]
        labels.append(f"P{i:>2d}:{head}{'…' if len(p) > 34 else ''}")
    for i, lab in enumerate(labels):
        print(f"  {lab}")
    print()

    for hf_id, short, pt_name, layers in SPECS:
        print(f"\n{'='*72}\n {short} ({hf_id})\n{'='*72}")
        pt_path = ROOT / "activations" / pt_name
        model, tok = load_model(hf_id)

        # baseline once
        print("computing baselines...")
        baselines = [gen(model, tok, p) for p in prompts]
        b_ref = [is_refusal(b) for b in baselines]
        print(f"  baseline refused: {sum(b_ref)}/{len(prompts)}")

        # for each diagnostic layer, get ablated outputs and per-prompt flip
        # status (R = refused, . = complied, F = flipped from baseline-refused)
        rows = []  # (label, layer, [per_prompt_status])
        for kind, L in layers:
            direction = load_direction(pt_path, L)
            handles = register_ablation_hooks(model, direction)
            try:
                ablated = [gen(model, tok, p) for p in prompts]
            finally:
                unregister_hooks(handles)
            a_ref = [is_refusal(a) for a in ablated]
            status = []
            for br, ar in zip(b_ref, a_ref):
                if not br:
                    status.append(".")    # baseline didn't refuse — n/a
                elif ar:
                    status.append("R")    # ablation kept refusal (resisted)
                else:
                    status.append("F")    # flipped from refused → complied
            asr = sum(s == "F" for s in status) / max(sum(b_ref), 1)
            rows.append((kind, L, status, asr, ablated))
            print(f"  L={L:>2d} ({kind:>8s})  "
                  f"per-prompt: {''.join(status)}  ASR={asr:.0%}")

        # grid: each row is a layer, each column a prompt
        print("\n  grid (rows=layers, cols=prompt index; F=flipped, R=resisted, .=baseline didn't refuse):")
        header = "      " + " ".join(f"P{i:>2d}" for i in range(len(prompts)))
        print(header)
        for kind, L, status, asr, _ in rows:
            row = f"  L={L:>2d}  " + "  ".join(s for s in status)
            print(row + f"   ASR={asr:.0%} ({kind})")

        # which specific prompts resisted (R) at each layer?
        print("\n  prompts that RESISTED ablation:")
        for kind, L, status, asr, ablated_outs in rows:
            resisters = [i for i, s in enumerate(status) if s == "R"]
            if not resisters:
                print(f"    L={L:>2d}  none")
            else:
                for i in resisters:
                    print(f"    L={L:>2d}  P{i:>2d}: {prompts[i][:60]}{'…' if len(prompts[i]) > 60 else ''}")

        unload(model)


if __name__ == "__main__":
    main()
