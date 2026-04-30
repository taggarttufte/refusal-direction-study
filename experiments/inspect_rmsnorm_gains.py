"""Inspect RMSNorm gain magnitudes across our model lineup.

The "RMSNorm gain calibration" hypothesis (a) for Gemma's ablation
fragility predicts that Gemma 3/4 have systematically larger learned
gain values in their per-block RMSNorm layers than Qwen / Gemma 2.
Larger gains amplify perturbations introduced by projection-out
ablation through downstream norms, accumulating destructive interference
across the network.

This script reads `model.layers[i].input_layernorm.weight` and
`model.layers[i].post_attention_layernorm.weight` for each block in
each model, computes per-block magnitude statistics, and compares
across the 4-model lineup (Qwen 1.5B + Gemma 2 / 3 / 4).

We load on CPU to avoid GPU memory pressure (no inference needed —
only parameter inspection).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)


MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B"),
    ("google/gemma-2-2b-it",       "Gemma 2 2B"),
    ("google/gemma-3-1b-it",       "Gemma 3 1B"),
    ("google/gemma-4-e2b-it",      "Gemma 4 E2B"),
]


def find_layers(model):
    """Return the list of transformer-block modules."""
    paths = [
        "model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
    ]
    for path in paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(f"could not locate layers on {type(model).__name__}")


def inspect_one(hf_id: str, label: str):
    print(f"\n{'='*72}\n {label}  ({hf_id})\n{'='*72}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    blocks = find_layers(model)
    print(f"  blocks: {len(blocks)}")

    # Find what RMSNorm sub-modules exist on a block by looking at the first one
    sample = blocks[0]
    norm_attrs = []
    for name, mod in sample.named_modules():
        if "norm" in name.lower() and hasattr(mod, "weight"):
            norm_attrs.append(name)
    print(f"  norm submodules on block 0: {norm_attrs}")

    # For each named norm, collect per-block ‖gain‖ stats. Skip per-block if
    # a model has heterogeneous norm submodules across blocks (e.g., Gemma 4's
    # attention module has q_norm but not k_norm despite named_modules listing
    # both on block 0).
    for norm_name in norm_attrs:
        gains = []
        skipped = 0
        for b in blocks:
            try:
                obj = b
                for part in norm_name.split("."):
                    obj = getattr(obj, part)
                if not hasattr(obj, "weight") or obj.weight is None:
                    skipped += 1
                    continue
                w = obj.weight.detach().float()
                gains.append({
                    "norm": w.norm().item(),
                    "mean_abs": w.abs().mean().item(),
                    "max_abs": w.abs().max().item(),
                    "min_abs": w.abs().min().item(),
                    "n": w.numel(),
                })
            except AttributeError:
                skipped += 1
                continue
        if not gains:
            print(f"\n  {norm_name}: skipped ({skipped} blocks lacked this submodule)")
            continue

        norms = [g["norm"] for g in gains]
        mean_abs_per_block = [g["mean_abs"] for g in gains]
        print(f"\n  {norm_name}: n_per_block={gains[0]['n']}")
        print(f"    ‖gain‖ across blocks: mean={sum(norms)/len(norms):.3f}  "
              f"min={min(norms):.3f}  max={max(norms):.3f}")
        print(f"    mean|gain| (avg per block of |entry|): "
              f"avg={sum(mean_abs_per_block)/len(mean_abs_per_block):.3f}  "
              f"min={min(mean_abs_per_block):.3f}  "
              f"max={max(mean_abs_per_block):.3f}")
        # Quick first-vs-last comparison
        print(f"    first block mean|gain| = {mean_abs_per_block[0]:.3f}, "
              f"last block mean|gain| = {mean_abs_per_block[-1]:.3f}")

    del model
    import gc
    gc.collect()


def main():
    for hf_id, label in MODELS:
        try:
            inspect_one(hf_id, label)
        except Exception as e:
            print(f"  FAILED {label}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
