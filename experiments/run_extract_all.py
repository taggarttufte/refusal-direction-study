"""Overnight runner: extract refusal directions for all 5 models.

For each model:
    1. Load
    2. Run extract_layer_means on N harmful prompts
    3. Run extract_layer_means on N harmless prompts
    4. Compute refusal direction per layer
    5. Save (means_harmful, means_harmless, directions, layer_types) to .pt
    6. Unload and move on

Tonight's defaults: N=128 prompts each. Tunable via --n.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.extract import (
    compute_refusal_directions,
    extract_layer_means,
    load_model,
    unload,
)


MODELS = [
    # (hf_id, short_name, dtype)
    ("Qwen/Qwen2.5-0.5B-Instruct",  "qwen25_0p5b", torch.float16),
    ("Qwen/Qwen2.5-1.5B-Instruct",  "qwen25_1p5b", torch.float16),
    ("Qwen/Qwen2.5-3B-Instruct",    "qwen25_3b",   torch.float16),
    ("google/gemma-2-2b-it",        "gemma2_2b",   torch.float16),  # added 2026-04-27 keystone for fragility hypothesis
    ("google/gemma-3-1b-it",        "gemma3_1b",   torch.float16),
    ("google/gemma-4-e2b-it",       "gemma4_e2b",  torch.float16),
]


def get_layer_types(model_id: str):
    """Return list[str] of attention type per transformer layer, or None."""
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        for attr in ("layer_types",):
            if hasattr(cfg, attr):
                return list(getattr(cfg, attr))
        # gemma 4 may nest under text_config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "layer_types"):
            return list(cfg.text_config.layer_types)
    except Exception as e:
        print(f"  (could not read layer_types: {e})")
    return None


def load_prompts(path: Path, n: int) -> list[str]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["goal"] for r in rows[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128,
                    help="prompts per category (default 128)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "activations",
                    help="output dir")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated short_names to run; default all")
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)
    harmful = load_prompts(ROOT / "data" / "harmful.csv", args.n)
    harmless = load_prompts(ROOT / "data" / "harmless.csv", args.n)
    print(f"loaded {len(harmful)} harmful + {len(harmless)} harmless prompts")

    todo = MODELS
    if args.only:
        keep = set(args.only.split(","))
        todo = [m for m in MODELS if m[1] in keep]

    summary = []
    for model_id, short, dtype in todo:
        print(f"\n{'='*60}\n {short}: {model_id}\n{'='*60}")
        t0 = time.time()
        try:
            model, tok = load_model(model_id, dtype=dtype)
            print("  -- harmful pass --")
            mh = extract_layer_means(model, tok, harmful)
            print("  -- harmless pass --")
            mb = extract_layer_means(model, tok, harmless)
            dirs = compute_refusal_directions(mh, mb)
            layer_types = get_layer_types(model_id)
            payload = {
                "model_id": model_id,
                "short": short,
                "n": args.n,
                "mean_harmful": mh,
                "mean_harmless": mb,
                "refusal_direction": dirs,
                "layer_types": layer_types,
                "norms_harmful": mh.norm(dim=-1).tolist(),
                "norms_harmless": mb.norm(dim=-1).tolist(),
                "norms_direction": dirs.norm(dim=-1).tolist(),
            }
            out_path = args.out / f"{short}.pt"
            torch.save(payload, out_path)
            elapsed = time.time() - t0
            print(f"  saved -> {out_path}  ({elapsed:.0f}s)")
            summary.append((short, "OK", elapsed, dirs.shape, layer_types))
            unload(model)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summary.append((short, f"FAIL: {e}", time.time() - t0, None, None))
            try:
                unload(model)
            except Exception:
                pass

    print("\n" + "="*60 + "\n SUMMARY\n" + "="*60)
    for short, status, elapsed, shape, lt in summary:
        print(f"  {short:14s}  {status:10s}  {elapsed:5.0f}s  "
              f"shape={shape}  layer_types={'yes' if lt else 'no'}")


if __name__ == "__main__":
    main()
