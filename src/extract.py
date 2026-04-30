"""Model-agnostic refusal-direction extraction.

For a given model and a list of prompts, run a forward pass on each prompt
and capture the residual-stream activation at the *last* token position
(the position right before the model would emit its first response token).
Aggregate per-layer means.

Then: refusal_direction[l] = mean(harmful[l]) - mean(harmless[l]).

Usage (programmatic):
    from src.extract import extract_layer_means, compute_refusal_directions
    means_h = extract_layer_means(model, tok, harmful_prompts)
    means_b = extract_layer_means(model, tok, harmless_prompts)
    dirs    = compute_refusal_directions(means_h, means_b)
"""
from __future__ import annotations

import gc
import time
from typing import Iterable

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------------
# model + tokenizer loading
# ----------------------------------------------------------------------------
def load_model(model_id: str, dtype: torch.dtype = torch.float16):
    """Load model in eval mode on CUDA. Returns (model, tokenizer)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def unload(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ----------------------------------------------------------------------------
# prompt formatting
# ----------------------------------------------------------------------------
def format_prompt(tok, instruction: str) -> Tensor:
    """Apply chat template, tokenize, return input_ids on cuda."""
    msgs = [{"role": "user", "content": instruction}]
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        text = instruction
    enc = tok(text, return_tensors="pt").to("cuda")
    return enc


# ----------------------------------------------------------------------------
# core extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_layer_means(
    model,
    tok,
    prompts: Iterable[str],
    *,
    verbose: bool = True,
) -> Tensor:
    """Return a tensor of shape (num_layers + 1, hidden_size) holding the mean
    last-token residual-stream activation across `prompts`, per layer.

    Layer 0 is the embedding output; layer L is the output of the L-th
    transformer block. Standard `output_hidden_states=True` semantics.
    """
    prompts = list(prompts)
    n = len(prompts)
    assert n > 0

    # Probe shape from first prompt.
    enc0 = format_prompt(tok, prompts[0])
    out0 = model(**enc0, output_hidden_states=True)
    num_layers = len(out0.hidden_states)
    hidden = out0.hidden_states[0].shape[-1]
    if verbose:
        print(f"  num_layers (incl. embed) = {num_layers}, hidden = {hidden}")

    # Accumulator on CPU in float32 for numerical stability across many adds.
    acc = torch.zeros(num_layers, hidden, dtype=torch.float32, device="cpu")

    # First prompt was already forwarded; fold its last-token activations in.
    for l, hs in enumerate(out0.hidden_states):
        acc[l] += hs[0, -1, :].float().cpu()
    del out0

    t0 = time.time()
    for i, p in enumerate(prompts[1:], start=1):
        enc = format_prompt(tok, p)
        out = model(**enc, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states):
            acc[l] += hs[0, -1, :].float().cpu()
        del out
        if verbose and (i % 25 == 0 or i == n - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{n}]  {rate:.1f} prompt/s  eta {eta:.0f}s")
    acc /= n
    return acc  # (num_layers, hidden)


def compute_refusal_directions(
    mean_harmful: Tensor, mean_harmless: Tensor
) -> Tensor:
    """direction[l] = mean_harmful[l] - mean_harmless[l]. Returns same shape."""
    assert mean_harmful.shape == mean_harmless.shape
    return mean_harmful - mean_harmless
