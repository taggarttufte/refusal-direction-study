"""Direction ablation via forward hooks.

For a unit-norm direction d in residual-stream space, we modify the model
so that at every transformer block's output we replace the residual stream
x with x - (x . d) * d. This removes any component of x along d.

Why hooks on every block: Arditi et al. find that the "refusal direction"
acts as a feature throughout the network, not just at one location. Ablating
only at the layer where the direction was extracted is much weaker than
ablating it everywhere.

Usage:
    direction = load_direction(model_pt_path, layer=12)        # (hidden,)
    handles = register_ablation_hooks(model, direction)
    try:
        out = model.generate(...)
    finally:
        unregister_hooks(handles)
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# locating transformer blocks
# ----------------------------------------------------------------------------
def find_transformer_blocks(model) -> list[torch.nn.Module]:
    """Return the list of transformer-block modules across known architectures.

    Tries the common attribute paths used by HuggingFace decoder-only models.
    """
    candidates = [
        "model.layers",                  # Qwen2, Llama, Gemma3 (text-only)
        "model.language_model.layers",   # Gemma 4 (multimodal-style nesting)
        "model.text_model.layers",       # other multimodal layouts
        "transformer.h",                 # GPT-2 family
        "gpt_neox.layers",               # NeoX
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return list(obj)
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        f"add this architecture's path to find_transformer_blocks()"
    )


# ----------------------------------------------------------------------------
# the ablation operation
# ----------------------------------------------------------------------------
def _project_out(x: Tensor, d_unit: Tensor) -> Tensor:
    """Return x with the component along d_unit removed.

    x: (..., hidden)   d_unit: (hidden,)  must be unit-norm.
    """
    # (... , hidden) @ (hidden,) -> (...,)
    coef = x @ d_unit
    return x - coef.unsqueeze(-1) * d_unit


def _make_hook(d_unit: Tensor):
    """Return a forward hook that projects d_unit out of the block's output.

    HuggingFace decoder blocks return a tuple `(hidden_states, ...)`. We replace
    the first element.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            new_hs = _project_out(hs, d_unit.to(hs.device, dtype=hs.dtype))
            return (new_hs,) + output[1:]
        else:
            return _project_out(output, d_unit.to(output.device, dtype=output.dtype))
    return hook


def _make_add_hook(d_add: Tensor):
    """Return a forward hook that adds d_add to the block's output residual stream.

    d_add is the *full* vector we want to add (already scaled by alpha). The
    block output (hidden_states, ...) gets its first element shifted by d_add.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            new_hs = hs + d_add.to(hs.device, dtype=hs.dtype)
            return (new_hs,) + output[1:]
        else:
            return output + d_add.to(output.device, dtype=output.dtype)
    return hook


# ----------------------------------------------------------------------------
# register / unregister
# ----------------------------------------------------------------------------
def register_ablation_hooks(model, direction: Tensor) -> list:
    """Attach a projection-out hook to every transformer block. Return handles."""
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
    norm = direction.norm()
    if norm.item() < 1e-9:
        raise ValueError("direction has near-zero norm; nothing to ablate")
    d_unit = (direction / norm).to(dtype=torch.float32)

    blocks = find_transformer_blocks(model)
    handles = [b.register_forward_hook(_make_hook(d_unit)) for b in blocks]
    return handles


def _make_replace_hook(d_unit: Tensor, target_proj: float):
    """Replace x's scalar projection on d_unit with target_proj at every position.

    Mathematically: x_new = x - (x . d_unit - target_proj) * d_unit
                          = x with its d_unit-component set to target_proj.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            d = d_unit.to(hs.device, dtype=hs.dtype)
            curr = (hs @ d).unsqueeze(-1)         # (..., 1)
            new_hs = hs - curr * d + target_proj * d
            return (new_hs,) + output[1:]
        else:
            d = d_unit.to(output.device, dtype=output.dtype)
            curr = (output @ d).unsqueeze(-1)
            return output - curr * d + target_proj * d
    return hook


def register_replace_hooks(model, direction: Tensor,
                           mean_harmless: Tensor) -> list:
    """Project-and-replace ablation.

    Removes the projection of activations onto `direction` (just like
    register_ablation_hooks), but instead of replacing it with zero, replaces
    it with the population-mean harmless prompt's projection on the same
    direction. This keeps activations in-distribution and avoids the
    "residual stream destabilization" failure mode observed on models with
    large natural activation magnitudes (e.g., Gemma 3 / 4 with RMSNorm at
    high scale).

    direction:     the saved refusal direction (mean_harmful - mean_harmless),
                   shape (hidden,).
    mean_harmless: the saved harmless mean activation at the *same* source
                   layer, shape (hidden,). Used to compute the scalar target
                   projection.
    """
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got {tuple(direction.shape)}")
    if mean_harmless.shape != direction.shape:
        raise ValueError(
            f"mean_harmless shape {tuple(mean_harmless.shape)} != "
            f"direction shape {tuple(direction.shape)}"
        )
    norm = direction.norm()
    if norm.item() < 1e-9:
        raise ValueError("direction has near-zero norm")
    d_unit = (direction / norm).to(dtype=torch.float32)
    target_proj = float((mean_harmless.to(dtype=torch.float32) @ d_unit).item())
    blocks = find_transformer_blocks(model)
    handles = [b.register_forward_hook(_make_replace_hook(d_unit, target_proj))
               for b in blocks]
    return handles


def register_ablation_hooks_excluding(
    model,
    direction: Tensor,
    excluded_indices: list[int],
) -> list:
    """Attach projection-out hooks at every transformer block EXCEPT the
    specified excluded block indices. Inverse of register_ablation_hooks_at_indices.
    """
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got {tuple(direction.shape)}")
    norm = direction.norm()
    if norm.item() < 1e-9:
        raise ValueError("direction has near-zero norm")
    d_unit = (direction / norm).to(dtype=torch.float32)

    blocks = find_transformer_blocks(model)
    excluded = set(excluded_indices)
    handles = []
    for i, b in enumerate(blocks):
        if i not in excluded:
            handles.append(b.register_forward_hook(_make_hook(d_unit)))
    return handles


def register_ablation_hooks_at_indices(
    model,
    direction: Tensor,
    indices: list[int],
) -> list:
    """Attach projection-out hooks at ONLY the specified transformer block indices.

    For per-layer fragility tests. `indices` are 0-indexed positions in the
    list of transformer blocks (NOT hidden_states indices).

    To ablate only at the source layer, pass [source_layer - 1] (since
    hidden_states[k] is the output of transformer block k, and indexing of
    blocks is 0-based starting at 0 for the first block).
    """
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got {tuple(direction.shape)}")
    norm = direction.norm()
    if norm.item() < 1e-9:
        raise ValueError("direction has near-zero norm")
    d_unit = (direction / norm).to(dtype=torch.float32)

    blocks = find_transformer_blocks(model)
    handles = []
    for i in indices:
        if i < 0 or i >= len(blocks):
            raise IndexError(f"layer index {i} out of range [0, {len(blocks)})")
        handles.append(blocks[i].register_forward_hook(_make_hook(d_unit)))
    return handles


def register_ablation_hooks_selective(
    model,
    direction: Tensor,
    layer_types: list[str],
    allowed_types: set[str] | list[str],
) -> list:
    """Attach projection-out hooks ONLY at layers whose type is in `allowed_types`.

    For testing whether the every-layer-ablation fragility on Gemma 3/4 is
    driven by a specific attention type. layer_types must align with the
    transformer-block sequence (one entry per block, in order).
    """
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got {tuple(direction.shape)}")
    norm = direction.norm()
    if norm.item() < 1e-9:
        raise ValueError("direction has near-zero norm")
    d_unit = (direction / norm).to(dtype=torch.float32)

    blocks = find_transformer_blocks(model)
    if len(blocks) != len(layer_types):
        raise ValueError(
            f"layer_types has {len(layer_types)} entries but model has "
            f"{len(blocks)} transformer blocks"
        )
    allowed = set(allowed_types)
    handles = []
    for block, ltype in zip(blocks, layer_types):
        if ltype in allowed:
            handles.append(block.register_forward_hook(_make_hook(d_unit)))
    return handles


def register_amplification_hooks(model, direction: Tensor, alpha: float) -> list:
    """Attach an "add alpha * direction to every block's output" hook.

    direction is the raw (unnormalized) refusal direction. alpha = 1 means
    "shift activations by exactly the natural mean(harmful) - mean(harmless)
    difference." Larger alpha drives activations further into the harmful
    regime; very large alpha drives them out of distribution and the model
    degenerates.
    """
    if direction.dim() != 1:
        raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
    d_add = (alpha * direction).to(dtype=torch.float32)
    blocks = find_transformer_blocks(model)
    handles = [b.register_forward_hook(_make_add_hook(d_add)) for b in blocks]
    return handles


def unregister_hooks(handles: Iterable):
    for h in handles:
        h.remove()


# ----------------------------------------------------------------------------
# loading saved directions
# ----------------------------------------------------------------------------
def load_direction(pt_path, layer: int) -> Tensor:
    """Load the refusal direction at a specific layer index from a saved .pt.

    Layer indices match `output_hidden_states`: index 0 is embeddings, index
    L is the output of transformer block L. Direction at index L is extracted
    from after-block-L last-token activations, which is the appropriate
    "feature space" of that layer.
    """
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    dirs = payload["refusal_direction"]  # (n_layers + 1, hidden)
    if layer < 0 or layer >= dirs.shape[0]:
        raise IndexError(
            f"layer {layer} out of range; have 0..{dirs.shape[0]-1}")
    return dirs[layer].clone()
