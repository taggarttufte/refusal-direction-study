"""Centralized matplotlib style for refusal-direction-study figures.

Usage:
    from src.style import apply, COLORS, MARKERS, label_for
    apply()
    ax.plot(..., color=COLORS["gemma3_1b"], label=label_for("gemma3_1b"))

Conventions:
- Qwen models use cool blue-teal tones (light -> dark with size).
- Gemma models use warm orange-red tones (light -> dark with generation).
- A reserved red `accent_broken` is used ONLY for "fails methodology /
  produces degenerate output" indicators. Readers should learn to associate
  this color with negative findings.
- A reserved green `accent_works` is used for "produces coherent harmful
  compliance / methodology succeeds" outcomes.
- Layer-type markers: star (*) for full-attention, circle (o) for
  sliding-window attention. Empty (.) for embedding layers.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# Per-model colors. Cool blues for Qwen, warm oranges/reds for Gemma.
COLORS: dict[str, str] = {
    # Qwen 2.5 family — light to dark blue-teal as size increases
    "qwen25_0p5b": "#9ecae1",
    "qwen25_1p5b": "#3182bd",
    "qwen25_3b":   "#08519c",
    # Gemma family — light to dark warm tones across generations
    "gemma2_2b":   "#fdae6b",
    "gemma3_1b":   "#e6550d",
    "gemma4_e2b":  "#a63603",
    # Semantic accents — meaning, not identity
    "accent_broken":  "#d62728",   # methodology fails / degenerate
    "accent_works":   "#2ca02c",   # coherent compliance
    "accent_partial": "#ff9f1a",   # coherent non-refusal but not full compliance
    "accent_refused": "#7f7f7f",   # baseline / refused after intervention (neutral)
}


# Pretty labels for legends.
LABELS: dict[str, str] = {
    "qwen25_0p5b": "Qwen 2.5 0.5B",
    "qwen25_1p5b": "Qwen 2.5 1.5B",
    "qwen25_3b":   "Qwen 2.5 3B",
    "gemma2_2b":   "Gemma 2 2B",
    "gemma3_1b":   "Gemma 3 1B",
    "gemma4_e2b":  "Gemma 4 E2B",
}


# Markers per attention type.
MARKERS: dict[str, str] = {
    "full_attention":     "*",
    "sliding_attention":  "o",
    "embedding":          ".",
    "n/a":                "o",
}


# Categorical palette for the coherence matrix (rows=models, cols=layers, cells colored by outcome).
COHERENCE_PALETTE: dict[str, str] = {
    "compliance":   COLORS["accent_works"],     # coherent harmful compliance
    "non_refusal":  COLORS["accent_partial"],   # coherent text but not full compliance
    "refused":      COLORS["accent_refused"],   # still refused
    "degraded":     COLORS["accent_broken"],    # broken model
}


def label_for(short: str) -> str:
    """Pretty label for a model short_name (falls back to short itself)."""
    return LABELS.get(short, short)


def color_for(short: str) -> str:
    """Color for a model short_name (raises if unknown)."""
    return COLORS[short]


def apply() -> None:
    """Set matplotlib rcParams for the project's figure style.

    Call once at the top of any plotting script.
    """
    mpl.rcParams.update({
        # Typography
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.labelweight":  "bold",
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        # Layout
        "figure.dpi":        140,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
        # Spines / grid
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,
        "grid.linestyle":    "-",
        # Lines
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
        # Legend
        "legend.frameon":    False,
        "legend.borderpad":  0.4,
        # Math
        "mathtext.default":  "regular",
    })


__all__ = [
    "apply", "COLORS", "LABELS", "MARKERS", "COHERENCE_PALETTE",
    "color_for", "label_for",
]
