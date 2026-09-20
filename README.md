# Refusal-Direction Ablation Across the Gemma Family

A small empirical study testing how the "refusal in language models is mediated by a single direction" finding ([Arditi et al., NeurIPS 2024](https://arxiv.org/abs/2406.11717)) transfers across model families and architectures, with a particular focus on the Gemma generation transition (Gemma 2 → Gemma 3 → Gemma 4). Includes a mechanistic explanation, backed by direct parameter inspection, for why Gemma 3 specifically behaves differently from its predecessor and successor.

## Headline findings

- **Replication on Qwen 2.5 1.5B confirmed cleanly.** Every-layer projection-out ablation with the difference-of-means refusal direction at layer 14 produces 10/10 coherent harmful compliance on a 12-prompt held-out set, with 0/10 degenerate output. Verified at N=12 with explicit coherence checks (substring-classifier + degradation heuristic).
- **Cross-Gemma asymmetry is dramatic.** Single-layer ablation works on Gemma 3 1B (broadly: 4 of 15 cells in a 5×3 block × direction matrix produce clean jailbreaks) but largely fails on Gemma 2 2B (1/15 working) and Gemma 4 E2B (0/15 working). Within the Gemma family, Gemma 3 is the architectural outlier — not the rule.
- **Mechanistic explanation: Gemma 3 has anomalously inflated post-norm RMSNorm gains.** The `post_attention_layernorm` and `post_feedforward_layernorm` mean |gain| values are 5–30× larger than in Gemma 2 or Qwen, and Gemma 4 corrected the calibration. These post-norms amplify per-block residual-stream perturbations as they propagate, making single-layer interventions disproportionately impactful on Gemma 3 specifically.
- **Refusal direction is real and meaningful, not a noise artifact.** Random unit vectors of equal magnitude have near-zero behavioral effect at any tested block (control experiment ruling out the "ablation works because we're injecting noise" alternative).

The corrected high-level story: rather than "refusal-direction ablation generalizes universally," the technique works on Qwen and on Gemma 3 with a single hook, but Gemma 2 and Gemma 4 require either multi-direction interventions (cone-style, à la [Wollschläger et al., ICML 2025](https://arxiv.org/abs/2502.17420)) or higher-magnitude protocols to overcome refusal.

## How the technique works

For each model and each layer, average the residual-stream activation at the last token across N harmful prompts (AdvBench), and across N harmless prompts (Alpaca). The difference of those two means is a vector in residual stream space — the *refusal direction*. To test whether this direction causally controls refusal behavior, register a forward hook at every transformer block (or just one) that subtracts the direction's projection from the residual stream during inference. If the direction is causally controlling refusal, the model should now comply with harmful prompts that it would otherwise refuse. The Qwen 1.5B every-layer ablation result above (10/10 coherent compliance, 0/10 degenerate) is the clean reference case; the Gemma family results document where this technique works and where it doesn't, with `RESULTS.md` F44 explaining why.

## Models tested

| Family | Sizes | Architecture |
|---|---|---|
| Qwen 2.5 | 0.5B / 1.5B / 3B Instruct | Dense, full attention throughout |
| Gemma 2 | 2B-IT | Alternating sliding/full attention, GeGLU, large sliding window (4096) |
| Gemma 3 | 1B-IT | Alternating sliding/full, GeGLU, smaller sliding window (1024), inflated post-norm gains |
| Gemma 4 | E2B-IT | Alternating + Per-Layer Embeddings, corrected post-norm calibration |

All six models tested at fp16 on a single 12 GB consumer GPU (RTX 3080 Ti). No HPC required; total compute ~2 hours.

## Detailed documentation

The day-to-day work and full findings are in three files:

| File | What's in it |
|---|---|
| [`PLAN.md`](PLAN.md) | Project plan, schedule, risks, model lineup, methodology |
| [`RESULTS.md`](RESULTS.md) | Structured findings table (F1–F45), open questions, pre-registered hypotheses, full record of retractions |
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | Append-only chronological experiment log, organized by date |

The findings include several explicit retractions (F25, F26, F27 partial, F31, F33, F34, F35, F36 mechanism, F39 universality, F43). These are documented as part of the rigor — every wrong hypothesis is a step toward the correct one, and the retraction trail is preserved on purpose.

## Repository structure

```
.
├── README.md              ← you are here
├── PLAN.md                project plan + methodology
├── LAB_NOTEBOOK.md        chronological experiment log
├── RESULTS.md             structured findings F1–F45 + retractions
│
├── src/                   reusable library modules
│   ├── extract.py             activation collection + difference-of-means
│   ├── ablate.py              forward-hook plumbing for ablation, amplification, replace
│   ├── refusal_classifier.py  substring refusal + heuristic degradation classifier
│   ├── style.py               consistent figure styling
│   ├── diag_prompts.py        standard 12-prompt diagnostic set
│   └── get_data.py            downloads AdvBench harmful + Alpaca harmless prompts
│
├── experiments/           runner scripts — one entry point per experiment block
│   ├── preflight*.py          environment + model-loading sanity checks
│   ├── run_extract_all.py     extract refusal directions for all 6 models
│   ├── analyze.py             per-layer norm + cosine plots
│   ├── layer_sweep.py         ASR-vs-depth sweep (Qwen)
│   ├── alpha_scan.py          amplification-magnitude scan
│   ├── inspect_rmsnorm_gains.py    architectural parameter inspection (F44)
│   ├── cross_model_check.py   cross-architecture gut-check
│   ├── smoke_*.py             ablation / amplification smoke tests
│   ├── diagnose_*.py          ~15 diagnostic scripts, one per hypothesis test
│   └── replot_*.py, plot_*.py, salvage_*.py    figure regeneration helpers
│
├── activations/           saved per-model directions + class-mean activations (.pt files)
├── data/                  AdvBench harmful + Alpaca harmless prompts (CSV)
└── results/               figures (.png), summary tables (.csv)
```

## Reproduction

Inference-only — no training required. Total compute: ~2 hours of overnight work on a 12 GB consumer GPU.

```bash
# 1. install dependencies
pip install torch transformers matplotlib

# 2. authenticate to HuggingFace (Gemma models are gated)
python -c "from huggingface_hub import login; login()"

# 3. download prompts
python src/get_data.py

# 4. extract refusal directions for all 6 models (~7 min)
python experiments/run_extract_all.py --n 512

# 5. validate the technique on Qwen 2.5 1.5B
python experiments/diagnose_qwen15b_reverify_n12.py

# 6. cross-Gemma matrix (the centerpiece)
python experiments/diagnose_depth_alignment_matrix.py --model gemma2_2b
python experiments/diagnose_depth_alignment_matrix.py --model gemma3_1b
python experiments/diagnose_depth_alignment_matrix.py --model gemma4_e2b

# 7. mechanistic check
python experiments/inspect_rmsnorm_gains.py
```

Saved per-model `.pt` files in `activations/` allow skipping step 4 — the data is small (~3 MB total).

## Status & next directions

Testing phase complete; writeup in progress (planned LessWrong / Alignment Forum post). Active project — three queued directions in priority order:

1. **Implement Wollschläger et al.'s gradient-based direction extraction** to find multiple orthogonal refusal directions per model, then test whether multi-direction ablation defeats Gemma 2 / Gemma 4 (which resist single-direction attempts).
2. **Reduced-magnitude every-layer ablation** (α-scan with α < 1.0) on Gemma 3 to test whether smaller per-block perturbations recover coherent jailbreak instead of degenerate output.
3. **Grey-zone evaluation on XSTest** — does the refusal direction's projection magnitude on borderline prompts correlate with refusal probability? Tests whether refusal is a continuous gauge or a binary switch.

## Related work

The four most directly-relevant references:

- **Arditi, A. et al. (2024)** — *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717. NeurIPS 2024. The foundational paper this project builds on.
- **Wollschläger, T. et al. (2025)** — *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence.* arXiv:2502.17420. ICML 2025. Showed refusal lives in multi-dimensional cones, not a single direction. Tested Qwen 2.5, Gemma 2, Llama-3 (notably not Gemma 3 or 4).
- **Kissane, C., Krzyzanowski, R., Conmy, A., Nanda, N. (2024)** — [*Base LLMs refuse too.*](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too) AI Alignment Forum, 2024-09-29. Established that refusal circuitry pre-exists fine-tuning.
- **grimjim — Norm-preserving biprojected abliteration.** Community work on preserving activation L2 norms during ablation, developed specifically because standard projection-out degrades performance on newer Gemma models. This project's F44 finding suggests a mechanistic reason (post-norm gain amplification) for why norm preservation is needed on Gemma 3 specifically.

The full bibliography — including dataset citations, architecture documentation, and adjacent work — lives in [`references/REFERENCES.md`](references/REFERENCES.md).

## Ethics & dual-use

This work studies a known white-box safety vulnerability in open-weight LLMs. The technique is not novel — it's a public method, and many "abliterated" models exist on HuggingFace already. The contribution here is empirical and mechanistic: documenting *which* models are vulnerable, *which* are not, and *why*. All experiments use publicly-available models and standard adversarial-prompt benchmarks (AdvBench). The repository contains analysis artifacts (plots, summary statistics, refusal classifications) but **excludes raw model outputs containing harmful content from logs** (see `.gitignore`), to avoid distributing additional copies of harmful generations.

If you're a model developer, the F44 finding may be relevant: post-norm gain calibration influences how much your model's refusal training survives standard ablation attempts.

## License

MIT. See [`LICENSE`](LICENSE).
