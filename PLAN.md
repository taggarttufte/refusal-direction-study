# Refusal Direction Across Model Families: A Cross-Architecture Study

**Author:** Taggart Tufte
**Drafted:** 2026-04-26
**Target deadline:** 2026-05-10 (10 working days, ~3 hr/day part-time)

---

## Background

Arditi et al. (2024), *"Refusal in Language Models Is Mediated by a Single Direction,"* showed that an LLM's decision to refuse a harmful request can be reduced to a single linear direction in its activation space. The procedure:

1. Run the model on N harmful prompts; record the mean residual-stream activation at each layer.
2. Run the model on N harmless prompts; record the same.
3. Subtract: `refusal_direction_layer_l = mean(harmful_l) - mean(harmless_l)`.

Two surprising properties of that direction:

- **Ablating it** (projecting it out of activations during inference) makes the model comply with harmful requests it would otherwise refuse — a one-vector jailbreak.
- **Adding more of it** makes the model refuse benign requests too.

This was originally interpreted to mean refusal training is "shallow": a single direction can be nullified, suggesting safety post-training does not deeply restructure model cognition.

**Framing update (2026-04-27):** Subsequent work by Kissane, Krzyzanowski, Conmy, and Nanda (*"Base LLMs refuse too"*, AI Alignment Forum, 2024-09-29) showed that base/pre-trained models *already* exhibit refusal circuitry before any safety post-training — base models refuse ~48% of harmful prompts (vs ~90% for their chat counterparts), with the same refusal direction as the chat models. The corrected interpretation: **chat fine-tuning amplifies and consolidates pre-existing refusal mechanisms rather than creating them.** When we ablate the refusal direction in an instruct model, we are suppressing a feature whose substrate already existed in the base model before any RLHF or safety training was applied. This reframing tightens the architectural story: differences in *how easily the direction ablates* across model families must be due to architectural properties of the residual stream itself, not to differences in post-training recipe (since the feature is largely pre-trained). See `LAB_NOTEBOOK.md` for the framing-change entry and the implications it has for our F25/F26 hypotheses.

Whether the *protocol* (every-layer projection-out ablation) generalizes across model families and architectures is unsettled. That is the specific question this project addresses.

## Research question

Does the single-direction refusal property generalize across:
1. Model families with different RLHF/safety pipelines (Qwen 2.5 from Alibaba, Gemma 3 and Gemma 4 from Google)?
2. Architectural paradigms (Qwen's dense global attention vs Gemma's alternating local/global attention, with Gemma 4 adding Per-Layer Embeddings on top)?

Six publishable sub-questions:

**On clear-cut prompts (Arditi-style binary harmful/harmless):**
- **Q1. Generalization across families.** Does Gemma 4 admit a refusal direction with comparable ablation-induced ASR as Qwen 2.5?
- **Q2. Layer-wise structure.** In the alternating-attention stacks (Gemma 3 and Gemma 4), does the refusal direction live preferentially in local-attention or global-attention layers? In Qwen, where does it peak?
- **Q3. Scale dependence.** Does the property strengthen or weaken from ~0.5B to ~3B parameters in the Qwen family? Does PLE (Gemma 3 → Gemma 4) change the picture?

**On grey-zone prompts (this is the novel contribution):**
- **Q4. Smoothness.** Does the projection of grey-zone prompt activations onto the refusal direction (extracted from clear-cut data) interpolate continuously between harmless and harmful magnitudes? If yes, the direction is a real continuous "refusal-ness" axis. If no, refusal is decision-boundary-shaped, not direction-shaped, and the "single direction" framing was misleading.
- **Q5. Cross-family divergence.** On prompts where Qwen and Gemma disagree (one refuses, the other complies), are their refusal directions similar in direction but different in threshold, or fundamentally different?
- **Q6. Intra-model stability.** Identify prompts on which a single model is unstable across resamples (refuses sometimes, complies others, at temperature 0.7). **Hypothesis:** unstable prompts have activations sitting near zero projection on the refusal direction, i.e. on the decision boundary. If true, the direction isn't just a switch but a calibration axis whose magnitude tracks model confidence.

## Models (revised after Day-1 pre-flight)

All fit in 12 GB VRAM at fp16 (RTX 3080 Ti). Pre-flight 2026-04-26 confirmed all five load and expose `output_hidden_states=True` cleanly.

| Family | Size | HF id | Layers | Hidden | VRAM | Architecture |
|---|---|---|---|---|---|---|
| Qwen 2.5 | 0.5B Instruct | `Qwen/Qwen2.5-0.5B-Instruct`  | 24 | 896  | 1.0 GB  | Dense, all `full_attention` |
| Qwen 2.5 | 1.5B Instruct | `Qwen/Qwen2.5-1.5B-Instruct`  | 28 | 1536 | ~3 GB   | Dense |
| Qwen 2.5 | 3B Instruct   | `Qwen/Qwen2.5-3B-Instruct`    | 36 | 2048 | ~6 GB   | Dense |
| Gemma 3 | 1B IT          | `google/gemma-3-1b-it`        | 26 | 1152 | 2.0 GB  | 5:1 sliding/full ratio, no PLE |
| Gemma 4 | E2B IT         | `google/gemma-4-e2b-it`       | 36 | 1536 | 10.2 GB | Alternating + PLE (5.10B total params) |

**Cut from original plan:** Gemma 4 E4B (would not fit in 12 GB at fp16). Gemma 3 1B added in its place — gives a "before-PLE" Gemma data point that isolates the PLE variable cleanly.

**Stretch:** Qwen 2.5 7B at 4-bit if Week 2 has slack.

## Datasets

- **Harmful prompts (training):** AdvBench `harmful_behaviors.csv` (520 prompts; same set Arditi used). Downloaded 2026-04-26 → `data/harmful.csv`.
- **Harmless prompts (training):** Stanford Alpaca instructions, filtered to no-input single-instruction items < 200 chars, matched count to harmful (520). Downloaded 2026-04-26 → `data/harmless.csv`.
- **Eval (held out, clear-cut):** HarmBench standard test set (200 prompts) for Q1.
- **Grey-zone (Q4–Q6):** **XSTest** (Röttger et al. 2024) — 250 "safe but looks unsafe" + 200 "unsafe" prompts, purpose-built for over-/under-refusal study. Plus optionally **OR-Bench** if XSTest feels limited.
- **Stability eval (Q6):** sample each grey-zone prompt N=10 times at temperature 0.7; refusal rate per prompt = fraction refusing.

## Pipeline

```
1. Load model + tokenizer; format with model's chat template.
2. Forward pass on N harmful prompts; capture residual stream at every layer.
3. Repeat for N harmless prompts.
4. For each layer l: direction_l = mean(harmful_l) - mean(harmless_l).
5. For each direction_l: project it out of the forward pass on held-out
   harmful prompts; generate completion; classify refused vs complied.
6. Report which layer's direction yields the highest jailbreak ASR after
   ablation; compare across models and architectures.
```

~300-500 lines of Python. Most lifted from Arditi et al.'s open-source repo; the cross-model orchestration is the original work.

## Metrics

- **Attack Success Rate (ASR)** before vs after ablation. Refusal classifier: Llama-Guard 3 + substring match (both, as cross-check).
- **KL divergence on benign tasks** before vs after ablation. Confirms ablation does not break the model on safe inputs.
- **Per-layer ASR curve.** Which layer's direction ablates refusal most cleanly.
- **Cross-model effect-size table.** All 5 models on one comparison plot.

## Schedule

### Week 1: Single-model proof of concept on Qwen 2.5 1.5B

- **Day 1.** Environment setup: transformers, torch, model download. Confirm Qwen 2.5 1.5B loads, runs inference, accepts forward hooks. Pre-flight: load Gemma 4 E2B; confirm transformers version supports it. *Failure here is the project's biggest risk.*
- **Day 2.** Implement activation collection (forward hooks on each layer's residual stream output). Verify shapes. Cache to disk.
- **Day 3.** Compute refusal direction per layer. Implement ablation via projection. Sanity check: model still produces coherent text post-ablation.
- **Day 4.** Build eval harness. Run held-out harmful prompts pre/post-ablation. Compute ASR. **Milestone: Qwen 2.5 1.5B ASR plot.**
- **Day 5.** Polish into a clean, model-agnostic library. Document.

### Week 2: Cross-model + cross-architecture + grey zone

- **Day 6.** Replicate on Qwen 2.5 0.5B and 3B. Plot ASR-by-layer for all three Qwen sizes.
- **Day 7.** Gemma 3 1B (alternating attention, no PLE — easier than Gemma 4). Validate per-layer-type analysis via `cfg.layer_types`.
- **Day 8.** Gemma 4 E2B. Most engineering risk lives here: PLE-aware activation extraction, sub-config layer types. Budget full day.
- **Day 9.** Grey-zone work. Load XSTest. For each model: run all XSTest prompts, classify outputs (refused / hedged / complied), project activations onto refusal direction, compute correlation between projection magnitude and refusal rate. Run T=0.7 stability eval (N=10 samples per prompt) on the grey-zone subset. Identify "boundary prompts" (refusal rate near 0.5).
- **Day 10.** Writeup. README, short paper (~5 pages, 6–8 plots). GitHub repo public. Optional LessWrong / Alignment Forum cross-post.

## Deliverables

1. **GitHub repo** (`taggarttufte/refusal-direction-study`): clean, runnable, with `requirements.txt`, README walking through methodology, and a one-script reproducer per model.
2. **Writeup**: short paper or detailed blog post (Anthropic Fellows / Astra submission material).
3. **Plots**: per-layer ASR curve per model; cross-model comparison; Gemma local-vs-global breakdown.
4. **Optional**: post results to LessWrong or AI Alignment Forum.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gemma 4 unsupported in current `transformers` | Medium | Day-1 pre-flight; if blocked, swap to Gemma 3 (still architecturally distinct) |
| Refusal classifier inaccurate | Medium | Use both substring match and Llama-Guard 3 as cross-check |
| 7B doesn't fit even at 4-bit | Low | Drop 7B; 5-model study still strong |
| PLE breaks naive activation extraction | Medium | Read Gemma 3n paper Day 1; allocate Day 7 buffer |
| 2 weeks turns into 4 | High (always) | Cut Gemma E4B before extending. Single-Gemma comparison is still a contribution. |

## References

- Arditi, A., Obeso, O.B., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717. — primary methodology source.
- Kissane, C., Krzyzanowski (robertzk), Conmy, A., Nanda, N. (2024-09-29). *Base LLMs refuse too.* AI Alignment Forum. https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too — establishes that refusal circuitry pre-exists fine-tuning; chat fine-tuning reinforces existing mechanisms rather than creating them.
- Wollschläger, T. et al. (2025). *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence.* arXiv:2502.17420 (ICML 2025). — refutes strict single-direction claim; finds multi-dimensional concept cones up to ~5D in tested Qwen 2.5, Gemma 2, Llama-3.
- Röttger, P. et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* — grey-zone dataset.
- Zou, A., Wang, Z., Kolter, J.Z., Fredrikson, M. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* — AdvBench source.
- Open-source code: github.com/andyrdt/refusal_direction

## Project log

- **2026-04-26 evening.** Day-1 pre-flight: env (Python 3.13.5, torch 2.6.0+cu124, transformers 5.6.2 after upgrade) and all 5 model loads confirmed. Pipeline scaffolded (`src/extract.py`, `run_extract_all.py`). Data downloaded: 520 AdvBench harmful + 520 Alpaca harmless. Overnight extraction run launched at N=512 prompts/category across all 5 models.
