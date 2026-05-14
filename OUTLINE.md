# Writeup Outline — LessWrong / Alignment Forum post

**Working title:** *Why does refusal-direction ablation work on Gemma 3 but not Gemma 2 or Gemma 4? A post-norm calibration story.*

**Target audience:** AF / LessWrong — alignment researchers, mech-interp readers familiar with Arditi et al. but not specialists in Gemma internals.
**Target length:** ~3,000 words (≈ 12-min read). Stretch up to 4,000 if F44 needs more parameter-inspection prose.
**Target tone:** empirical, hypothesis-trail-honest. Lead with what's surprising, surface retractions explicitly, give the mechanism enough room to be checkable.

---

## Section-by-section budget

| § | Title | Words | Key claim(s) | Figures |
|---|---|---|---|---|
| 1 | Hook + TL;DR | 250 | The Arditi protocol replicates cleanly on Qwen, fails on Gemma 2 and Gemma 4, but works on Gemma 3 — and post-norm RMSNorm gains explain the asymmetry | — |
| 2 | Background | 350 | What Arditi found; why Kissane et al.'s "base LLMs refuse too" reframes the mechanism as pre-trained rather than installed by RLHF; what Wollschläger's cone framing adds | — |
| 3 | Setup | 300 | 5 models (Qwen 2.5 0.5/1.5/3B, Gemma 2 2B, Gemma 3 1B, Gemma 4 E2B), 12 GB consumer GPU, AdvBench + Alpaca for direction extraction, 12-prompt held-out set with explicit coherence classification | model lineup table |
| 4 | Qwen replication (positive control) | 350 | F5 + F45: every-layer ablation at L=14 produces 10/10 coherent harmful compliance, 0/10 degenerate. F8 forced-refusal α-scan confirms continuous gauge. F9 asymmetric robustness — ablation never breaks coherence, amplification breaks above α≈0.6 | `layer_sweep_qwen25_1p5b.png`, `alpha_scan_qwen15b.png` |
| 5 | The cross-Gemma surprise | 500 | F25 + F27 + F42: cross-Gemma 5×3 (block × direction) matrix. Gemma 2: 1/15 working cells. Gemma 3: 4/15 jailbreak + 4/15 break. Gemma 4: 0/15. Gemma 3 is the outlier — not Gemma broadly. Walk through the substring-classifier blind spot (F25) that originally hid this | matrix grid figure (need to make), `cosine_similarity_to_peak.png` |
| 6 | Why is Gemma 3 different? Eliminating wrong answers | 500 | Hypothesis-trail honestly: (b) alternating attention — refuted by F43 (Gemma 2 also has alternating). (c) distillation-based safety post-training — refuted by Kissane (refusal is pre-trained). (d) specific broken blocks — refuted by F34/F35. (e) cumulative dose alone — refuted by F35. The retractions are part of the rigor | none |
| 7 | The mechanism: post-norm gain calibration (F44) | 600 | Direct parameter inspection: Gemma 3 `post_attention_layernorm` mean \|gain\| = 19.76 (Gemma 2: 0.89, Qwen: 0.96); `post_feedforward_layernorm` mean \|gain\| = 33.69 (Gemma 2: 1.42). Gemma 4 corrected to 1.27 / 3.57. Why these gains amplify per-block residual perturbations 20–30× → single-layer ablation defeats refusal across many downstream blocks on Gemma 3, but only at the cost of coherence-breaking on the unusually-fragile blocks. F36 first-write fragility at block 3 fits this story (early ablation cascades the most). F38 random-direction control rules out "ablation breaks because we're injecting noise" | `inspect_rmsnorm_gains` bar chart (need to make from `.log` data) |
| 8 | Methodology lesson: classifier coherence checks | 200 | F25 retraction is the warning. Substring refusal classifiers without coherence checks systematically misclassify "broken model output" as "successful jailbreak." Anyone replicating Arditi-style work on a new architecture should publish coherence stats alongside ASR | none |
| 9 | What this means for refusal interpretability | 300 | Gemma 4's 0/15 single-layer susceptibility is consistent with Wollschläger's higher-dimensional cone picture — refusal lives in a feature subspace single-direction ablation can't fully puncture. Gemma 3's anomaly is *implementation-level* (norm calibration), not a deeper alignment property. Implication: the "single direction" framing was always architecture-dependent; F44 names one of the architectural variables that controls when it holds | none |
| 10 | Limitations + next directions | 200 | Three queued: (i) Wollschläger gradient method to test multi-direction ablation on Gemma 2/4; (ii) reduced-α every-layer ablation on Gemma 3 to recover coherent jailbreak; (iii) XSTest grey-zone correlation between projection magnitude and refusal probability. Compute is not the bottleneck (~2 hr GPU); time is | none |
| — | **Total** | **~3,550** | | 4–5 figures |

---

## Pre-draft decisions still open

These are choices I should make before writing prose; flagging now so we can resolve in conversation rather than mid-draft.

1. **Lead with the mechanism (F44) or with the cross-Gemma surprise (F42)?**
   - Lead with surprise → reader earns the mechanism. More AF-native.
   - Lead with mechanism → cleaner abstract, but the "why is this interesting" load is heavier upfront.
   - **Tentative pick:** open §1 with the surprise, then §7 delivers the mechanism. Standard mech-interp narrative arc.

2. **How honest about retractions?** RESULTS.md has 8 explicit retractions (F25, F26 partial, F27 partial, F31, F32 partial, F34, F35 partial, F39 universality, F43). Two options:
   - Footnote them ("we initially thought X; corrected after Y").
   - Dedicate §6 to walking the trail.
   - **Tentative pick:** §6 walks the trail. AF rewards epistemic transparency; collapsing 8 retractions into footnotes obscures how much the post-norm answer was *earned*.

3. **Cite Wollschläger as competitor or complement?** Their cone result was published Feb 2025 (ICML 2025). They tested Qwen 2.5 + Gemma 2 but not Gemma 3 or Gemma 4. F44 is consistent with their cone framing for Gemma 2/4 but adds a Gemma-3-specific implementation explanation they didn't reach.
   - **Tentative pick:** complement, not competitor. Frame F44 as "the implementation-level variable that explains *which* models the cone framing matters for."

4. **Include a code/repro section?** README already has it.
   - **Tentative pick:** brief paragraph at end + GitHub link. Don't duplicate the README.

5. **Title.** Working title above is descriptive but long. Alternatives:
   - *"Refusal-direction ablation breaks on Gemma 4. Here's why it worked on Gemma 3."*
   - *"Post-norm gains explain why a one-vector jailbreak works on some Gemmas and not others."*
   - *"What broke between Gemma 2 and Gemma 3 (and got fixed in Gemma 4) for refusal-direction ablation."*

---

## Figures to make / verify

| # | Figure | Source | Status |
|---|---|---|---|
| F-1 | Qwen 1.5B layer sweep — ASR vs depth | `results/layer_sweep_qwen25_1p5b.png` | exists |
| F-2 | Qwen α-scan — forced-refusal + degradation curves | `results/alpha_scan_qwen15b.png` | exists |
| F-3 | Cross-Gemma 5×3 matrix (block × source-direction → outcome) | `results/cross_gemma_matrix.png` via `experiments/plot_cross_gemma_matrix.py` | **done 2026-05-14** — see cell-count note below |
| F-4 | RMSNorm gain comparison bar chart (per-norm-type, per-model) | `results/rmsnorm_gains.png` via `experiments/plot_rmsnorm_gains.py` (data: `results/rmsnorm_gains.csv`) | **done 2026-05-14** |
| F-5 | (optional) Cosine similarity of refusal direction across layers — illustrates "stable mid-network" extraction zone | `results/cosine_similarity_to_peak.png` | exists, may not be needed |

**F-3 cell-count note (2026-05-14):** the original `.log` files were gitignored and gone, so F-3 was rebuilt by re-running `diagnose_depth_alignment_matrix.py` for all three Gemmas. That generalized runner picks direction sources by depth *fraction* (indices 4/15/26 for Gemma 2/3, 5/19/35 for Gemma 4), so Gemma 3's mid-direction here is ‖d‖≈1050 vs the original F39's dir[14] at ‖d‖≈676 — a larger direction that yields more BREAK and fewer clean JBR. **Re-run Gemma 3 result: 4 BREAK, 4 PART, 3 BASE, 2 JBR, 2 MIXED** (vs F42's "4 BASE / 4 BREAK / 4 JBR / 3 MIXED-PART"). Gemma 2 (14 BASE + 1 JBR) and Gemma 4 (15 BASE) reproduced exactly. The qualitative cross-Gemma story is unchanged and arguably starker. **§5 and §7 prose should cite the re-run counts that the figure shows, not the old F39/F42 numbers.**

---

## Open thread for next session

When picking this up:
1. Lock the §1 → §10 structure above (or revise).
2. Resolve the 5 pre-draft decisions.
3. Make F-3 and F-4 plots.
4. Begin §1 + §2 prose draft (the cheap pages — 600 words combined).

The hard sections are §5, §6, §7. §6 in particular needs careful retraction prose so the trail reads as rigor, not flailing.
