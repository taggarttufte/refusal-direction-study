# References

Papers, posts, and tools that this project drew on. Each entry notes
what we used it for and which findings (F-numbers from `RESULTS.md`) cite it.

PDFs (when desired for offline reading) go in `references/papers/`.
That subfolder is gitignored to avoid distributing copyrighted PDFs in the
repo — download from the linked URLs as needed.

---

## Foundational methods

### Arditi et al. (2024) — *Refusal in Language Models Is Mediated by a Single Direction*
- **Authors:** Andy Arditi, Oscar Balcells Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
- **Venue:** NeurIPS 2024
- **arXiv:** [2406.11717](https://arxiv.org/abs/2406.11717)
- **Code:** [github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
- **Used for:** the entire methodology — difference-of-means refusal-direction extraction, every-layer projection-out ablation, AdvBench evaluation. Our project replicates and extends this technique to the Gemma family.
- **Cited in:** F5, F8, F12, F18, README

### Wollschläger et al. (2025) — *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence*
- **Authors:** Tom Wollschläger and colleagues
- **Venue:** ICML 2025
- **arXiv:** [2502.17420](https://arxiv.org/abs/2502.17420)
- **Code:** https://www.cs.cit.tum.de/daml/geometry-of-refusal
- **Used for:** the cone-vs-single-direction framing, gradient-based direction extraction (planned future work). Tested Qwen 2.5, Gemma 2, Llama-3 (notably *not* Gemma 3 or Gemma 4 — leaving the gap our F40/F41/F42 fills).
- **Cited in:** Q7, Q8, F39, F42, README

---

## Concurrent / adjacent work

### Pan et al. (2025) — Multi-direction refusal
- **Used for:** confirmation that the cone-style refinement to Arditi is independently arrived at by multiple groups (referenced via search summaries; primary source not directly read).

### COSMIC: Generalized Refusal Direction Identification (Findings of ACL 2025)
- **PDF:** [aclanthology.org/2025.findings-acl.1310](https://aclanthology.org/2025.findings-acl.1310.pdf)
- **Used for:** background on alternative direction-extraction methodologies.

### Comparative Analysis of LLM Abliteration Methods (2025)
- **arXiv:** [2512.13655](https://arxiv.org/abs/2512.13655)
- **Used for:** context on the current state of cross-architecture abliteration tooling (Heretic, DECCP, ErisForge, FailSpy). Documents that abliteration of various models is well-explored ground.

### Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability (2025)
- **arXiv:** [2504.04215](https://arxiv.org/abs/2504.04215)
- **Used for:** related but tangential — addresses refusal in model-compression contexts.

### grimjim — *Norm-Preserving Biprojected Abliteration* (HuggingFace blog)
- **URL:** [huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
- **Used for:** community context on why standard projection-out abliteration *degrades* Gemma 3 specifically. Multiple `Gemma-3-27b-Abliterated-Normpreserve` models exist on HuggingFace because of this. Supports our F44 mechanism (post-norm gain amplification → degeneracy under naive ablation).
- **Cited in:** F44, F25 reinterpretation, README

---

## Framing & background

### Kissane, Krzyzanowski (robertzk), Conmy, Nanda (2024) — *Base LLMs refuse too*
- **Venue:** AI Alignment Forum, 2024-09-29
- **URL:** [alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)
- **Used for:** the framing pivot in our project — base/pretrained models already exhibit refusal circuitry before any safety post-training, so chat fine-tuning *amplifies* refusal mechanisms rather than installing them. This rules out a class of hypotheses about "post-training method differences" and points toward architectural causes for the cross-Gemma asymmetry we observed.
- **Cited in:** PLAN.md framing update, F26 retraction, F44, README

### Mike Lewis et al. — Original LessWrong refusal-direction post
- **URL:** [lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)
- **Used for:** the LW community version of the Arditi paper. Cite this rather than (or alongside) the arXiv link if writing for LW/AF audiences.

---

## Datasets

### AdvBench — Zou et al. (2023)
- **Title:** *Universal and Transferable Adversarial Attacks on Aligned Language Models*
- **arXiv:** [2307.15043](https://arxiv.org/abs/2307.15043)
- **Data:** [github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- **Used for:** harmful-prompt source for direction extraction (520 prompts; rows 0–511 used for training, 512–519 held out for evaluation).
- **Cited in:** PLAN.md datasets, F5, F45

### Stanford Alpaca — Taori et al. (2023)
- **Repo:** [github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- **Data:** [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) (52,002 instructions)
- **Used for:** harmless-prompt source. We filter to no-input single-instruction items < 200 chars and sample 520 to match the harmful set in count.

### XSTest — Röttger et al. (2024)
- **Title:** *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*
- **arXiv:** [2308.01263](https://arxiv.org/abs/2308.01263)
- **Used for:** queued for future grey-zone evaluation (see "Status & next directions" in README). Not yet incorporated.

---

## Model & architecture documentation

### Gemma 3 Technical Report (2025)
- **Authors:** Gemma Team, Google DeepMind
- **arXiv:** [2503.19786](https://arxiv.org/abs/2503.19786)
- **Used for:** confirming that Gemma 3 uses 4 RMSNorm layers per block with the `(1 + weight)` zero-centered convention (relevant to interpreting our F44 raw weight values), alternating sliding/full attention with reduced sliding window (4096 → 1024 in some sources, ≈ 512 in others; conflicting docs), and increased RoPE base on global layers (10K → 1M).
- **Cited in:** F43 architectural correction, F44, README

### Gemma 4 model card / overview (2026)
- **URL:** [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4)
- **Used for:** sizes (E2B, E4B, 26B MoE, 31B Dense), Per-Layer Embeddings, Apache 2.0 license. Confirmed Gemma 4 is openly available without HF gating.
- **Cited in:** PLAN.md model lineup, F11, README

### Gemma explained — Google Developers Blog
- **URL:** [developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)
- **Used for:** confirms `(1 + weight)` RMSNorm convention with stated design rationale: *"This design choice helps avoid the problematic behavior seen in some earlier Gemma implementations where weights balloon to huge values (~300) in later layers."* Our F44 finding documents that Gemma 3 still exhibits this ballooning despite the convention — Gemma 4 corrected it further.

### Building Gemma 3 from Scratch — Prashant Lakhera (Medium)
- **URL:** [devopslearning.medium.com/building-gemma-3-from-scratch-323112c544e2](https://devopslearning.medium.com/building-gemma-3-from-scratch-323112c544e2)
- **Used for:** independent confirmation of architectural details (4 norms per block, GeGLU activation, alternating attention).

### Naman Goyal — Gemma 3 Technical Deep Dive
- **URL:** [namangoyal.com/blog/2025/gemma3/](https://namangoyal.com/blog/2025/gemma3/)
- **Used for:** further confirmation of architectural specifics relevant to F43/F44.

### Zero-centered Re-parameterization of LayerNorm (Ceramic AI blog)
- **URL:** [ceramic.ai/blog/zerocentered](https://www.ceramic.ai/blog/zerocentered)
- **Used for:** background on the `(1 + weight)` convention and its training-dynamics rationale.

---

## How to add a new reference

When citing a new paper or post in a finding (F-number), add an entry here with:
1. Full citation (authors, year, title, venue if applicable)
2. Stable URL (arXiv preferred over journal landing pages where both exist)
3. **What we used it for** — one or two sentences
4. **Cited in:** which F-numbers, doc sections, or pieces of code reference it
