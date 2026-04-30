"""Day-1 pre-flight load test.

Goal: confirm we can (a) load Qwen 2.5 and Gemma 4 in this env,
(b) capture per-layer hidden states, (c) identify per-layer attention
type for Gemma (so we can label local vs global later).

We use the smallest variants of each family to keep the download cheap.
"""

import gc
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def gpu_free_gb():
    return (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()) / 1e9


def test_model(model_id: str, label: str):
    print(f"\n{'='*60}\n {label}: {model_id}\n{'='*60}")
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  config OK. type={cfg.model_type}, "
              f"hidden_layers={getattr(cfg, 'num_hidden_layers', '?')}, "
              f"hidden_size={getattr(cfg, 'hidden_size', '?')}")
        # Print attention pattern info if present
        for k in ("sliding_window", "layer_types", "attention_pattern",
                  "use_per_layer_embeddings", "per_layer_embedding"):
            if hasattr(cfg, k):
                print(f"  cfg.{k} = {getattr(cfg, k)}")
    except Exception as e:
        print(f"  CONFIG FAILED: {type(e).__name__}: {e}")
        return False

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"  tokenizer OK. vocab={tok.vocab_size}")
    except Exception as e:
        print(f"  TOKENIZER FAILED: {type(e).__name__}: {e}")
        return False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        print(f"  model loaded. params={sum(p.numel() for p in model.parameters())/1e9:.2f}B  "
              f"vram_used={(torch.cuda.memory_allocated()/1e9):.2f}GB  "
              f"vram_free={gpu_free_gb():.2f}GB")
    except Exception as e:
        print(f"  MODEL LOAD FAILED: {type(e).__name__}: {e}")
        return False

    try:
        msgs = [{"role": "user", "content": "How do I make a cake?"}]
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            enc = tok(text, return_tensors="pt").to("cuda")
        else:
            enc = tok("How do I make a cake?", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        print(f"  forward + hidden_states OK. "
              f"layers={len(hs)}, shape_per_layer={tuple(hs[0].shape)}")
    except Exception as e:
        print(f"  FORWARD/HOOK FAILED: {type(e).__name__}: {e}")
        del model
        torch.cuda.empty_cache(); gc.collect()
        return False

    # Inspect a layer to see if we can identify local vs global attention
    try:
        layer0 = model.model.layers[0]
        print(f"  layer0 class: {type(layer0).__name__}")
        # Look for attention module
        for name, mod in layer0.named_modules():
            cn = type(mod).__name__
            if "Attention" in cn or "attn" in name.lower():
                print(f"    sub: {name} -> {cn}")
                for attr in ("sliding_window", "is_sliding", "layer_idx",
                             "attention_type"):
                    if hasattr(mod, attr):
                        print(f"      {attr}={getattr(mod, attr)}")
                break
    except Exception as e:
        print(f"  layer inspection failed (non-fatal): {e}")

    del model, out
    torch.cuda.empty_cache(); gc.collect()
    return True


def main():
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"total vram = "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    targets = [
        ("Qwen/Qwen2.5-0.5B-Instruct",       "Qwen 2.5 0.5B Instruct"),
        ("google/gemma-3-1b-it",             "Gemma 3 1B IT (control / fallback)"),
        ("google/gemma-4-e2b-it",            "Gemma 4 E2B IT (primary target)"),
    ]
    results = {}
    for model_id, label in targets:
        results[label] = test_model(model_id, label)

    print("\n" + "="*60 + "\n SUMMARY\n" + "="*60)
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")


if __name__ == "__main__":
    main()
