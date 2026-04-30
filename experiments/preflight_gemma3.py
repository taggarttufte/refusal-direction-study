"""Quick re-run of just the Gemma 3 1B leg of the preflight after HF auth."""
from preflight import test_model
import torch

print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
ok = test_model("google/gemma-3-1b-it", "Gemma 3 1B IT")
print(f"\n{'PASS' if ok else 'FAIL'}: Gemma 3 1B IT")
