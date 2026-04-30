"""Parse Qwen 1.5B layer sweep results from the crashed log into the CSV+plot
that layer_sweep.py would have written. One-shot recovery script.
"""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
LOG = ROOT / "results" / "layer_sweep.log"

# match e.g.  "  L=14  ||d||=  10.82  abl_ref= 0/10  flips=10/10  ASR=100%  (full_attention)"
LINE_RE = re.compile(
    r"^\s*L=\s*(\d+)\s+\|\|d\|\|=\s*([\d.]+)\s+abl_ref=\s*(\d+)/(\d+)\s+"
    r"flips=\s*(\d+)/(\d+)\s+ASR=\s*(\d+)%\s+\((\w+)\)\s*$"
)
SKIP_RE = re.compile(r"^\s*L=\s*(\d+)\s+skipped\s*\(\|\|d\|\|~0\)")
N_BASE_RE = re.compile(r"baselines done in [\d.]+s; (\d+)/(\d+) refused")

rows = []
n_base_ref = None
in_qwen15b = False
with LOG.open(encoding="utf-8") as f:
    for line in f:
        if "LAYER SWEEP: qwen25_1p5b" in line:
            in_qwen15b = True
            continue
        if in_qwen15b and "LAYER SWEEP: qwen25_3b" in line:
            in_qwen15b = False
            break
        if not in_qwen15b:
            continue
        m = N_BASE_RE.search(line)
        if m:
            n_base_ref = int(m.group(1))
            n_total = int(m.group(2))
        m = SKIP_RE.search(line)
        if m:
            L = int(m.group(1))
            rows.append({"layer": L, "n_base_ref": n_base_ref or 10,
                         "n_abl_ref": n_base_ref or 10, "flips": 0,
                         "asr": 0.0, "layer_type": "embedding",
                         "norm_dir": 0.0})
            continue
        m = LINE_RE.search(line)
        if m:
            L = int(m.group(1))
            norm = float(m.group(2))
            abl_ref = int(m.group(3))
            flips = int(m.group(5))
            n_base = int(m.group(6))
            asr = int(m.group(7)) / 100.0
            ltype = m.group(8)
            rows.append({"layer": L, "n_base_ref": n_base,
                         "n_abl_ref": abl_ref, "flips": flips,
                         "asr": asr, "layer_type": ltype,
                         "norm_dir": norm})

print(f"parsed {len(rows)} rows from log")
assert n_base_ref == 10
n_layers = max(r["layer"] for r in rows) + 1

# write CSV
out_csv = ROOT / "results" / "layer_sweep_qwen25_1p5b.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_csv}")

# plot
fig, ax = plt.subplots(figsize=(10, 5))
L = [r["layer"] for r in rows]
asr = [r["asr"] for r in rows]
norm = [r["norm_dir"] for r in rows]

ax.plot(L, asr, marker="o", color="tab:red", label="ASR (flip rate)")
ax.set_xlabel("layer index (0 = embedding)")
ax.set_ylabel("ASR — fraction of baseline-refused prompts that flipped",
              color="tab:red")
ax.tick_params(axis="y", labelcolor="tab:red")
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)

ax2 = ax.twinx()
ax2.plot(L, norm, marker="s", color="tab:blue", linewidth=1, alpha=0.6,
         label="||direction||")
ax2.set_ylabel("||direction[L]||", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax.set_title(f"Layer sweep: Qwen 2.5 1.5B   "
             f"({n_base_ref}/10 prompts refused at baseline)")
fig.tight_layout()
out_png = ROOT / "results" / "layer_sweep_qwen25_1p5b.png"
fig.savefig(out_png, dpi=140)
print(f"wrote {out_png}")

# top-3 layers
print("\nTop-3 layers by ASR:")
for r in sorted(rows, key=lambda x: x["asr"], reverse=True)[:3]:
    print(f"  L={r['layer']:>2d}  ASR={r['asr']:.0%}  "
          f"||d||={r['norm_dir']:6.2f}  type={r['layer_type']}")
