"""Regenerate alpha_scan_qwen15b.png from the saved CSV using the new style."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.style import apply, COLORS, label_for
apply()

CSV = ROOT / "results" / "alpha_scan_qwen15b.csv"

rows = []
with CSV.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({
            "alpha":          float(r["alpha"]),
            "refused_rate":   float(r["refused_rate"]),
            "degraded_rate":  float(r["degraded_rate"]),
            "compliant_rate": float(r["compliant_rate"]),
        })

fig, ax = plt.subplots(figsize=(8, 4.5))
a = [r["alpha"] for r in rows]
ax.plot(a, [r["compliant_rate"] for r in rows],
        marker="o", label="coherent + compliant", color=COLORS["accent_works"])
ax.plot(a, [r["refused_rate"] for r in rows],
        marker="o", label="forced refusal", color=COLORS["qwen25_1p5b"])
ax.plot(a, [r["degraded_rate"] for r in rows],
        marker="o", label="degraded (broken)", color=COLORS["accent_broken"])
ax.set_xlabel(r"$\alpha$ (multiplier on raw refusal direction)")
ax.set_ylabel("rate (fraction of harmless prompts)")
ax.set_title(f"Amplification scan: {label_for('qwen25_1p5b')} layer 14")
ax.legend(loc="center right")
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
out = ROOT / "results" / "alpha_scan_qwen15b.png"
fig.savefig(out)
print(f"wrote {out}")
