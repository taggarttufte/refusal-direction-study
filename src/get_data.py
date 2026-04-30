"""Download harmful + harmless prompt sets.

Harmful: AdvBench's harmful_behaviors.csv (Zou et al. 2023; same set used by Arditi).
Harmless: Alpaca instructions, filtered to short, single-turn, no input field.

Outputs:
    data/harmful.csv     -- "goal" column with harmful instructions
    data/harmless.csv    -- "goal" column with harmless instructions
"""
import csv
import io
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
# Stanford's released Alpaca data, mirrored on github
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/"
    "alpaca_data.json"
)


def fetch(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    random.seed(0)

    print("Downloading AdvBench harmful_behaviors...")
    advbench_bytes = fetch(ADVBENCH_URL)
    rows = list(csv.DictReader(io.StringIO(advbench_bytes.decode())))
    harmful = [r["goal"].strip() for r in rows if r.get("goal", "").strip()]
    print(f"  got {len(harmful)} harmful prompts")

    print("Downloading Stanford Alpaca instructions...")
    import json
    alpaca = json.loads(fetch(ALPACA_URL).decode())
    # Keep only instructions with no `input` field, short (<200 chars), and
    # single-sentence-ish to match AdvBench's distribution
    candidates = [
        a["instruction"].strip()
        for a in alpaca
        if not a.get("input", "").strip()
        and len(a["instruction"]) < 200
        and a["instruction"].count(".") <= 2
    ]
    random.shuffle(candidates)
    harmless = candidates[: len(harmful)]  # match count
    print(f"  picked {len(harmless)} harmless prompts (filtered from {len(alpaca)})")

    with (DATA / "harmful.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal"])
        for h in harmful:
            w.writerow([h])
    with (DATA / "harmless.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal"])
        for h in harmless:
            w.writerow([h])
    print(f"\nWrote {DATA/'harmful.csv'} and {DATA/'harmless.csv'}")
    print(f"\nSamples:\n  harmful[0]:  {harmful[0]}")
    print(f"  harmless[0]: {harmless[0]}")


if __name__ == "__main__":
    main()
