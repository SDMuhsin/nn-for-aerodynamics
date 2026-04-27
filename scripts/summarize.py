"""Aggregate per-seed score.json files into one results/summary.md.

For each ablation:
  - Read score.json from every seed run found
  - Compute median and IQR over 8 (or fewer) seeds for ML/Physics/OOD/Global
  - Compare to the paper target from CONTEXT.md §3.2
  - Mark pass/fail at ±2 score-point tolerance

Usage: python scripts/summarize.py [--write_csv]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Headline targets (median Global Score) from CONTEXT.md §3.2
PAPER_TARGETS = {
    "mlp": 22,
    "gnn": 28,
    "s2v": 36,
    "s2v_gnn": 36,   # ≈ S2V (negative-result ablation)
    "trail": 43,
    "polar": 45,
    "sine": 44,
    "sph": 47,
    "inlet": 52,
    "geompnn": 54,
}

ABL_ORDER = ["mlp", "gnn", "s2v", "s2v_gnn", "trail", "polar", "sine", "sph", "inlet", "geompnn"]
TOLERANCE = 2.0


def collect(ab: str):
    rows = []
    ab_dir = RESULTS_DIR / ab
    if not ab_dir.is_dir():
        return rows
    for seed_dir in sorted(ab_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        score_path = seed_dir / "score.json"
        if not score_path.exists():
            continue
        score = json.loads(score_path.read_text())
        rows.append({
            "seed": seed_dir.name,
            "ML": score["ML"],
            "Physics": score["Physics"],
            "OOD": score["OOD"],
            "Global": score["Global"],
        })
    return rows


def stats(values):
    if not values:
        return None
    if len(values) == 1:
        return {"median": values[0], "min": values[0], "max": values[0], "iqr": 0.0}
    s = sorted(values)
    n = len(s)
    median = statistics.median(s)
    q1 = statistics.median(s[:n // 2])
    q3 = statistics.median(s[(n + 1) // 2:])
    return {"median": median, "min": s[0], "max": s[-1], "iqr": q3 - q1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_csv", action="store_true")
    args = parser.parse_args()

    summary = {}
    for ab in ABL_ORDER:
        rows = collect(ab)
        summary[ab] = {
            "n": len(rows),
            "rows": rows,
            "stats": {k: stats([r[k] for r in rows]) for k in ("ML", "Physics", "OOD", "Global")},
        }

    out = RESULTS_DIR / "summary.md"
    lines = ["# GeoMPNN reproduction — results summary", "",
             f"Aggregated from {sum(s['n'] for s in summary.values())} seed runs.", ""]
    lines += ["| Ablation | n | ML (med) | Physics | OOD | Global (med [iqr]) | Paper target | Δ | Pass? |",
              "|----------|--:|---------:|--------:|----:|--------------------|-------------:|--:|:-----:|"]
    for ab in ABL_ORDER:
        s = summary[ab]
        n = s["n"]
        st = s["stats"]
        if n == 0:
            lines.append(f"| {ab} | 0 | — | — | — | — | {PAPER_TARGETS[ab]} | — | ⏸ |")
            continue
        ml = st["ML"]["median"]
        ph = st["Physics"]["median"]
        ood = st["OOD"]["median"]
        gl = st["Global"]["median"]
        iqr = st["Global"]["iqr"]
        delta = gl - PAPER_TARGETS[ab]
        ok = "✅" if abs(delta) <= TOLERANCE else "❌"
        lines.append(
            f"| {ab} | {n} | {ml:5.2f} | {ph:5.2f} | {ood:5.2f} | {gl:5.2f} [{iqr:5.2f}] | {PAPER_TARGETS[ab]} | {delta:+.2f} | {ok} |"
        )
    lines += ["", "## Per-seed details", ""]
    for ab in ABL_ORDER:
        s = summary[ab]
        if s["n"] == 0:
            continue
        lines += [f"### {ab}", "", "| Seed | ML | Physics | OOD | Global |", "|------|---:|--------:|----:|-------:|"]
        for r in s["rows"]:
            lines.append(f"| {r['seed']} | {r['ML']:.2f} | {r['Physics']:.2f} | {r['OOD']:.2f} | {r['Global']:.2f} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")

    if args.write_csv:
        import csv
        csv_path = RESULTS_DIR / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ablation", "seed", "ML", "Physics", "OOD", "Global"])
            for ab in ABL_ORDER:
                for r in summary[ab]["rows"]:
                    w.writerow([ab, r["seed"], r["ML"], r["Physics"], r["OOD"], r["Global"]])
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
