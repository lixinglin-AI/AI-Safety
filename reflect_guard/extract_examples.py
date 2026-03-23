"""
Extract qualitative examples from ablation results for paper analysis.

Three categories:
  1. D catches, 0 and B miss  — adversarial, harmful, both baseline & SFT-only predict safe but Full does not
  2. D false positives         — benign, baseline predicts safe, D predicts unsafe (over-triggering)
  3. Both fail                 — harmful, all three (0, B, D) predict safe (method's upper bound)

Usage:
  python extract_examples.py [--n N] [--output FILE]
"""

import argparse
import json
import os
import textwrap

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load(suffix: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, f"wildguard_results_{suffix}.json")
    with open(path) as f:
        return json.load(f)


def by_prompt(data: list[dict]) -> dict[str, dict]:
    return {r["prompt"]: r for r in data}


def fmt(title: str, rec_0: dict, rec_b: dict, rec_d: dict) -> str:
    lines = [f"\n{'='*80}", f"  {title}", "="*80]
    lines.append(f"\nPROMPT:")
    lines.append(textwrap.fill(rec_0["prompt"], width=78, initial_indent="  ", subsequent_indent="  "))
    lines.append(f"\nGROUND TRUTH: {rec_0['ground_truth']}   ADVERSARIAL: {rec_0.get('adversarial')}   SUBCATEGORY: {rec_0.get('subcategory')}")
    lines.append(f"\nPREDICTIONS:")
    lines.append(f"  Condition 0 (Clean Baseline):   {rec_0['predicted']}")
    lines.append(f"  Condition B (SFT Labels Only):  {rec_b['predicted']}")
    lines.append(f"  Condition D (Full Reflect-Guard): {rec_d['predicted']}")
    if rec_d.get("reflection"):
        lines.append(f"\nREFLECTION (Condition D):")
        lines.append(textwrap.fill(rec_d["reflection"], width=78, initial_indent="  ", subsequent_indent="  "))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Examples per category")
    parser.add_argument("--output", type=str, default=None, help="Save to file")
    args = parser.parse_args()

    data_0 = by_prompt(load("clean_baseline"))
    data_b = by_prompt(load("ablation_b"))
    data_d = by_prompt(load("reflect_guard"))

    common = set(data_0) & set(data_b) & set(data_d)
    print(f"Records in all three conditions: {len(common)}")

    # ── Category 1: D catches, 0 and B miss (adversarial harmful TN→TP) ──────
    cat1 = [
        p for p in common
        if data_0[p]["ground_truth"] == "harmful"
        and data_0[p]["predicted"] == "unharmful"
        and data_b[p]["predicted"] == "unharmful"
        and data_d[p]["predicted"] == "harmful"
    ]
    # Prefer adversarial examples
    cat1_adv = [p for p in cat1 if data_0[p].get("adversarial") is True]
    cat1_sel = cat1_adv[:args.n] if len(cat1_adv) >= args.n else cat1[:args.n]

    # ── Category 2: D false positives (benign, 0 correct, D wrong) ───────────
    cat2 = [
        p for p in common
        if data_0[p]["ground_truth"] == "unharmful"
        and data_0[p]["predicted"] == "unharmful"
        and data_d[p]["predicted"] == "harmful"
    ]
    cat2_sel = cat2[:args.n]

    # ── Category 3: All fail (harmful, 0+B+D all predict safe) ──────────────
    cat3 = [
        p for p in common
        if data_0[p]["ground_truth"] == "harmful"
        and data_0[p]["predicted"] == "unharmful"
        and data_b[p]["predicted"] == "unharmful"
        and data_d[p]["predicted"] == "unharmful"
    ]
    cat3_sel = cat3[:args.n]

    output_lines = []
    output_lines.append("REFLECT-GUARD QUALITATIVE EXAMPLES")
    output_lines.append("="*80)
    output_lines.append(f"\nCategory 1 — D catches, baseline & SFT-only miss  ({len(cat1)} total, {len(cat1_adv)} adversarial)")
    output_lines.append(f"Category 2 — D false positives                   ({len(cat2)} total)")
    output_lines.append(f"Category 3 — All three conditions fail            ({len(cat3)} total)")

    output_lines.append(f"\n\n{'#'*80}")
    output_lines.append("# CATEGORY 1: Reflection catches what SFT alone and baseline miss")
    output_lines.append(f"# (harmful, adversarial, baseline=safe, SFT=safe, Full=unsafe)")
    output_lines.append(f"{'#'*80}")
    for i, p in enumerate(cat1_sel, 1):
        output_lines.append(fmt(f"Example 1.{i}", data_0[p], data_b[p], data_d[p]))

    output_lines.append(f"\n\n{'#'*80}")
    output_lines.append("# CATEGORY 2: Reflect-Guard false positives (over-triggering)")
    output_lines.append(f"# (benign, baseline=safe, Full=unsafe)")
    output_lines.append(f"{'#'*80}")
    for i, p in enumerate(cat2_sel, 1):
        output_lines.append(fmt(f"Example 2.{i}", data_0[p], data_b[p], data_d[p]))

    output_lines.append(f"\n\n{'#'*80}")
    output_lines.append("# CATEGORY 3: All conditions fail (upper bound on difficulty)")
    output_lines.append(f"# (harmful, all predict safe)")
    output_lines.append(f"{'#'*80}")
    for i, p in enumerate(cat3_sel, 1):
        output_lines.append(fmt(f"Example 3.{i}", data_0[p], data_b[p], data_d[p]))

    report = "\n".join(output_lines)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved to {args.output}")
    else:
        out_path = os.path.join(RESULTS_DIR, "qualitative_examples.txt")
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
