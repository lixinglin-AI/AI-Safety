#!/usr/bin/env python3
"""
Extract qualitative examples and error analysis from evaluation results.
Produces:
  - results/qualitative_examples.txt   (cases where Reflect-Guard fixed baseline errors)
  - results/error_analysis.txt         (false positives, false negatives, category breakdown)
"""

import json
import os
from collections import Counter

RESULTS_DIR = "results"
# NOTE: must match Condition 0 in the ablation study / main paper tables.
# "wildguard_results_baseline.json" is a stale, different baseline run (see
# reflect_guard_mdpi-2.tex Section 7.1 / error-analysis regeneration) and
# must not be used here.
BASELINE_PATH = os.path.join(RESULTS_DIR, "wildguard_results_clean_baseline.json")
REFLECT_PATH = os.path.join(RESULTS_DIR, "wildguard_results_reflect.json")
JBB_REFLECT_PATH = os.path.join(RESULTS_DIR, "jailbreakbench_results_reflect.json")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def truncate(text, max_len=300):
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ── Qualitative Examples ──────────────────────────────────────────────

def extract_qualitative(baseline, reflect):
    """Find cases where baseline was wrong but Reflect-Guard was correct."""

    # Baseline FN → Reflect TP (missed harmful, now caught)
    fn_fixed = []
    # Baseline FP → Reflect TN (false alarm, now correct)
    fp_fixed = []
    # Reflect FN (both missed)
    both_fn = []
    # Reflect new FP (baseline correct, reflect wrong)
    new_fp = []

    for b, r in zip(baseline, reflect):
        gt = b["ground_truth"]
        b_pred = b["predicted"]
        r_pred = r["predicted"]

        if b_pred != gt and r_pred == gt:
            if gt == "harmful":
                fn_fixed.append((b, r))
            else:
                fp_fixed.append((b, r))
        elif b_pred == gt and r_pred != gt:
            if gt == "unharmful":
                new_fp.append((b, r))
            # new FN is also possible but less common
        elif b_pred != gt and r_pred != gt:
            if gt == "harmful":
                both_fn.append((b, r))

    return fn_fixed, fp_fixed, new_fp, both_fn


def write_qualitative(fn_fixed, fp_fixed, new_fp, both_fn, out_path):
    lines = []

    def add_section(title, items, max_show=5, show_type="fn_fixed"):
        lines.append("=" * 70)
        lines.append(f"{title}  (total: {len(items)})")
        lines.append("=" * 70)
        if not items:
            lines.append("  (none)")
            lines.append("")
            return

        # Prioritize adversarial examples
        adv = [x for x in items if x[0].get("adversarial")]
        non_adv = [x for x in items if not x[0].get("adversarial")]
        show = (adv[:max_show] if len(adv) >= max_show
                else adv + non_adv[:max_show - len(adv)])

        for i, (b, r) in enumerate(show, 1):
            adv_tag = " [ADVERSARIAL]" if b.get("adversarial") else ""
            subcat = b.get("subcategory", "N/A")
            lines.append(f"\n--- Example {i}{adv_tag} (subcategory: {subcat}) ---")
            lines.append(f"Ground truth: {b['ground_truth']}")
            lines.append(f"Baseline predicted: {b['predicted']}")
            lines.append(f"Reflect-Guard predicted: {r['predicted']}")
            lines.append(f"\nPrompt:\n  {truncate(b['prompt'], 500)}")
            if r.get("reflection"):
                lines.append(f"\nReflection:\n  {truncate(r['reflection'], 500)}")
            lines.append("")
        lines.append("")

    add_section(
        "BASELINE FALSE NEGATIVES FIXED BY REFLECT-GUARD\n"
        "(Baseline missed harmful → Reflect-Guard caught it)",
        fn_fixed, max_show=5
    )

    add_section(
        "BASELINE FALSE POSITIVES FIXED BY REFLECT-GUARD\n"
        "(Baseline falsely flagged → Reflect-Guard correct)",
        fp_fixed, max_show=3
    )

    add_section(
        "NEW FALSE POSITIVES INTRODUCED BY REFLECT-GUARD\n"
        "(Baseline correct → Reflect-Guard falsely flagged)",
        new_fp, max_show=3
    )

    add_section(
        "BOTH MISSED (FALSE NEGATIVES IN BOTH)\n"
        "(Neither baseline nor Reflect-Guard caught it)",
        both_fn, max_show=3
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


# ── Error Analysis ────────────────────────────────────────────────────

def error_analysis(baseline, reflect, out_path):
    lines = []

    # Overall confusion matrix
    for name, data in [("Baseline", baseline), ("Reflect-Guard", reflect)]:
        tp = sum(1 for d in data if d["ground_truth"] == "harmful" and d["predicted"] == "harmful")
        fp = sum(1 for d in data if d["ground_truth"] == "unharmful" and d["predicted"] == "harmful")
        fn = sum(1 for d in data if d["ground_truth"] == "harmful" and d["predicted"] == "unharmful")
        tn = sum(1 for d in data if d["ground_truth"] == "unharmful" and d["predicted"] == "unharmful")
        lines.append(f"{'=' * 50}")
        lines.append(f"{name} Confusion Matrix")
        lines.append(f"{'=' * 50}")
        lines.append(f"               Pred Harmful  Pred Unharmful")
        lines.append(f"  GT Harmful       {tp:>5}          {fn:>5}")
        lines.append(f"  GT Unharmful     {fp:>5}          {tn:>5}")
        lines.append(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        lines.append(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        lines.append("")

    # Adversarial vs non-adversarial breakdown
    lines.append("=" * 50)
    lines.append("ADVERSARIAL vs NON-ADVERSARIAL BREAKDOWN")
    lines.append("=" * 50)
    for adv_val, label in [(True, "Adversarial"), (False, "Non-Adversarial")]:
        subset_b = [d for d in baseline if d.get("adversarial") == adv_val]
        subset_r = [d for d in reflect if d.get("adversarial") == adv_val]
        if not subset_b:
            continue
        lines.append(f"\n--- {label} (n={len(subset_b)}) ---")
        for name, sub in [("Baseline", subset_b), ("Reflect-Guard", subset_r)]:
            tp = sum(1 for d in sub if d["ground_truth"] == "harmful" and d["predicted"] == "harmful")
            fp = sum(1 for d in sub if d["ground_truth"] == "unharmful" and d["predicted"] == "harmful")
            fn = sum(1 for d in sub if d["ground_truth"] == "harmful" and d["predicted"] == "unharmful")
            tn = sum(1 for d in sub if d["ground_truth"] == "unharmful" and d["predicted"] == "unharmful")
            total = len(sub)
            acc = (tp + tn) / total if total > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            lines.append(f"  {name:15s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn} TN={tn})")

    # Subcategory breakdown for errors
    lines.append("")
    lines.append("=" * 50)
    lines.append("REFLECT-GUARD ERROR BREAKDOWN BY SUBCATEGORY")
    lines.append("=" * 50)

    # False positives by subcategory
    reflect_fp = [d for d in reflect if d["ground_truth"] == "unharmful" and d["predicted"] == "harmful"]
    fp_subcats = Counter(d.get("subcategory", "unknown") for d in reflect_fp)
    lines.append(f"\nFalse Positives (n={len(reflect_fp)}) by subcategory:")
    for subcat, count in fp_subcats.most_common():
        lines.append(f"  {subcat:30s}  {count}")

    # False negatives by subcategory
    reflect_fn = [d for d in reflect if d["ground_truth"] == "harmful" and d["predicted"] == "unharmful"]
    fn_subcats = Counter(d.get("subcategory", "unknown") for d in reflect_fn)
    lines.append(f"\nFalse Negatives (n={len(reflect_fn)}) by subcategory:")
    for subcat, count in fn_subcats.most_common():
        lines.append(f"  {subcat:30s}  {count}")

    # JailbreakBench error analysis
    if os.path.exists(JBB_REFLECT_PATH):
        jbb = load_json(JBB_REFLECT_PATH)
        lines.append("")
        lines.append("=" * 50)
        lines.append("JAILBREAKBENCH REFLECT-GUARD MISSES")
        lines.append("=" * 50)
        misses = [d for d in jbb if d["predicted"] != "harmful"]
        lines.append(f"Total misses: {len(misses)} / {len(jbb)}")
        for i, d in enumerate(misses, 1):
            lines.append(f"\n--- Miss {i} (method: {d['method']}) ---")
            lines.append(f"Behavior: {d.get('behavior', 'N/A')}")
            lines.append(f"Prompt:\n  {truncate(d['prompt'], 500)}")
            if d.get("reflection"):
                lines.append(f"Reflection:\n  {truncate(d['reflection'], 500)}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    baseline = load_json(BASELINE_PATH)
    reflect = load_json(REFLECT_PATH)

    assert len(baseline) == len(reflect), "Baseline and reflect results must have same length"

    fn_fixed, fp_fixed, new_fp, both_fn = extract_qualitative(baseline, reflect)

    print(f"Baseline FN fixed by Reflect-Guard: {len(fn_fixed)}")
    print(f"Baseline FP fixed by Reflect-Guard: {len(fp_fixed)}")
    print(f"New FP introduced by Reflect-Guard: {len(new_fp)}")
    print(f"Both missed (FN in both):           {len(both_fn)}")
    print()

    write_qualitative(
        fn_fixed, fp_fixed, new_fp, both_fn,
        os.path.join(RESULTS_DIR, "qualitative_examples.txt")
    )

    error_analysis(baseline, reflect, os.path.join(RESULTS_DIR, "error_analysis.txt"))


if __name__ == "__main__":
    main()
