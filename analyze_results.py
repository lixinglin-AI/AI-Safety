"""
Analyze and report evaluation results.

Single-model mode (default):
  Reads:  results/wildguard_results.json
          results/jailbreakbench_results.json
  Output: results/summary_report.txt

Comparison mode (--compare):
  Reads:  results/wildguard_results_baseline.json
          results/wildguard_results_reflect.json
          results/jailbreakbench_results_baseline.json
          results/jailbreakbench_results_reflect.json
  Output: results/comparison_report.txt

Usage:
  python analyze_results.py              # single-model report
  python analyze_results.py --compare    # baseline vs Reflect-Guard comparison
"""

import argparse
import json
import os
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

WILDGUARD_PATH = os.path.join(RESULTS_DIR, "wildguard_results.json")
JBB_PATH = os.path.join(RESULTS_DIR, "jailbreakbench_results.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "summary_report.txt")

# Comparison mode paths
WILDGUARD_BASELINE_PATH = os.path.join(RESULTS_DIR, "wildguard_results_baseline.json")
WILDGUARD_REFLECT_PATH  = os.path.join(RESULTS_DIR, "wildguard_results_reflect.json")
JBB_BASELINE_PATH       = os.path.join(RESULTS_DIR, "jailbreakbench_results_baseline.json")
JBB_REFLECT_PATH        = os.path.join(RESULTS_DIR, "jailbreakbench_results_reflect.json")
COMPARISON_REPORT_PATH  = os.path.join(RESULTS_DIR, "comparison_report.txt")


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def fmt_pct(num: int, denom: int) -> str:
    return f"{num}/{denom} ({100 * num / denom:.1f}%)" if denom else "N/A"


def analyze_wildguard(results: list[dict]) -> list[str]:
    lines = []
    lines.append("=" * 60)
    lines.append("WILDGUARD EVALUATION RESULTS")
    lines.append("=" * 60)

    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted"] for r in results]

    lines.append(f"\nTotal examples evaluated: {len(results)}")
    lines.append(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    lines.append(f"Precision: {precision_score(y_true, y_pred, pos_label='harmful'):.4f}")
    lines.append(f"Recall:    {recall_score(y_true, y_pred, pos_label='harmful'):.4f}")
    lines.append(f"F1 Score:  {f1_score(y_true, y_pred, pos_label='harmful'):.4f}")

    lines.append("\nClassification Report:")
    lines.append(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["harmful", "unharmful"])
    lines.append("Confusion Matrix (rows=true, cols=predicted):")
    lines.append(f"                 harmful  unharmful")
    lines.append(f"  harmful        {cm[0][0]:7d}  {cm[0][1]:9d}")
    lines.append(f"  unharmful      {cm[1][0]:7d}  {cm[1][1]:9d}")

    # Breakdown by adversarial flag
    lines.append("\n--- Breakdown by Adversarial Flag ---")
    for flag in [True, False, None]:
        subset = [r for r in results if r.get("adversarial") == flag]
        if not subset:
            continue
        yt = [r["ground_truth"] for r in subset]
        yp = [r["predicted"] for r in subset]
        f1 = f1_score(yt, yp, pos_label="harmful", zero_division=0)
        acc = accuracy_score(yt, yp)
        lines.append(f"  adversarial={flag}: n={len(subset)}, accuracy={acc:.3f}, F1={f1:.3f}")

    # Breakdown by subcategory (worst 5 by F1)
    lines.append("\n--- Worst 5 Subcategories (by F1) ---")
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        cat = r.get("subcategory") or "unknown"
        by_cat[cat].append(r)

    cat_f1 = []
    for cat, items in by_cat.items():
        yt = [r["ground_truth"] for r in items]
        yp = [r["predicted"] for r in items]
        f1 = f1_score(yt, yp, pos_label="harmful", zero_division=0)
        cat_f1.append((cat, len(items), f1))

    cat_f1.sort(key=lambda x: x[2])
    for cat, n, f1 in cat_f1[:5]:
        lines.append(f"  {cat}: n={n}, F1={f1:.3f}")

    return lines


def analyze_jailbreakbench(results: list[dict]) -> list[str]:
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("JAILBREAKBENCH EVALUATION RESULTS")
    lines.append("=" * 60)

    total = len(results)
    detected = sum(1 for r in results if r["predicted"] == "harmful")
    lines.append(f"\nTotal jailbreak prompts: {total}")
    lines.append(f"Detected as harmful:     {fmt_pct(detected, total)}")
    lines.append(f"Missed (bypassed guard): {fmt_pct(total - detected, total)}")

    # Breakdown by method
    lines.append("\n--- Detection Rate by Attack Method ---")
    by_method: dict[str, list] = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    for method, items in sorted(by_method.items()):
        det = sum(1 for r in items if r["predicted"] == "harmful")
        lines.append(f"  {method}: {fmt_pct(det, len(items))}")

    # Most common violated categories
    lines.append("\n--- Violated Categories (when detected) ---")
    cat_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r["predicted"] == "harmful":
            for cat in r.get("violated_categories", []):
                cat_counts[cat] += 1

    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {count}")

    return lines


def _metrics_dict(results: list[dict]) -> dict:
    """Return a flat dict of key metrics for a result set."""
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted"] for r in results]
    return {
        "n": len(results),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="harmful", zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label="harmful", zero_division=0),
        "f1":        f1_score(y_true, y_pred, pos_label="harmful", zero_division=0),
    }


def compare_wildguard(baseline: list[dict], reflect: list[dict]) -> list[str]:
    lines = []
    lines.append("=" * 70)
    lines.append("WILDGUARD COMPARISON: Baseline vs. Reflect-Guard")
    lines.append("=" * 70)

    bm = _metrics_dict(baseline)
    rm = _metrics_dict(reflect)

    header = f"{'Metric':<14} {'Baseline':>10} {'Reflect-Guard':>15} {'Delta':>10}"
    lines.append(header)
    lines.append("-" * 54)
    for key in ("accuracy", "precision", "recall", "f1"):
        delta = rm[key] - bm[key]
        sign = "+" if delta >= 0 else ""
        lines.append(f"{key:<14} {bm[key]:>10.4f} {rm[key]:>15.4f} {sign}{delta:>9.4f}")

    # Adversarial subset comparison
    lines.append("\n--- Adversarial Subset (adversarial=True) ---")
    for label, data in [("Baseline", baseline), ("Reflect-Guard", reflect)]:
        sub = [r for r in data if r.get("adversarial") is True]
        if sub:
            m = _metrics_dict(sub)
            lines.append(
                f"  {label}: n={m['n']}, acc={m['accuracy']:.3f}, "
                f"recall={m['recall']:.3f}, F1={m['f1']:.3f}"
            )

    # Per-attack-type breakdown (if method field present)
    by_method_b: dict[str, list] = defaultdict(list)
    by_method_r: dict[str, list] = defaultdict(list)
    for r in baseline:
        if r.get("method"):
            by_method_b[r["method"]].append(r)
    for r in reflect:
        if r.get("method"):
            by_method_r[r["method"]].append(r)

    if by_method_b:
        lines.append("\n--- Per-method Detection Rate ---")
        all_methods = sorted(set(by_method_b) | set(by_method_r))
        lines.append(f"  {'Method':<25} {'Baseline':>10} {'Reflect':>10}")
        for method in all_methods:
            b_sub = by_method_b.get(method, [])
            r_sub = by_method_r.get(method, [])
            b_det = sum(1 for x in b_sub if x["predicted"] == "harmful")
            r_det = sum(1 for x in r_sub if x["predicted"] == "harmful")
            b_rate = f"{b_det}/{len(b_sub)} ({100*b_det/len(b_sub):.0f}%)" if b_sub else "N/A"
            r_rate = f"{r_det}/{len(r_sub)} ({100*r_det/len(r_sub):.0f}%)" if r_sub else "N/A"
            lines.append(f"  {method:<25} {b_rate:>10} {r_rate:>10}")

    return lines


def run_comparison(report_path: str) -> None:
    lines = []

    missing = []
    for path in (WILDGUARD_BASELINE_PATH, WILDGUARD_REFLECT_PATH):
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print(f"Missing files for comparison: {missing}")
        print("Run evaluate_wildguard.py twice, saving outputs as "
              "wildguard_results_baseline.json and wildguard_results_reflect.json")
        return

    baseline_wg = load_json(WILDGUARD_BASELINE_PATH)
    reflect_wg  = load_json(WILDGUARD_REFLECT_PATH)
    lines.extend(compare_wildguard(baseline_wg, reflect_wg))

    # JBB comparison (optional)
    if os.path.exists(JBB_BASELINE_PATH) and os.path.exists(JBB_REFLECT_PATH):
        b_jbb = load_json(JBB_BASELINE_PATH)
        r_jbb = load_json(JBB_REFLECT_PATH)
        b_det = sum(1 for x in b_jbb if x["predicted"] == "harmful")
        r_det = sum(1 for x in r_jbb if x["predicted"] == "harmful")
        lines.append("\n" + "=" * 70)
        lines.append("JAILBREAKBENCH COMPARISON")
        lines.append("=" * 70)
        lines.append(
            f"Baseline      detected: {fmt_pct(b_det, len(b_jbb))}"
        )
        lines.append(
            f"Reflect-Guard detected: {fmt_pct(r_det, len(r_jbb))}"
        )
        lines.append(f"ASR reduction: {100*(b_jbb.__len__()-b_det)/len(b_jbb):.1f}% → "
                     f"{100*(len(r_jbb)-r_det)/len(r_jbb):.1f}%  "
                     f"(lower is better for attacker)")

        # Per-method
        by_method_b: dict[str, list] = defaultdict(list)
        by_method_r: dict[str, list] = defaultdict(list)
        for x in b_jbb:
            by_method_b[x["method"]].append(x)
        for x in r_jbb:
            by_method_r[x["method"]].append(x)
        lines.append(f"\n  {'Method':<25} {'Baseline DR':>12} {'Reflect DR':>12}")
        for method in sorted(set(by_method_b) | set(by_method_r)):
            b_s = by_method_b.get(method, [])
            r_s = by_method_r.get(method, [])
            b_d = sum(1 for x in b_s if x["predicted"] == "harmful")
            r_d = sum(1 for x in r_s if x["predicted"] == "harmful")
            lines.append(
                f"  {method:<25} {fmt_pct(b_d, len(b_s)):>12} {fmt_pct(r_d, len(r_s)):>12}"
            )

    report = "\n".join(lines)
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nComparison report saved to {report_path}")


ABLATION_CONDITIONS = [
    ("0. Clean Baseline",     "clean_baseline"),
    ("A. Prompted Reflection","ablation_a"),
    ("B. SFT Labels Only",   "ablation_b"),
    ("C. Blind Reflections",  "ablation_c"),
    ("D. Full Reflect-Guard", "reflect_guard"),
]

ABLATION_REPORT_PATH = os.path.join(RESULTS_DIR, "ablation_report.txt")


def run_ablation(report_path: str) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("ABLATION STUDY: Isolating Contributions of Reflection and SFT")
    lines.append("=" * 90)

    # ── WildGuardTest table ──
    lines.append("\n--- WildGuardTest Results ---")
    header = f"{'Condition':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Adv-Rec':>8} {'Adv-F1':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for label, suffix in ABLATION_CONDITIONS:
        wg_path = os.path.join(RESULTS_DIR, f"wildguard_results_{suffix}.json")
        if not os.path.exists(wg_path):
            lines.append(f"{label:<28} {'(missing)':>7}")
            continue
        data = load_json(wg_path)
        m = _metrics_dict(data)
        # Adversarial subset
        adv = [r for r in data if r.get("adversarial") is True]
        if adv:
            am = _metrics_dict(adv)
            adv_rec = f"{am['recall']:.3f}"
            adv_f1 = f"{am['f1']:.3f}"
        else:
            adv_rec = "N/A"
            adv_f1 = "N/A"
        lines.append(
            f"{label:<28} {m['accuracy']:>7.3f} {m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>7.3f} {adv_rec:>8} {adv_f1:>8}"
        )

    # ── JailbreakBench table ──
    lines.append("\n--- JailbreakBench Detection Rate ---")
    jbb_header = f"{'Condition':<28} {'Detected':>10} {'Total':>7} {'DR':>8}"
    lines.append(jbb_header)
    lines.append("-" * len(jbb_header))

    for label, suffix in ABLATION_CONDITIONS:
        jbb_path = os.path.join(RESULTS_DIR, f"jailbreakbench_results_{suffix}.json")
        if not os.path.exists(jbb_path):
            lines.append(f"{label:<28} {'(missing)':>10}")
            continue
        data = load_json(jbb_path)
        detected = sum(1 for r in data if r["predicted"] == "harmful")
        total = len(data)
        dr = detected / total if total else 0
        lines.append(f"{label:<28} {detected:>10} {total:>7} {dr:>8.1%}")

    # ── Delta analysis ──
    lines.append("\n--- Ablation Delta Analysis ---")
    # Try to compute deltas relative to clean baseline
    wg_base_path = os.path.join(RESULTS_DIR, "wildguard_results_clean_baseline.json")
    wg_full_path = os.path.join(RESULTS_DIR, "wildguard_results_reflect_guard.json")
    if os.path.exists(wg_base_path) and os.path.exists(wg_full_path):
        base_m = _metrics_dict(load_json(wg_base_path))
        full_m = _metrics_dict(load_json(wg_full_path))
        total_gain = full_m["f1"] - base_m["f1"]
        lines.append(f"Total F1 gain (D vs 0):    {total_gain:+.4f}")

        wg_b_path = os.path.join(RESULTS_DIR, "wildguard_results_ablation_b.json")
        if os.path.exists(wg_b_path):
            b_m = _metrics_dict(load_json(wg_b_path))
            sft_gain = b_m["f1"] - base_m["f1"]
            reflect_gain = full_m["f1"] - b_m["f1"]
            lines.append(f"SFT-only gain (B vs 0):    {sft_gain:+.4f}  ({100*sft_gain/total_gain:.0f}% of total)" if total_gain > 0 else f"SFT-only gain (B vs 0):    {sft_gain:+.4f}")
            lines.append(f"Reflection gain (D vs B):  {reflect_gain:+.4f}  ({100*reflect_gain/total_gain:.0f}% of total)" if total_gain > 0 else f"Reflection gain (D vs B):  {reflect_gain:+.4f}")

        wg_c_path = os.path.join(RESULTS_DIR, "wildguard_results_ablation_c.json")
        if os.path.exists(wg_c_path):
            c_m = _metrics_dict(load_json(wg_c_path))
            gt_gain = full_m["f1"] - c_m["f1"]
            lines.append(f"GT label gain (D vs C):    {gt_gain:+.4f}")

    report = "\n".join(lines)
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nAblation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs Reflect-Guard results.")
    parser.add_argument("--ablation", action="store_true",
                        help="Generate ablation study comparison report.")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(ABLATION_REPORT_PATH)
        return

    if args.compare:
        run_comparison(COMPARISON_REPORT_PATH)
        return

    report_lines = []

    if os.path.exists(WILDGUARD_PATH):
        wg_results = load_json(WILDGUARD_PATH)
        report_lines.extend(analyze_wildguard(wg_results))
    else:
        print(f"Warning: {WILDGUARD_PATH} not found. Run evaluate_wildguard.py first.")

    if os.path.exists(JBB_PATH):
        jbb_results = load_json(JBB_PATH)
        report_lines.extend(analyze_jailbreakbench(jbb_results))
    else:
        print(f"Warning: {JBB_PATH} not found. Run evaluate_jailbreakbench.py first.")

    report = "\n".join(report_lines)
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
