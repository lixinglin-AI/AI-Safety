"""
Analyze and report baseline evaluation results.

Reads:
  results/wildguard_results.json
  results/jailbreakbench_results.json

Prints a formatted report and saves it to results/summary_report.txt.
"""

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

from config import RESULTS_DIR

WILDGUARD_PATH = os.path.join(RESULTS_DIR, "wildguard_results.json")
JBB_PATH = os.path.join(RESULTS_DIR, "jailbreakbench_results.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "summary_report.txt")


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


def main():
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
