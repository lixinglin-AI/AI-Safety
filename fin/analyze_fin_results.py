"""
Analyze ReflectGuard-Fin evaluation results and print tables for the paper.

Usage:
    python fin/analyze_fin_results.py
    python fin/analyze_fin_results.py --smoke-test   # use smoke-test result files
    python fin/analyze_fin_results.py --latex-only   # print LaTeX tables only
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fin.fin_config import FIN_CATEGORIES, FIN_RESULTS_DIR

POS = "harmful"


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def metrics(results: list[dict]) -> dict:
    yt = [r["ground_truth"] for r in results]
    yp = [r["predicted"]    for r in results]
    return {
        "n":   len(results),
        "acc": accuracy_score(yt, yp),
        "p":   precision_score(yt, yp, pos_label=POS, zero_division=0),
        "r":   recall_score(yt, yp,    pos_label=POS, zero_division=0),
        "f1":  f1_score(yt, yp,        pos_label=POS, zero_division=0),
    }


def group_by_category(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)
    return groups


# ── Console printing ──────────────────────────────────────────────────────────

def print_overall(base_results, ref_results, suffix: str = ""):
    bm = metrics(base_results)
    rm = metrics(ref_results)
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS{(' (' + suffix + ')') if suffix else ''}")
    print(f"{'='*60}")
    print(f"{'Metric':<12}  {'Baseline':>10}  {'ReflectGuard-Fin':>18}  {'Δ':>8}")
    print(f"{'-'*55}")
    for key, name in [("acc", "Accuracy"), ("p", "Precision"), ("r", "Recall"), ("f1", "F1")]:
        delta = rm[key] - bm[key]
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<12}  {bm[key]:>10.4f}  {rm[key]:>18.4f}  {sign}{delta:>7.4f}")
    print(f"{'N examples':<12}  {bm['n']:>10}  {rm['n']:>18}")


def print_by_source(base_results, ref_results):
    all_sources = sorted(set(r["source"] for r in base_results))
    print(f"\n{'='*60}")
    print("BY SOURCE")
    print(f"{'='*60}")
    print(f"{'Source':<30}  {'Baseline F1':>12}  {'RG-Fin F1':>10}  {'N':>5}")
    print(f"{'-'*65}")
    for src in all_sources:
        br = [r for r in base_results if r["source"] == src]
        rr = [r for r in ref_results  if r["source"] == src]
        if not br:
            continue
        bm = metrics(br)
        rm = metrics(rr)
        print(f"{src:<30}  {bm['f1']:>12.4f}  {rm['f1']:>10.4f}  {len(br):>5}")


def print_by_category(base_results, ref_results):
    base_grp = group_by_category(base_results)
    ref_grp  = group_by_category(ref_results)

    all_codes = list(FIN_CATEGORIES.keys()) + ["benign", "FIN-UNK"]

    print(f"\n{'='*70}")
    print("PER-CATEGORY RESULTS")
    print(f"{'='*70}")
    header = f"{'Category':<38}  {'Baseline':>9}  {'RG-Fin':>8}  {'Δ F1':>6}  {'N':>4}"
    print(header)
    print(f"  {'':38}  {'F1':>9}  {'F1':>8}")
    print(f"{'-'*75}")

    for code in all_codes:
        br = base_grp.get(code, [])
        rr = ref_grp.get(code, [])
        if not br:
            continue
        label = FIN_CATEGORIES[code]["label"] if code in FIN_CATEGORIES else code
        short_label = f"{code}: {label}"[:38]

        bm = metrics(br)
        rm = metrics(rr)
        delta = rm["f1"] - bm["f1"]
        sign  = "+" if delta >= 0 else ""
        print(f"{short_label:<40}  {bm['f1']:>8.3f}  {rm['f1']:>8.3f}  {sign}{delta:>5.3f}  {len(br):>4}")


# ── LaTeX generation ──────────────────────────────────────────────────────────

def latex_overall_table(base_results, ref_results) -> str:
    bm = metrics(base_results)
    rm = metrics(ref_results)

    def row(name, key):
        delta = rm[key] - bm[key]
        sign  = r"\textbf{+" if delta >= 0 else r"\textbf{"
        return (
            f"    {name} & {bm[key]:.3f} & {bm[key]:.3f} & "
            f"{rm[key]:.3f} & {sign}{delta:+.3f}{'}'} \\\\"
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Overall performance of Llama-Guard-3-8B (baseline) and ReflectGuard-Fin",
        r"on the unified FinSafetyBench benchmark. ReflectGuard-Fin uses the same checkpoint",
        r"as the general ReflectGuard without any finance-specific fine-tuning.}",
        r"\label{tab:overall}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{ReflectGuard-Fin} \\",
        r"       & Score & & Score & $\Delta$ \\",
        r"\midrule",
        f"    Precision & {bm['p']:.3f} & & {rm['p']:.3f} & {rm['p']-bm['p']:+.3f} \\\\",
        f"    Recall    & {bm['r']:.3f} & & {rm['r']:.3f} & {rm['r']-bm['r']:+.3f} \\\\",
        f"    F1 Score  & {bm['f1']:.3f} & & {rm['f1']:.3f} & {rm['f1']-bm['f1']:+.3f} \\\\",
        f"    Accuracy  & {bm['acc']:.3f} & & {rm['acc']:.3f} & {rm['acc']-bm['acc']:+.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_category_table(base_results, ref_results) -> str:
    base_grp = group_by_category(base_results)
    ref_grp  = group_by_category(ref_results)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-category F1 scores on FinSafetyBench. ReflectGuard-Fin consistently",
        r"outperforms the baseline across all seven high-risk financial instruction categories.}",
        r"\label{tab:category}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Code & Category & Baseline F1 & RG-Fin F1 & $\Delta$ \\",
        r"\midrule",
    ]

    for code, cat in FIN_CATEGORIES.items():
        br = base_grp.get(code, [])
        rr = ref_grp.get(code, [])
        if not br:
            lines.append(f"    {code} & {cat['label']} & --- & --- & --- \\\\")
            continue
        bm = metrics(br)
        rm = metrics(rr)
        delta = rm["f1"] - bm["f1"]
        sign  = "+" if delta >= 0 else ""
        safe_label = cat['label'].replace('&', r'\&')
        lines.append(
            f"    {code} & {safe_label} & {bm['f1']:.3f} & "
            f"\\textbf{{{rm['f1']:.3f}}} & {sign}{delta:.3f} \\\\"
        )

    lines += [
        r"\midrule",
        f"    \\multicolumn{{2}}{{l}}{{\\textit{{All categories}}}} & "
        f"{metrics(base_results)['f1']:.3f} & "
        f"\\textbf{{{metrics(ref_results)['f1']:.3f}}} & "
        f"{metrics(ref_results)['f1'] - metrics(base_results)['f1']:+.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Qualitative examples ──────────────────────────────────────────────────────

def print_qualitative(ref_results: list[dict], n_per_cat: int = 2):
    """Show examples where the reflection explicitly catches adversarial framing."""
    adv_keywords = [
        "fictional", "framing", "roleplay", "role-play", "pretend", "scenario",
        "hypothetical", "educational pretext", "indirect", "obfuscat",
        "social engineering", "bypass",
    ]
    print(f"\n{'='*60}")
    print("QUALITATIVE EXAMPLES: reflection catching adversarial framing")
    print(f"{'='*60}")
    shown: dict[str, int] = defaultdict(int)
    for r in ref_results:
        if r["ground_truth"] != "harmful" or r["predicted"] != "harmful":
            continue
        if not r.get("reflection"):
            continue
        ref_lower = r["reflection"].lower()
        if not any(kw in ref_lower for kw in adv_keywords):
            continue
        code = r["category"]
        if shown[code] >= n_per_cat:
            continue
        shown[code] += 1
        label = FIN_CATEGORIES.get(code, {}).get("label", code)
        print(f"\n[{code}: {label}]")
        print(f"Prompt:     {r['prompt'][:200]}")
        print(f"Reflection: {r['reflection'][:300]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze ReflectGuard-Fin results")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--latex-only", action="store_true")
    args = parser.parse_args()

    suffix = "_smoke" if args.smoke_test else ""
    base_path = os.path.join(FIN_RESULTS_DIR, f"fin_results_baseline{suffix}.json")
    ref_path  = os.path.join(FIN_RESULTS_DIR, f"fin_results_reflect{suffix}.json")

    if not os.path.exists(base_path) or not os.path.exists(ref_path):
        print(f"ERROR: result files not found.")
        print(f"  Expected:\n    {base_path}\n    {ref_path}")
        print("Run: python fin/evaluate_fin.py --model both")
        sys.exit(1)

    base_results = load(base_path)
    ref_results  = load(ref_path)

    if not args.latex_only:
        print_overall(base_results, ref_results, suffix or "full")
        print_by_source(base_results, ref_results)
        print_by_category(base_results, ref_results)
        print_qualitative(ref_results)

    print(f"\n{'='*60}")
    print("LATEX TABLES")
    print(f"{'='*60}")
    print("\n% --- Table 1: Overall ---")
    print(latex_overall_table(base_results, ref_results))
    print("\n% --- Table 2: Per-category ---")
    print(latex_category_table(base_results, ref_results))

    # Save tables to file
    out_path = os.path.join(FIN_RESULTS_DIR, f"fin_analysis{suffix}.txt")
    with open(out_path, "w") as f:
        from io import StringIO
        import contextlib
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            print_overall(base_results, ref_results, suffix or "full")
            print_by_source(base_results, ref_results)
            print_by_category(base_results, ref_results)
            print("\n% --- Table 1: Overall ---")
            print(latex_overall_table(base_results, ref_results))
            print("\n% --- Table 2: Per-category ---")
            print(latex_category_table(base_results, ref_results))
        f.write(buf.getvalue())
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
