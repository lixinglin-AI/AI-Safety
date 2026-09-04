"""
Merge our own Conditions 0/B/D with the competitive-baseline results (WildGuard,
ShieldGemma, PromptGuard, prompted-CoT) into one comparison report, plus a
ready-to-paste LaTeX table snippet for reflect_guard_mdpi-2.tex.

Run this only after the relevant evaluate_*.py scripts have produced their JSON
outputs in results/baselines/. Missing files are skipped with a warning rather than
failing the whole report, so you can run this incrementally as results land.

Usage:
    python baselines/aggregate_comparison.py
    python baselines/aggregate_comparison.py --dataset jailbreakbench
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")
REPORT_PATH = os.path.join(RESULTS_DIR, "competitive_baselines_report.txt")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def metrics(records):
    tp = sum(1 for r in records if r["ground_truth"] == "harmful" and r["predicted"] == "harmful")
    fp = sum(1 for r in records if r["ground_truth"] == "unharmful" and r["predicted"] == "harmful")
    fn = sum(1 for r in records if r["ground_truth"] == "harmful" and r["predicted"] == "unharmful")
    tn = sum(1 for r in records if r["ground_truth"] == "unharmful" and r["predicted"] == "unharmful")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
    acc = (tp + tn) / len(records) if records else float("nan")
    return dict(n=len(records), tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, acc=acc)


# model_key -> (display name, results/ path relative pattern for WGT, JBB)
OUR_CONDITIONS = {
    "0": ("Llama-Guard-3-8B (clean baseline)", "wildguard_results_clean_baseline.json", "jailbreakbench_results_clean_baseline.json"),
    "B": ("Reflect-Guard, SFT-only (Condition B)", "wildguard_results_ablation_b.json", "jailbreakbench_results_ablation_b.json"),
    "D": ("Reflect-Guard (full, Condition D)", "wildguard_results_reflect_guard.json", "jailbreakbench_results_reflect_guard.json"),
}

BASELINE_MODELS = {
    "wildguard": "WildGuard",
    "shieldgemma": "ShieldGemma-9b",
    "promptguard": "Llama-Prompt-Guard-2-86M",
    "promptguard_v1": "Prompt-Guard-86M (v1, reference only)",
    "promptedcot": "Prompted CoT (GPT-4o-mini, zero-shot)",
    "openai_moderation": "OpenAI Moderation API",
}


def wgt_row(name, records):
    m = metrics(records)
    adv = [r for r in records if r.get("adversarial") is True]
    m_adv = metrics(adv) if adv else None
    return name, m, m_adv


def jbb_row(name, records):
    m = metrics(records)
    return name, m


def fmt(x):
    return f"{x:.3f}" if x == x else "  --"  # NaN check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "finance", "all"], default="all")
    args = parser.parse_args()

    lines = []
    lines.append("=" * 78)
    lines.append("COMPETITIVE BASELINE COMPARISON")
    lines.append("=" * 78)

    targets = ["wildguard", "jailbreakbench", "finance"] if args.dataset == "all" else [args.dataset]

    if "wildguard" in targets:
        lines.append("\n--- WildGuardTest ---")
        lines.append(f"{'Model':45s} {'n':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}   {'AdvRec':>6}")
        for key, (name, wgt_path, _) in OUR_CONDITIONS.items():
            records = load(os.path.join(RESULTS_DIR, wgt_path))
            if records is None:
                lines.append(f"{name:45s}  MISSING ({wgt_path})")
                continue
            _, m, m_adv = wgt_row(name, records)
            adv_rec = fmt(m_adv["recall"]) if m_adv else "  --"
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['acc']):>6} {fmt(m['precision']):>6} "
                         f"{fmt(m['recall']):>6} {fmt(m['f1']):>6}   {adv_rec:>6}")
        for key, name in BASELINE_MODELS.items():
            path = os.path.join(BASELINES_DIR, f"{key}_on_wildguardtest.json")
            records = load(path)
            if records is None:
                lines.append(f"{name:45s}  NOT YET RUN ({os.path.relpath(path, ROOT_DIR)})")
                continue
            _, m, m_adv = wgt_row(name, records)
            adv_rec = fmt(m_adv["recall"]) if m_adv else "  --"
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['acc']):>6} {fmt(m['precision']):>6} "
                         f"{fmt(m['recall']):>6} {fmt(m['f1']):>6}   {adv_rec:>6}")

    if "jailbreakbench" in targets:
        lines.append("\n--- JailbreakBench (all ground-truth harmful; report is detection rate = recall) ---")
        lines.append(f"{'Model':45s} {'n':>6} {'DR (recall)':>12} {'GBR=1-DR':>10}")
        for key, (name, _, jbb_path) in OUR_CONDITIONS.items():
            records = load(os.path.join(RESULTS_DIR, jbb_path))
            if records is None:
                lines.append(f"{name:45s}  MISSING ({jbb_path})")
                continue
            _, m = jbb_row(name, records)
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['recall']):>12} {fmt(1 - m['recall']) if m['recall'] == m['recall'] else '  --':>10}")
        for key, name in BASELINE_MODELS.items():
            path = os.path.join(BASELINES_DIR, f"{key}_on_jailbreakbench.json")
            records = load(path)
            if records is None:
                lines.append(f"{name:45s}  NOT YET RUN ({os.path.relpath(path, ROOT_DIR)})")
                continue
            _, m = jbb_row(name, records)
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['recall']):>12} {fmt(1 - m['recall']) if m['recall'] == m['recall'] else '  --':>10}")

    if "finance" in targets:
        lines.append("\n--- Finance benchmark (329 examples) ---")
        lines.append(f"{'Model':45s} {'n':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        for key, (name, _, _) in OUR_CONDITIONS.items():
            fname = {"0": "fin_results_baseline.json", "B": None, "D": "fin_results_reflect.json"}[key]
            if fname is None:
                continue
            records = load(os.path.join(RESULTS_DIR, fname))
            if records is None:
                lines.append(f"{name:45s}  MISSING ({fname})")
                continue
            m = metrics(records)
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['acc']):>6} {fmt(m['precision']):>6} {fmt(m['recall']):>6} {fmt(m['f1']):>6}")
        for key, name in BASELINE_MODELS.items():
            path = os.path.join(BASELINES_DIR, f"{key}_on_finance.json")
            records = load(path)
            if records is None:
                lines.append(f"{name:45s}  NOT YET RUN ({os.path.relpath(path, ROOT_DIR)})")
                continue
            m = metrics(records)
            lines.append(f"{name:45s} {m['n']:6d} {fmt(m['acc']):>6} {fmt(m['precision']):>6} {fmt(m['recall']):>6} {fmt(m['f1']):>6}")

    lines.append("\nNOTE: PromptGuard is not a general content-harm classifier (see "
                 "baselines/evaluate_promptguard.py docstring) — its non-adversarial-subset "
                 "and finance numbers are expected to be weak by design, not comparably worse.")
    lines.append("NOTE: ShieldGemma verdicts are a union over 4 per-policy scores, an adaptation "
                 "to a single binary verdict it was not originally evaluated on (see "
                 "baselines/evaluate_shieldgemma.py docstring).")

    report = "\n".join(lines)
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
