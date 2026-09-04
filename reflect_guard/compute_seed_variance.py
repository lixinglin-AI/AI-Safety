"""
Aggregate multi-seed training results for Conditions B (SFT-only) and D
(Reflect-Guard) into mean +/- std, a pooled bootstrap CI, and a paired
significance test between B and D — the statistical-rigor items reviewers asked
for on top of the single-run numbers already in the paper.

Expects result files produced by:
  reflect_guard/train_ablations.py --ablation b --seed {N}   (+ evaluate_ablations.py --condition b --seed {N})
  reflect_guard/main.py --train-only --seed {N}               (+ evaluate_ablations.py --condition d --seed {N})
for each seed in --seeds (default: 42 123 2024 — 42 reuses the original,
already-trained/evaluated run under its unsuffixed paths; only 123 and 2024
need actual retraining).

What this reports, and why each piece is a distinct axis of uncertainty:
  - Mean +/- std across seeds: variance from training randomness (weight init,
    data-loader shuffling order) with the test set and training data both held
    fixed. This is what "N random seeds" conventionally means in the reviews.
  - Pooled bootstrap 95% CI: for each seed, bootstrap-resample that seed's fixed
    predictions (test-set sampling uncertainty), then pool all seeds' resamples
    together into one distribution and take its 2.5/97.5 percentiles. This
    combines test-set-size uncertainty AND seed-to-seed variance into one
    interval, unlike the per-run bootstrap CIs already in the paper (which only
    capture the former for a single seed).
  - Paired significance test (D vs B) on the per-seed point estimates: paired
    t-test (scipy.stats.ttest_rel) plus a Wilcoxon signed-rank test as a
    non-parametric check, since N=3-5 seeds is too small for the t-test's
    normality assumption to be trustworthy on its own. Report both, and treat
    p-values from either as indicative rather than definitive at this N — say so
    explicitly rather than overstating power from 3-5 paired samples.

Usage:
    python reflect_guard/compute_seed_variance.py
    python reflect_guard/compute_seed_variance.py --seeds 42 123 2024 777
    python reflect_guard/compute_seed_variance.py --dataset finance
"""

import argparse
import json
import os
import random
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "seed_variance_report.txt")

N_BOOTSTRAP = 1000
RANDOM_SEED_FOR_BOOTSTRAP = 12345  # local analysis RNG seed, unrelated to training seeds


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
    return {"precision": prec, "recall": rec, "f1": f1, "n": len(records)}


def bootstrap_replicates(records, metric_key, n_boot=N_BOOTSTRAP, rng=None):
    rng = rng or random.Random(RANDOM_SEED_FOR_BOOTSTRAP)
    n = len(records)
    vals = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        vals.append(metrics(sample)[metric_key])
    return vals


def mean_std(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def result_path(dataset, condition_name, seed):
    suffix = "" if seed == 42 else f"_seed{seed}"
    if dataset == "wildguard":
        return os.path.join(RESULTS_DIR, f"wildguard_results_{condition_name}{suffix}.json")
    if dataset == "jailbreakbench":
        return os.path.join(RESULTS_DIR, f"jailbreakbench_results_{condition_name}{suffix}.json")
    if dataset == "finance":
        return os.path.join(RESULTS_DIR, f"fin_results_{condition_name}{suffix}.json")
    raise ValueError(dataset)


# Condition names as written by evaluate_ablations.py (see CONDITIONS dict there).
CONDITION_NAMES = {"b": "ablation_b", "d": "reflect_guard"}


def load_seed_records(dataset, cond_key, seeds):
    """Return {seed: records} for whichever seeds have a result file on disk."""
    name = CONDITION_NAMES[cond_key]
    out = {}
    for seed in seeds:
        path = result_path(dataset, name, seed)
        records = load(path)
        if records is None:
            print(f"  [missing] seed={seed}: {os.path.relpath(path, RESULTS_DIR)}", file=sys.stderr)
            continue
        out[seed] = records
    return out


def analyze_dataset(dataset, seeds, metric_key="f1", subset_filter=None):
    lines = []
    lines.append(f"\n{'='*70}\n{dataset.upper()}" + (f" ({subset_filter[0]})" if subset_filter else "") + f"\n{'='*70}")

    per_condition = {}
    for cond_key in ("b", "d"):
        seed_records = load_seed_records(dataset, cond_key, seeds)
        if not seed_records:
            lines.append(f"Condition {cond_key.upper()}: no result files found, skipping.")
            continue

        per_seed_metric = {}
        pooled_bootstrap = []
        rng = random.Random(RANDOM_SEED_FOR_BOOTSTRAP)
        for seed, records in seed_records.items():
            if subset_filter:
                field, value = subset_filter
                records = [r for r in records if r.get(field) is value]
            m = metrics(records)
            per_seed_metric[seed] = m[metric_key]
            pooled_bootstrap.extend(bootstrap_replicates(records, metric_key, rng=rng))

        mean, std = mean_std(list(per_seed_metric.values()))
        pooled_bootstrap_sorted = sorted(x for x in pooled_bootstrap if x == x)
        if pooled_bootstrap_sorted:
            lo = pooled_bootstrap_sorted[int(0.025 * len(pooled_bootstrap_sorted))]
            hi = pooled_bootstrap_sorted[int(0.975 * len(pooled_bootstrap_sorted))]
        else:
            lo, hi = float("nan"), float("nan")

        lines.append(f"\nCondition {cond_key.upper()} ({CONDITION_NAMES[cond_key]}), n_seeds={len(per_seed_metric)}")
        lines.append(f"  per-seed {metric_key}: " +
                      ", ".join(f"seed{s}={v:.4f}" for s, v in sorted(per_seed_metric.items())))
        lines.append(f"  mean +/- std: {mean:.4f} +/- {std:.4f}")
        lines.append(f"  pooled bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]  (test-set + seed variance combined, {len(pooled_bootstrap_sorted)} replicates)")

        per_condition[cond_key] = per_seed_metric

    if "b" in per_condition and "d" in per_condition:
        common_seeds = sorted(set(per_condition["b"]) & set(per_condition["d"]))
        if len(common_seeds) >= 2:
            b_vals = [per_condition["b"][s] for s in common_seeds]
            d_vals = [per_condition["d"][s] for s in common_seeds]
            lines.append(f"\nPaired D vs B significance test on {metric_key} across {len(common_seeds)} common seeds {common_seeds}:")
            try:
                from scipy import stats
                t_stat, t_p = stats.ttest_rel(d_vals, b_vals)
                lines.append(f"  paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
                if len(common_seeds) >= 3:
                    try:
                        w_stat, w_p = stats.wilcoxon(d_vals, b_vals)
                        lines.append(f"  Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")
                    except ValueError as e:
                        lines.append(f"  Wilcoxon signed-rank: not computable ({e})")
                lines.append(f"  NOTE: n={len(common_seeds)} paired seeds is small; treat these p-values as "
                              "indicative, not definitive — see module docstring.")
            except ImportError:
                lines.append("  scipy not installed — skipping significance test (pip install scipy)")
        else:
            lines.append("\nNeed at least 2 common seeds between B and D to run a paired significance test.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed variance for Conditions B and D")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "finance", "all"], default="all")
    parser.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"])
    args = parser.parse_args()

    report_lines = [f"Seed-variance report (seeds={args.seeds}, metric={args.metric})"]

    if args.dataset in ("wildguard", "all"):
        report_lines.append(analyze_dataset("wildguard", args.seeds, args.metric))
        report_lines.append(analyze_dataset("wildguard", args.seeds, args.metric, subset_filter=("adversarial", True)))
    if args.dataset in ("jailbreakbench", "all"):
        report_lines.append(analyze_dataset("jailbreakbench", args.seeds, "recall"))  # DR = recall (all ground-truth harmful)
    if args.dataset in ("finance", "all"):
        report_lines.append(analyze_dataset("finance", args.seeds, args.metric))

    report = "\n".join(report_lines)
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
