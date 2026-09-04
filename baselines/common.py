"""
Shared paths and dataset loaders for the competitive-baseline evaluation scripts
(evaluate_wildguard.py, evaluate_shieldgemma.py, evaluate_promptguard.py,
evaluate_prompted_cot.py).

These scripts evaluate *other* published safety classifiers (not Reflect-Guard or
any of its ablations) on the same three benchmarks used elsewhere in this repo, so
that the paper's comparison table isn't limited to Reflect-Guard vs. its own
un-fine-tuned self. Outputs are written to results/baselines/ with the naming
convention {model}_on_{dataset}.json, kept separate from the existing
results/wildguard_results_*.json files (which are WildGuardTest evaluated by our
own Conditions 0/A/B/C/D, not by external models) to avoid collisions.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root (for utils/)

from utils.jailbreakbench_loader import load_jailbreakbench_artifacts

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "baselines")

FIN_BENCHMARK_PATH = os.path.join(ROOT_DIR, "fin", "fin_benchmark.jsonl")


def load_wildguardtest(hf_token: str):
    """Return a list of {'prompt', 'ground_truth', 'adversarial', 'subcategory'} dicts,
    identical in shape to what reflect_guard/evaluate_*.py consume, so downstream
    analysis code can treat all conditions/models uniformly."""
    from datasets import load_dataset

    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    split = list(ds.keys())[0]
    dataset = ds[split]
    rows = []
    for row in dataset:
        gt = row.get("prompt_harm_label")
        if gt is None:
            continue
        rows.append({
            "prompt": row["prompt"],
            "ground_truth": gt,
            "adversarial": row.get("adversarial"),
            "subcategory": row.get("subcategory"),
        })
    return rows


def load_finance_benchmark():
    """Return the 329-example finance benchmark as a list of dicts (prompt, label
    renamed to ground_truth, category, source, subcategory, adversarial)."""
    if not os.path.exists(FIN_BENCHMARK_PATH):
        print(f"ERROR: finance benchmark not found at {FIN_BENCHMARK_PATH}", file=sys.stderr)
        print("Run fin/build_benchmark.py first, or omit --dataset finance.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(FIN_BENCHMARK_PATH) as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "prompt": row["prompt"],
                "ground_truth": row["label"],
                "category": row["category"],
                "source": row["source"],
                "subcategory": row["subcategory"],
                "adversarial": row.get("adversarial", False),
            })
    return rows


def save_results(results: list[dict], model_name: str, dataset_name: str, suffix: str = ""):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{model_name}_on_{dataset_name}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return out_path


def summarize(results: list[dict]):
    tp = sum(1 for r in results if r["ground_truth"] == "harmful" and r["predicted"] == "harmful")
    fp = sum(1 for r in results if r["ground_truth"] == "unharmful" and r["predicted"] == "harmful")
    fn = sum(1 for r in results if r["ground_truth"] == "harmful" and r["predicted"] == "unharmful")
    tn = sum(1 for r in results if r["ground_truth"] == "unharmful" and r["predicted"] == "unharmful")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    print(f"  n={len(results)}  TP={tp} FP={fp} FN={fn} TN={tn}  "
          f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")


JBB_CACHE_PATH = os.path.join(RESULTS_DIR, "_jbb_prompt_cache.json")


def load_jailbreakbench():
    """Return a list of {'method', 'behavior_id', 'behavior', 'prompt', 'jbb_success',
    'ground_truth': 'harmful'} dicts. All JBB prompts are ground-truth harmful.

    Thin wrapper around utils.jailbreakbench_loader (see that module's docstring for
    why this fetches directly over HTTP instead of importing the `jailbreakbench`
    PyPI package). Cached to results/baselines/_jbb_prompt_cache.json after the
    first fetch.
    """
    if os.path.exists(JBB_CACHE_PATH):
        with open(JBB_CACHE_PATH) as f:
            return json.load(f)

    rows = [{**e, "ground_truth": "harmful"} for e in load_jailbreakbench_artifacts()]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(JBB_CACHE_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Cached {len(rows)} JailbreakBench prompts to {JBB_CACHE_PATH}")
    return rows
