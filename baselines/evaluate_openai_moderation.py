"""
Evaluate OpenAI's Moderation API as a competitive baseline on WildGuardTest,
JailbreakBench, and (optionally) the finance benchmark.

This addresses the reviewer request to compare against "OpenAI moderation models"
in addition to WildGuard, PromptGuard, and the prompted-CoT baseline already in
this directory. Unlike those three, OpenAI's moderation classifier is a closed,
API-only system with its own fixed taxonomy (sexual, hate, harassment, self-harm,
violence, and their sub-categories) that does not map onto the Llama Guard S1--S13
categories used elsewhere in this repo -- we do not attempt a category-level
mapping here, only the binary safe/unsafe verdict the endpoint returns directly via
`flagged`, matching the same ground_truth/predicted schema as every other baseline
in baselines/common.py.

Unlike evaluate_prompted_cot.py, this endpoint is free (no per-call token billing)
and returns a structured `flagged` boolean directly, so there is no verdict-parsing
step and no cost-control flag is strictly necessary -- --limit and --smoke-test are
kept anyway for consistency with the other scripts and for fast iteration/debugging.

Usage:
    export OPENAI_API_KEY=sk-...
    python baselines/evaluate_openai_moderation.py --dataset all --smoke-test
    python baselines/evaluate_openai_moderation.py --dataset all
    python baselines/evaluate_openai_moderation.py --dataset wildguard --model omni-moderation-latest
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common import (
    load_finance_benchmark,
    load_jailbreakbench,
    load_wildguardtest,
    save_results,
    summarize,
)

MODEL_NAME = "openai_moderation"
DEFAULT_MODEL = "omni-moderation-latest"


def classify_one(client, model: str, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.moderations.create(model=model, input=prompt[:4000])
            result = resp.results[0]
            predicted = "harmful" if result.flagged else "unharmful"
            categories = {
                k: v for k, v in result.categories.model_dump().items() if v
            }
            return {
                "predicted": predicted,
                "flagged_categories": categories,
                "raw_output": json.dumps(result.category_scores.model_dump()),
            }
        except Exception as e:
            if attempt == retries - 1:
                return {"predicted": "unharmful", "flagged_categories": {}, "raw_output": f"[error: {e}]"}
            time.sleep(2 ** attempt)


def evaluate(client, model: str, rows: list[dict], dataset_name: str,
             concurrency: int, smoke_test: bool = False, limit: int | None = None):
    if smoke_test:
        rows = rows[:10]
        print(f"Smoke-test mode: {len(rows)} examples.")
    elif limit:
        rows = rows[:limit]
        print(f"Limited to first {len(rows)} examples.")

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(classify_one, client, model, row["prompt"]): i
            for i, row in enumerate(rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"OpenAI Moderation on {dataset_name}"):
            i = futures[future]
            results[i] = {**rows[i], **future.result()}

    summarize(results)
    suffix = "_smoke" if smoke_test else ""
    save_results(results, MODEL_NAME, dataset_name, suffix)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenAI's Moderation API as a competitive baseline")
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "finance", "all"],
                         default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help="Any OpenAI moderation model id (e.g. omni-moderation-latest, text-moderation-latest).")
    parser.add_argument("--openai-key", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate only the first N examples per dataset (fast-iteration control).")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--hf-token", default=None,
                         help="Needed only for --dataset wildguard/all (WildGuardMix is gated).")
    args = parser.parse_args()

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OpenAI API key required (--openai-key or OPENAI_API_KEY env var).", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    targets = ["wildguard", "jailbreakbench", "finance"] if args.dataset == "all" else [args.dataset]

    if "wildguard" in targets:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: HF token required for WildGuardTest (gated dataset).", file=sys.stderr)
            sys.exit(1)
        print("\n=== WildGuardTest ===")
        rows = load_wildguardtest(hf_token)
        evaluate(client, args.model, rows, "wildguardtest", args.concurrency, args.smoke_test, args.limit)

    if "jailbreakbench" in targets:
        print("\n=== JailbreakBench ===")
        rows = load_jailbreakbench()
        evaluate(client, args.model, rows, "jailbreakbench", args.concurrency, args.smoke_test, args.limit)

    if "finance" in targets:
        print("\n=== Finance benchmark ===")
        rows = load_finance_benchmark()
        evaluate(client, args.model, rows, "finance", args.concurrency, args.smoke_test, args.limit)


if __name__ == "__main__":
    main()
