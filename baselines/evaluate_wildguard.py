"""
Evaluate AllenAI's WildGuard (allenai/wildguard) as a competitive baseline on
WildGuardTest, JailbreakBench, and (optionally) the finance benchmark.

WildGuard is the most important baseline to include: our training data (500 of the
1,000 training examples) and our main evaluation benchmark (WildGuardTest) both come
from WildGuardMix, which WildGuard was purpose-built and trained on. Any claim that
Reflect-Guard improves adversarial-prompt detection needs to be checked against the
classifier that was already trained for exactly this distribution.

We use AllenAI's official `wildguard` PyPI package rather than hand-rolling the
prompt template, since it owns the exact instruction format and output parser
(including its own malformed-output detection via `is_parsing_error`).

Setup (do this once, in a fresh/isolated venv or conda env — NOT the main
`reflect_guard` env — since `wildguard` hard-requires vllm>=0.4.2 and torch>=2.3.0,
which will upgrade past the pinned torch==2.2.0/transformers==4.40.0 the rest of this
repo's training/eval pipeline (peft/bitsandbytes) was validated against):
    pip install wildguard
(the JailbreakBench prompts are fetched directly over HTTP by
utils/jailbreakbench_loader.py, not the `jailbreakbench` package — do NOT
additionally `pip install jailbreakbench` here, it pulls in litellm and breaks
vllm's transformers/protobuf pins; see that module's docstring)

NOTE on --no-vllm below: `wildguard/wildguard.py` does an unconditional top-level
`from vllm import LLM`, so vllm must be installed regardless of which backend you
use at runtime — `import wildguard` itself fails without it. --no-vllm only changes
which backend `load_wildguard()` dispatches to internally (HF vs. vLLM inference),
it does not let you skip installing vllm.

Usage:
    python baselines/evaluate_wildguard.py --dataset wildguard
    python baselines/evaluate_wildguard.py --dataset jailbreakbench
    python baselines/evaluate_wildguard.py --dataset finance
    python baselines/evaluate_wildguard.py --dataset all
    python baselines/evaluate_wildguard.py --dataset all --smoke-test   # 10 examples/dataset
    python baselines/evaluate_wildguard.py --dataset all --no-vllm      # HF backend at inference time (vllm still required at import time)
"""

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.common import (
    load_finance_benchmark,
    load_jailbreakbench,
    load_wildguardtest,
    save_results,
    summarize,
)

MODEL_NAME = "wildguard"


def load_classifier(use_vllm: bool, batch_size: int | None):
    from wildguard import load_wildguard

    classifier = load_wildguard(
        use_vllm=use_vllm,
        device="cuda",
        ephemeral_model=False,
        batch_size=batch_size,
    )
    print(f"WildGuard classifier loaded (use_vllm={use_vllm}).")
    return classifier


def _label(v: str | None) -> str:
    """Map WildGuard's prompt_harmfulness ('harmful'/'unharmful'/None on parse failure)
    onto our binary schema. A parsing failure defaults to 'unharmful', mirroring the
    default-to-safe behavior of our own parser (utils/llama_guard.py) so the two are
    comparable rather than penalizing one model's failure mode more than the other's."""
    return v if v in ("harmful", "unharmful") else "unharmful"


def evaluate(classifier, rows: list[dict], dataset_name: str, smoke_test: bool = False):
    if smoke_test:
        rows = rows[:10]
        print(f"Smoke-test mode: {len(rows)} examples.")

    items = [{"prompt": r["prompt"]} for r in rows]
    raw_results = classifier.classify(items)

    results = []
    for row, raw in zip(rows, raw_results):
        results.append({
            **row,
            "predicted": _label(raw.get("prompt_harmfulness")),
            "raw_prompt_harmfulness": raw.get("prompt_harmfulness"),
            "is_parsing_error": raw.get("is_parsing_error"),
        })

    n_parse_err = sum(1 for r in results if r.get("is_parsing_error"))
    print(f"  Parsing errors: {n_parse_err}/{len(results)}")
    summarize(results)
    suffix = "_smoke" if smoke_test else ""
    save_results(results, MODEL_NAME, dataset_name, suffix)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WildGuard as a competitive baseline")
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "finance", "all"],
                         default="all")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-vllm", action="store_true",
                         help="Use the HuggingFace backend instead of vLLM at inference time (slower). "
                              "NOTE: vllm must still be installed either way — wildguard/wildguard.py "
                              "imports it unconditionally at module load, this flag only changes which "
                              "backend load_wildguard() dispatches to at runtime.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or getpass.getpass("HuggingFace token: ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token required (WildGuardMix is gated).", file=sys.stderr)
        sys.exit(1)
    os.environ.setdefault("HF_TOKEN", hf_token)

    classifier = load_classifier(use_vllm=not args.no_vllm, batch_size=args.batch_size)

    targets = ["wildguard", "jailbreakbench", "finance"] if args.dataset == "all" else [args.dataset]

    if "wildguard" in targets:
        print("\n=== WildGuardTest ===")
        rows = load_wildguardtest(hf_token)
        evaluate(classifier, rows, "wildguardtest", args.smoke_test)

    if "jailbreakbench" in targets:
        print("\n=== JailbreakBench ===")
        rows = load_jailbreakbench()
        evaluate(classifier, rows, "jailbreakbench", args.smoke_test)

    if "finance" in targets:
        print("\n=== Finance benchmark ===")
        rows = load_finance_benchmark()
        evaluate(classifier, rows, "finance", args.smoke_test)


if __name__ == "__main__":
    main()
