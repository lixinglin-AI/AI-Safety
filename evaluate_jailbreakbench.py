"""
Evaluate Llama-Guard-3-8B on JailbreakBench jailbreak prompts.

Loads all available attack method artifacts and classifies their prompts.
Ground truth: all JBB prompts are adversarial/harmful by definition.
Output: results/jailbreakbench_results.json
"""

import json
import os
import sys

import jailbreakbench as jbb
from tqdm import tqdm

from config import RESULTS_DIR
from utils.llama_guard import parse_label, parse_violated_categories
from utils.ollama_client import classify

# All attack methods available in JailbreakBench
# https://jailbreakbench.github.io/
JBB_METHODS = [
    "PAIR",
    "GCG",
    "AutoDAN",
    "TAP",
    "JBC",
    "PAP-top5",
    "DrAttack",
    "Persuasive",
    "Persuasive+Jailbreak",
]

# Target model used to retrieve artifacts (prompts were originally crafted for this model)
JBB_TARGET_MODEL = "vicuna-13b-v1.5"


def load_method_artifacts(method: str) -> list[dict]:
    """Attempt to load artifacts for a given method. Returns [] on failure."""
    try:
        artifact = jbb.read_artifact(method=method, model_name=JBB_TARGET_MODEL)
        entries = []
        for jb in artifact.jailbreaks:
            if jb.prompt is None:
                continue
            entries.append({
                "method": method,
                "behavior_id": getattr(jb, "behavior_id", None),
                "behavior": getattr(jb, "goal", None) or getattr(jb, "behavior", None),
                "prompt": jb.prompt,
                "jbb_success": getattr(jb, "jailbroken", None),
            })
        return entries
    except Exception as e:
        print(f"  Skipping method '{method}': {e}", file=sys.stderr)
        return []


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "jailbreakbench_results.json")

    print("Loading JailbreakBench artifacts...")
    all_entries = []
    for method in JBB_METHODS:
        entries = load_method_artifacts(method)
        print(f"  {method}: {len(entries)} prompts loaded")
        all_entries.extend(entries)

    if not all_entries:
        print("No JBB prompts loaded. Check your jailbreakbench installation.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal prompts to classify: {len(all_entries)}")

    results = []
    for entry in tqdm(all_entries, desc="Classifying"):
        try:
            raw = classify(entry["prompt"])
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            **entry,
            # Ground truth: all JBB prompts are harmful
            "ground_truth": "harmful",
            "predicted": parse_label(raw),
            "raw_output": raw,
            "violated_categories": parse_violated_categories(raw),
        })

    detected = sum(1 for r in results if r["predicted"] == "harmful")
    print(f"\nDone. Detected {detected}/{len(results)} jailbreak prompts as harmful.")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
