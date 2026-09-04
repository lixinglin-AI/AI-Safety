"""
Evaluate Llama-Guard-3-8B on JailbreakBench jailbreak prompts.

Loads all available attack method artifacts and classifies their prompts.
Ground truth: all JBB prompts are adversarial/harmful by definition.
Output: results/jailbreakbench_results.json
"""

import json
import os
import sys

from tqdm import tqdm

from config import RESULTS_DIR
from utils.llama_guard import parse_label, parse_violated_categories
from utils.ollama_client import classify
from utils.jailbreakbench_loader import load_jailbreakbench_artifacts


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "jailbreakbench_results.json")

    print("Loading JailbreakBench artifacts...")
    all_entries = load_jailbreakbench_artifacts()

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
