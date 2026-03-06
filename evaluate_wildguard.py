"""
Evaluate Llama-Guard-3-8B baseline performance on WildGuardTest.

Dataset: allenai/wildguardmix (wildguardtest split, ~1725 examples)
Ground truth: prompt_harm_label ("harmful" / "unharmful"; skip if None)
Output: results/wildguard_results.json
"""

import json
import os
import sys

from datasets import load_dataset
from huggingface_hub import HfFolder
from tqdm import tqdm

from config import RESULTS_DIR
from utils.llama_guard import parse_label, parse_violated_categories
from utils.ollama_client import classify


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "wildguard_results.json")

    token = HfFolder.get_token()
    if token is None:
        print("ERROR: Not logged in to HuggingFace. Run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    print("Loading WildGuardTest dataset...")
    dataset = load_dataset("allenai/wildguardmix", "wildguardtest", split="test", token=token)
    print(f"  Total examples: {len(dataset)}")

    results = []
    skipped = 0

    for row in tqdm(dataset, desc="Classifying"):
        ground_truth = row.get("prompt_harm_label")
        if ground_truth is None:
            skipped += 1
            continue

        prompt = row["prompt"]
        try:
            raw = classify(prompt)
        except Exception as e:
            print(f"\nError classifying prompt: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "predicted": parse_label(raw),
            "raw_output": raw,
            "violated_categories": parse_violated_categories(raw),
            "adversarial": row.get("adversarial"),
            "subcategory": row.get("subcategory"),
        })

    print(f"\nDone. Evaluated {len(results)} examples, skipped {skipped} (no label).")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
