"""
Evaluate baseline Llama-Guard-3-8B (no LoRA) on WildGuardTest and JailbreakBench.
For use on HPC where Ollama is not available.

Usage:
    python evaluate_baseline_hpc.py --dataset wildguard
    python evaluate_baseline_hpc.py --dataset jailbreakbench
    python evaluate_baseline_hpc.py --dataset both
"""

import argparse
import getpass
import json
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root (for utils/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # reflect_guard/ (takes precedence for config)

from config import MODEL_ID, LLAMAGUARD_COT_INSTRUCTION
from utils.llama_guard import parse_label, parse_violated_categories

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

JBB_METHODS = [
    "PAIR", "GCG", "AutoDAN", "TAP", "JBC",
    "PAP-top5", "DrAttack", "Persuasive", "Persuasive+Jailbreak",
]
JBB_TARGET_MODEL = "vicuna-13b-v1.5"


def load_model(hf_token: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Baseline model loaded.")
    return model, tokenizer


def classify(prompt: str, model, tokenizer, max_new_tokens: int = 100) -> str:
    # Baseline uses standard Llama-Guard format (no LLAMAGUARD_COT_INSTRUCTION wrapper)
    formatted = LLAMAGUARD_COT_INSTRUCTION.format(prompt=prompt)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def evaluate_wildguard(model, tokenizer, hf_token: str):
    output_path = os.path.join(RESULTS_DIR, "wildguard_results_baseline.json")

    if os.path.exists(output_path):
        print(f"Baseline WildGuard results already exist at {output_path}. Skipping.")
        return

    print("Loading WildGuardTest dataset...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    split = list(ds.keys())[0]
    dataset = ds[split]
    print(f"  Split: '{split}', examples: {len(dataset)}")

    results = []
    skipped = 0

    for row in tqdm(dataset, desc="Classifying (baseline WildGuard)"):
        ground_truth = row.get("prompt_harm_label")
        if ground_truth is None:
            skipped += 1
            continue

        prompt = row["prompt"]
        try:
            raw = classify(prompt, model, tokenizer)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
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

    print(f"\nDone. Evaluated {len(results)} examples, skipped {skipped}.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def evaluate_jailbreakbench(model, tokenizer):
    output_path = os.path.join(RESULTS_DIR, "jailbreakbench_results_baseline.json")

    try:
        import jailbreakbench as jbb
    except ImportError:
        print("ERROR: jailbreakbench not installed. Run: pip install jailbreakbench", file=sys.stderr)
        sys.exit(1)

    print("Loading JailbreakBench artifacts...")
    all_entries = []
    for method in JBB_METHODS:
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
            all_entries.extend(entries)
            print(f"  {method}: {len(entries)} prompts loaded")
        except Exception as e:
            print(f"  Skipping '{method}': {e}", file=sys.stderr)

    if not all_entries:
        print("No JBB prompts loaded.", file=sys.stderr)
        return

    print(f"\nTotal prompts: {len(all_entries)}")

    results = []
    for entry in tqdm(all_entries, desc="Classifying (baseline JailbreakBench)"):
        try:
            raw = classify(entry["prompt"], model, tokenizer)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            **entry,
            "ground_truth": "harmful",
            "predicted": parse_label(raw),
            "raw_output": raw,
            "violated_categories": parse_violated_categories(raw),
        })

    detected = sum(1 for r in results if r["predicted"] == "harmful")
    print(f"\nDone. Detected {detected}/{len(results)} as harmful.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Llama-Guard-3-8B on HPC")
    parser.add_argument(
        "--dataset",
        choices=["wildguard", "jailbreakbench", "both"],
        default="both",
        help="Which benchmark to evaluate on",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var; falls back to interactive prompt)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or getpass.getpass("Enter your HuggingFace token: ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token is required.", file=sys.stderr)
        sys.exit(1)
    login(token=hf_token)

    model, tokenizer = load_model(hf_token)

    if args.dataset in ("wildguard", "both"):
        evaluate_wildguard(model, tokenizer, hf_token)

    if args.dataset in ("jailbreakbench", "both"):
        evaluate_jailbreakbench(model, tokenizer)


if __name__ == "__main__":
    main()
