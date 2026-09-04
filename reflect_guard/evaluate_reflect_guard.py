"""
Evaluate Reflect-Guard (LoRA-finetuned Llama-Guard-3-8B) on WildGuardTest and JailbreakBench.

Usage:
    python evaluate_reflect_guard.py --dataset wildguard
    python evaluate_reflect_guard.py --dataset jailbreakbench
    python evaluate_reflect_guard.py --dataset both
"""

import argparse
import getpass
import json
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root (for utils/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # reflect_guard/ (takes precedence for config)

from config import MODEL_ID, ADAPTER_SAVE_PATH, LLAMAGUARD_COT_INSTRUCTION
from utils.llama_guard import parse_label, parse_reflection, parse_violated_categories
from utils.jailbreakbench_loader import load_jailbreakbench_artifacts

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_model(hf_token: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_SAVE_PATH)
    model.eval()
    # Ensure use_cache is on for inference (training disables it for gradient checkpointing)
    model.config.use_cache = True
    if hasattr(model, "base_model"):
        model.base_model.config.use_cache = True
    print("Reflect-Guard model loaded.")
    return model, tokenizer


def classify(prompt: str, model, tokenizer, max_new_tokens: int = 150) -> str:
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
    output_path = os.path.join(RESULTS_DIR, "wildguard_results_reflect.json")

    print("Loading WildGuardTest dataset...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    split = list(ds.keys())[0]
    dataset = ds[split]
    print(f"  Split: '{split}', examples: {len(dataset)}")

    results = []
    skipped = 0

    for row in tqdm(dataset, desc="Classifying (WildGuard)"):
        ground_truth = row.get("prompt_harm_label")
        if ground_truth is None:
            skipped += 1
            continue

        prompt = row["prompt"]
        try:
            raw = classify(prompt, model, tokenizer)
        except Exception as e:
            print(f"\nError classifying prompt: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "predicted": parse_label(raw),
            "raw_output": raw,
            "reflection": parse_reflection(raw),
            "violated_categories": parse_violated_categories(raw),
            "adversarial": row.get("adversarial"),
            "subcategory": row.get("subcategory"),
        })

    print(f"\nDone. Evaluated {len(results)} examples, skipped {skipped} (no label).")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def evaluate_jailbreakbench(model, tokenizer):
    output_path = os.path.join(RESULTS_DIR, "jailbreakbench_results_reflect.json")

    print("Loading JailbreakBench artifacts...")
    all_entries = load_jailbreakbench_artifacts()

    if not all_entries:
        print("No JBB prompts loaded.", file=sys.stderr)
        return

    print(f"\nTotal prompts to classify: {len(all_entries)}")

    results = []
    for entry in tqdm(all_entries, desc="Classifying (JailbreakBench)"):
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
            "reflection": parse_reflection(raw),
            "violated_categories": parse_violated_categories(raw),
        })

    detected = sum(1 for r in results if r["predicted"] == "harmful")
    print(f"\nDone. Detected {detected}/{len(results)} jailbreak prompts as harmful.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reflect-Guard on benchmarks")
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
