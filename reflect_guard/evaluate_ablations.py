"""
Evaluate all 5 ablation conditions on WildGuardTest and JailbreakBench.

Conditions:
  0: Clean baseline — base model + standard prompt (no reflection instruction)
  A: Prompted reflection — base model + CoT prompt (no fine-tuning)
  B: SFT labels only — ablation_b_lora + standard prompt
  C: Blind reflections — ablation_c_lora + CoT prompt
  D: Full Reflect-Guard — reflect_guard_lora + CoT prompt

Usage:
  python evaluate_ablations.py --condition 0 --dataset both
  python evaluate_ablations.py --condition all --dataset both
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_ID,
    ADAPTER_SAVE_PATH,
    ABLATION_B_ADAPTER_PATH,
    ABLATION_C_ADAPTER_PATH,
    LLAMAGUARD_COT_INSTRUCTION,
    LLAMAGUARD_STANDARD_INSTRUCTION,
)
from utils.llama_guard import parse_label, parse_reflection, parse_violated_categories

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

JBB_METHODS = [
    "PAIR", "GCG", "AutoDAN", "TAP", "JBC",
    "PAP-top5", "DrAttack", "Persuasive", "Persuasive+Jailbreak",
]
JBB_TARGET_MODEL = "vicuna-13b-v1.5"

# Condition definitions: (name, adapter_path, instruction_template, max_new_tokens)
CONDITIONS = {
    "0": ("clean_baseline", None, LLAMAGUARD_STANDARD_INSTRUCTION, 50),
    "a": ("ablation_a", None, LLAMAGUARD_COT_INSTRUCTION, 150),
    "b": ("ablation_b", ABLATION_B_ADAPTER_PATH, LLAMAGUARD_STANDARD_INSTRUCTION, 50),
    "c": ("ablation_c", ABLATION_C_ADAPTER_PATH, LLAMAGUARD_COT_INSTRUCTION, 150),
    "d": ("reflect_guard", ADAPTER_SAVE_PATH, LLAMAGUARD_COT_INSTRUCTION, 150),
}


def load_base_model(hf_token: str):
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
    return model, tokenizer


def load_model_for_condition(condition_key: str, hf_token: str):
    name, adapter_path, instruction, max_tokens = CONDITIONS[condition_key]
    print(f"\nLoading model for condition: {name}")

    model, tokenizer = load_base_model(hf_token)

    if adapter_path is not None:
        print(f"  Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    model.config.use_cache = True
    if hasattr(model, "base_model"):
        model.base_model.config.use_cache = True

    print(f"  Model ready. Instruction: {'CoT' if 'reflection' in instruction else 'Standard'}")
    return model, tokenizer, instruction, max_tokens, name


def classify(prompt: str, model, tokenizer, instruction: str, max_new_tokens: int) -> str:
    formatted = instruction.format(prompt=prompt)
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


def evaluate_wildguard(model, tokenizer, instruction, max_tokens, condition_name, hf_token):
    output_path = os.path.join(RESULTS_DIR, f"wildguard_results_{condition_name}.json")

    print(f"\nEvaluating WildGuardTest for condition: {condition_name}")
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    split = list(ds.keys())[0]
    dataset = ds[split]
    print(f"  {len(dataset)} examples")

    results = []
    for row in tqdm(dataset, desc=f"WildGuard ({condition_name})"):
        ground_truth = row.get("prompt_harm_label")
        if ground_truth is None:
            continue

        try:
            raw = classify(row["prompt"], model, tokenizer, instruction, max_tokens)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            "prompt": row["prompt"],
            "ground_truth": ground_truth,
            "predicted": parse_label(raw),
            "raw_output": raw,
            "reflection": parse_reflection(raw),
            "violated_categories": parse_violated_categories(raw),
            "adversarial": row.get("adversarial"),
            "subcategory": row.get("subcategory"),
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {len(results)} results saved to {output_path}")


def evaluate_jailbreakbench(model, tokenizer, instruction, max_tokens, condition_name):
    output_path = os.path.join(RESULTS_DIR, f"jailbreakbench_results_{condition_name}.json")

    try:
        import jailbreakbench as jbb
    except ImportError:
        print("ERROR: jailbreakbench not installed.", file=sys.stderr)
        return

    print(f"\nEvaluating JailbreakBench for condition: {condition_name}")
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
            print(f"  {method}: {len(entries)} prompts")
        except Exception as e:
            print(f"  Skipping '{method}': {e}", file=sys.stderr)

    if not all_entries:
        print("No JBB prompts loaded.", file=sys.stderr)
        return

    results = []
    for entry in tqdm(all_entries, desc=f"JBB ({condition_name})"):
        try:
            raw = classify(entry["prompt"], model, tokenizer, instruction, max_tokens)
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
    print(f"  Detected {detected}/{len(results)} as harmful")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


def main():
    import gc

    parser = argparse.ArgumentParser(description="Evaluate ablation conditions")
    parser.add_argument("--condition", default="all",
                        help="Condition to evaluate: 0, a, b, c, d, or all")
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "both"],
                        default="both")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                or getpass.getpass("HF token: ").strip())
    login(token=hf_token)

    conditions = list(CONDITIONS.keys()) if args.condition == "all" else [args.condition]

    for cond in conditions:
        if cond not in CONDITIONS:
            print(f"Unknown condition: {cond}", file=sys.stderr)
            continue

        model, tokenizer, instruction, max_tokens, name = load_model_for_condition(cond, hf_token)

        if args.dataset in ("wildguard", "both"):
            evaluate_wildguard(model, tokenizer, instruction, max_tokens, name, hf_token)

        if args.dataset in ("jailbreakbench", "both"):
            evaluate_jailbreakbench(model, tokenizer, instruction, max_tokens, name)

        # Free GPU memory before next condition
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
