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
from utils.jailbreakbench_loader import load_jailbreakbench_artifacts

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIN_BENCHMARK_PATH = os.path.join(ROOT_DIR, "fin", "fin_benchmark.jsonl")

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


def load_model_for_condition(condition_key: str, hf_token: str, seed: int = 42):
    name, adapter_path, instruction, max_tokens = CONDITIONS[condition_key]

    # Conditions b/c/d have a trained adapter and thus a seed-dependent checkpoint;
    # 0/a are training-free (deterministic base model, greedy decoding) so seed is
    # a no-op for them. 42 is the original run's seed — reuse its unsuffixed
    # adapter/result-file paths as-is rather than requiring a redundant retrain.
    if adapter_path is not None and seed != 42:
        adapter_path = f"{adapter_path}_seed{seed}"
        name = f"{name}_seed{seed}"

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


def evaluate_wildguard(model, tokenizer, instruction, max_tokens, condition_name, hf_token, force=False):
    output_path = os.path.join(RESULTS_DIR, f"wildguard_results_{condition_name}.json")
    if os.path.exists(output_path) and not force:
        print(f"\n[skip] {output_path} already exists (pass --force to re-evaluate).")
        return

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


def evaluate_finance(model, tokenizer, instruction, max_tokens, condition_name, force=False):
    output_path = os.path.join(RESULTS_DIR, f"fin_results_{condition_name}.json")
    if os.path.exists(output_path) and not force:
        print(f"\n[skip] {output_path} already exists (pass --force to re-evaluate).")
        return

    if not os.path.exists(FIN_BENCHMARK_PATH):
        print(f"ERROR: finance benchmark not found at {FIN_BENCHMARK_PATH}. "
              "Run fin/build_benchmark.py first.", file=sys.stderr)
        return

    print(f"\nEvaluating finance benchmark for condition: {condition_name}")
    rows = []
    with open(FIN_BENCHMARK_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows)} examples")

    results = []
    for row in tqdm(rows, desc=f"Finance ({condition_name})"):
        try:
            raw = classify(row["prompt"], model, tokenizer, instruction, max_tokens)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            "prompt": row["prompt"],
            "ground_truth": row["label"],
            "predicted": parse_label(raw),
            "raw_output": raw,
            "reflection": parse_reflection(raw),
            "violated_categories": parse_violated_categories(raw),
            "category": row["category"],
            "source": row["source"],
            "subcategory": row["subcategory"],
            "adversarial": row.get("adversarial", False),
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {len(results)} results saved to {output_path}")


def evaluate_jailbreakbench(model, tokenizer, instruction, max_tokens, condition_name, force=False):
    output_path = os.path.join(RESULTS_DIR, f"jailbreakbench_results_{condition_name}.json")
    if os.path.exists(output_path) and not force:
        print(f"\n[skip] {output_path} already exists (pass --force to re-evaluate).")
        return

    print(f"\nEvaluating JailbreakBench for condition: {condition_name}")
    all_entries = load_jailbreakbench_artifacts()

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
    parser.add_argument("--dataset", choices=["wildguard", "jailbreakbench", "finance", "both", "all"],
                        default="both",
                        help="'both' = wildguard+jailbreakbench (original behavior); "
                             "'all' = wildguard+jailbreakbench+finance")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=42,
                         help="Which trained seed's adapter to evaluate for conditions b/c/d "
                              "(42 = original run, unsuffixed paths; other values expect "
                              "{adapter}_seed{N} to already exist from train_ablations.py/"
                              "main.py --seed {N}). No effect on conditions 0/a.")
    parser.add_argument("--force", action="store_true",
                         help="Re-evaluate even if a results file already exists (default: skip "
                              "any dataset whose output file is already on disk, and skip loading "
                              "the model entirely if every requested dataset for this condition is "
                              "already done — useful after a preempted/requeued job).")
    args = parser.parse_args()

    hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                or getpass.getpass("HF token: ").strip())
    login(token=hf_token)

    conditions = list(CONDITIONS.keys()) if args.condition == "all" else [args.condition]
    datasets = {"both": ["wildguard", "jailbreakbench"], "all": ["wildguard", "jailbreakbench", "finance"]}.get(
        args.dataset, [args.dataset])

    for cond in conditions:
        if cond not in CONDITIONS:
            print(f"Unknown condition: {cond}", file=sys.stderr)
            continue

        # Compute the seed-suffixed condition name up front (mirrors
        # load_model_for_condition's own logic) so we can skip loading the
        # ~5GB model entirely if every requested dataset is already evaluated.
        base_name, adapter_path, _, _ = CONDITIONS[cond]
        name_preview = f"{base_name}_seed{args.seed}" if (adapter_path is not None and args.seed != 42) else base_name
        out_paths = {
            "wildguard": os.path.join(RESULTS_DIR, f"wildguard_results_{name_preview}.json"),
            "jailbreakbench": os.path.join(RESULTS_DIR, f"jailbreakbench_results_{name_preview}.json"),
            "finance": os.path.join(RESULTS_DIR, f"fin_results_{name_preview}.json"),
        }
        if not args.force and all(os.path.exists(out_paths[d]) for d in datasets):
            print(f"\n[skip] Condition {cond} ({name_preview}): all requested datasets already "
                  "evaluated, not loading model. Pass --force to re-run.")
            continue

        model, tokenizer, instruction, max_tokens, name = load_model_for_condition(cond, hf_token, args.seed)

        if "wildguard" in datasets:
            evaluate_wildguard(model, tokenizer, instruction, max_tokens, name, hf_token, args.force)

        if "jailbreakbench" in datasets:
            evaluate_jailbreakbench(model, tokenizer, instruction, max_tokens, name, args.force)

        if "finance" in datasets:
            evaluate_finance(model, tokenizer, instruction, max_tokens, name, args.force)

        # Free GPU memory before next condition
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
