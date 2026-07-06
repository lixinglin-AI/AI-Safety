"""
Evaluate ReflectGuard (or baseline Llama-Guard-3-8B) on the FinSafetyBench benchmark.

Usage:
    python fin/evaluate_fin.py --model reflect  [--hf-token TOKEN]
    python fin/evaluate_fin.py --model baseline [--hf-token TOKEN]
    python fin/evaluate_fin.py --model both     [--hf-token TOKEN]
    python fin/evaluate_fin.py --model reflect --smoke-test  # 10 examples only

    # On Mac (no CUDA) — fp16 on CPU/MPS, no bitsandbytes:
    python fin/evaluate_fin.py --model both --no-quantize
"""

import argparse
import getpass
import json
import os
import sys

import torch
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Allow imports from root and reflect_guard/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reflect_guard"))

from config import MODEL_ID, ADAPTER_SAVE_PATH, LLAMAGUARD_COT_INSTRUCTION, LLAMAGUARD_STANDARD_INSTRUCTION
from utils.llama_guard import parse_label, parse_reflection, parse_violated_categories
from fin.fin_config import FIN_BENCHMARK_PATH, FIN_RESULTS_DIR


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _device_and_dtype(no_quantize: bool):
    if no_quantize:
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        elif torch.cuda.is_available():
            return "cuda", torch.float16
        else:
            return "cpu", torch.float32
    return "auto", None  # bitsandbytes handles dtype


def load_reflect_guard(hf_token: str, no_quantize: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    device, dtype = _device_and_dtype(no_quantize)
    kwargs = dict(token=hf_token, device_map=device)
    if no_quantize:
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["quantization_config"] = _bnb_config()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model = PeftModel.from_pretrained(base, ADAPTER_SAVE_PATH)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "base_model"):
        model.base_model.config.use_cache = True
    print(f"ReflectGuard loaded (device={device}, quantize={not no_quantize}).")
    return model, tokenizer


def load_baseline(hf_token: str, no_quantize: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    device, dtype = _device_and_dtype(no_quantize)
    kwargs = dict(token=hf_token, device_map=device)
    if no_quantize:
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["quantization_config"] = _bnb_config()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print(f"Baseline loaded (device={device}, quantize={not no_quantize}).")
    return model, tokenizer



def classify(prompt: str, model, tokenizer, instruction_template: str, max_new_tokens: int = 150) -> str:
    formatted = instruction_template.format(prompt=prompt)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    attn_mask = encoded["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def evaluate(model, tokenizer, instruction_template: str, output_path: str, smoke_test: bool = False):
    if not os.path.exists(FIN_BENCHMARK_PATH):
        print(f"ERROR: benchmark not found at {FIN_BENCHMARK_PATH}")
        print("Run: python fin/build_benchmark.py first.")
        sys.exit(1)

    rows = []
    with open(FIN_BENCHMARK_PATH) as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    if smoke_test:
        rows = rows[:10]
        print(f"Smoke-test mode: evaluating {len(rows)} examples.")

    results = []
    for row in tqdm(rows, desc=f"Evaluating ({os.path.basename(output_path)})"):
        try:
            raw = classify(row["prompt"], model, tokenizer, instruction_template)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raw = "error"

        results.append({
            "prompt":              row["prompt"],
            "ground_truth":        row["label"],
            "predicted":           parse_label(raw),
            "raw_output":          raw,
            "reflection":          parse_reflection(raw),
            "violated_categories": parse_violated_categories(raw),
            "category":            row["category"],
            "source":              row["source"],
            "subcategory":         row["subcategory"],
            "adversarial":         row.get("adversarial", False),
        })

    harmful_detected = sum(1 for r in results if r["ground_truth"] == "harmful" and r["predicted"] == "harmful")
    harmful_total    = sum(1 for r in results if r["ground_truth"] == "harmful")
    print(f"\nDetected {harmful_detected}/{harmful_total} harmful prompts as unsafe.")

    os.makedirs(FIN_RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReflectGuard-Fin on FinSafetyBench")
    parser.add_argument(
        "--model", choices=["reflect", "baseline", "both"], default="both",
        help="Which model(s) to evaluate"
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run on first 10 examples only (sanity check)",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip 4-bit bitsandbytes quantization; use fp16 on MPS/CUDA or fp32 on CPU. "
             "Required on Mac (no CUDA).",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or getpass.getpass("HuggingFace token: ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token required.", file=sys.stderr)
        sys.exit(1)
    login(token=hf_token)

    suffix = "_smoke" if args.smoke_test else ""

    if args.model in ("reflect", "both"):
        model, tokenizer = load_reflect_guard(hf_token, no_quantize=args.no_quantize)
        out = os.path.join(FIN_RESULTS_DIR, f"fin_results_reflect{suffix}.json")
        evaluate(model, tokenizer, LLAMAGUARD_COT_INSTRUCTION, out, args.smoke_test)
        del model

    if args.model in ("baseline", "both"):
        model, tokenizer = load_baseline(hf_token, no_quantize=args.no_quantize)
        out = os.path.join(FIN_RESULTS_DIR, f"fin_results_baseline{suffix}.json")
        evaluate(model, tokenizer, LLAMAGUARD_STANDARD_INSTRUCTION, out, args.smoke_test)


if __name__ == "__main__":
    main()
