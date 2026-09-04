"""
Benchmark inference latency, throughput, and GPU memory for the clean baseline
(Condition 0), SFT-only (Condition B), and full Reflect-Guard (Condition D) on
identical hardware and prompts, so the paper's efficiency claims can be backed by
measurements instead of the "negligible overhead" language that was removed in the
last revision (see reflect_guard_mdpi-2.tex, Section "Efficiency Considerations").

Two scenarios are measured for each condition:
  1. Single-request latency: prompts processed one at a time (batch size 1),
     matching how the existing evaluate_*.py scripts already run and reflecting a
     low-QPS deployment. Reports P50/P95/P99/mean wall-clock latency per prompt.
  2. Batched throughput: prompts processed in batches (default 8) with padding,
     reflecting a higher-QPS serving scenario. Reports prompts/sec and tokens/sec.

Both scenarios also record generated-token counts per prompt (to check the "~50-100
additional reflection tokens" claim directly) and peak GPU memory
(torch.cuda.max_memory_allocated) for the whole run.

A fixed, identical set of prompts (sampled from WildGuardTest with a fixed seed) is
used across all three conditions so latency differences reflect the model/adapter,
not prompt-length variation. Run all three conditions in the SAME job/GPU/driver
state for a fair comparison — see benchmark_latency_job.sh.

Usage:
    python reflect_guard/benchmark_latency.py --condition 0 --hf-token TOKEN
    python reflect_guard/benchmark_latency.py --condition B --hf-token TOKEN
    python reflect_guard/benchmark_latency.py --condition D --hf-token TOKEN
    python reflect_guard/benchmark_latency.py --condition all --n-prompts 200 --batch-size 8
"""

import argparse
import getpass
import json
import os
import statistics
import sys
import time

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_ID, ADAPTER_SAVE_PATH, ABLATION_B_ADAPTER_PATH, LLAMAGUARD_COT_INSTRUCTION

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency")

# NOTE: the original seed=42 Condition B adapter (ABLATION_B_ADAPTER_PATH,
# unsuffixed) was never actually persisted on this machine — only its eval
# results were. Using the seed=123 adapter from the multi-seed run instead:
# latency depends on adapter architecture/rank, not the specific trained
# weights, so this is a valid substitute for a timing benchmark.
CONDITIONS = {
    "0": ("Clean Baseline", None),
    "B": ("SFT Labels Only", f"{ABLATION_B_ADAPTER_PATH}_seed123"),
    "D": ("Full Reflect-Guard", ADAPTER_SAVE_PATH),
}

N_WARMUP = 5
RANDOM_SEED = 42
MAX_NEW_TOKENS = 150


def load_model(adapter_path: str | None, hf_token: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=hf_token, quantization_config=bnb_config, device_map="auto",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "base_model"):
        model.base_model.config.use_cache = True
    return model, tokenizer


def load_fixed_prompts(n: int, hf_token: str) -> list[str]:
    import random
    ds = load_dataset("allenai/wildguardmix", "wildguardtest", token=hf_token)
    split = list(ds.keys())[0]
    dataset = ds[split]
    all_prompts = [row["prompt"] for row in dataset if row.get("prompt_harm_label") is not None]
    random.Random(RANDOM_SEED).shuffle(all_prompts)
    return all_prompts[:n]


def build_input(prompt: str, tokenizer):
    formatted = LLAMAGUARD_COT_INSTRUCTION.format(prompt=prompt)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted}], tokenize=False, add_generation_prompt=True,
    )


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_single_request(model, tokenizer, prompts: list[str]) -> dict:
    texts = [build_input(p, tokenizer) for p in prompts]

    # Warmup (not timed) — first few CUDA kernel launches / cudnn autotune are slow
    # and would otherwise bias the first timed samples.
    for t in texts[:N_WARMUP]:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    cuda_sync()

    latencies_s = []
    token_counts = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False).to(model.device)
        cuda_sync()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        cuda_sync()
        elapsed = time.perf_counter() - start
        n_generated = out.shape[1] - enc["input_ids"].shape[1]
        latencies_s.append(elapsed)
        token_counts.append(n_generated)

    latencies_s.sort()

    def pct(p):
        idx = min(len(latencies_s) - 1, int(p * len(latencies_s)))
        return latencies_s[idx]

    return {
        "n": len(prompts),
        "mean_latency_s": statistics.mean(latencies_s),
        "p50_latency_s": pct(0.50),
        "p95_latency_s": pct(0.95),
        "p99_latency_s": pct(0.99),
        "mean_generated_tokens": statistics.mean(token_counts),
        "raw_latencies_s": latencies_s,
        "raw_token_counts": token_counts,
    }


def bench_batched_throughput(model, tokenizer, prompts: list[str], batch_size: int) -> dict:
    texts = [build_input(p, tokenizer) for p in prompts]
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    # Warmup on the first batch.
    enc = tokenizer(batches[0], return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    cuda_sync()

    total_prompts = 0
    total_tokens = 0
    start = time.perf_counter()
    for batch in batches:
        enc = tokenizer(batch, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        n_generated = (out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0]
        total_tokens += n_generated
        total_prompts += len(batch)
    cuda_sync()
    elapsed = time.perf_counter() - start

    return {
        "n": total_prompts,
        "batch_size": batch_size,
        "wall_time_s": elapsed,
        "prompts_per_sec": total_prompts / elapsed,
        "tokens_per_sec": total_tokens / elapsed,
    }


def run_condition(cond: str, prompts: list[str], batch_size: int, hf_token: str) -> dict:
    name, adapter_path = CONDITIONS[cond]
    print(f"\n=== Condition {cond}: {name} ===")
    model, tokenizer = load_model(adapter_path, hf_token)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    single = bench_single_request(model, tokenizer, prompts)
    print(f"  single-request: mean={single['mean_latency_s']:.3f}s  "
          f"P50={single['p50_latency_s']:.3f}s  P95={single['p95_latency_s']:.3f}s  "
          f"P99={single['p99_latency_s']:.3f}s  mean_tokens={single['mean_generated_tokens']:.1f}")

    batched = bench_batched_throughput(model, tokenizer, prompts, batch_size)
    print(f"  batched (bs={batch_size}): {batched['prompts_per_sec']:.2f} prompts/sec, "
          f"{batched['tokens_per_sec']:.1f} tokens/sec")

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  peak GPU memory: {peak_mem_gb:.2f} GB")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "condition": cond,
        "name": name,
        "single_request": single,
        "batched_throughput": batched,
        "peak_gpu_memory_gb": peak_mem_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency/throughput/GPU memory across conditions 0/B/D")
    parser.add_argument("--condition", choices=["0", "B", "D", "all"], default="all")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or getpass.getpass("HuggingFace token: ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token required.", file=sys.stderr)
        sys.exit(1)
    login(token=hf_token)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device detected. Latency/throughput numbers from this "
              "run are NOT meaningful for the paper (CPU timings are not comparable to "
              "the A5000 GPU numbers everything else in this repo was measured on) — "
              "use this only to smoke-test that the script runs without errors.",
              file=sys.stderr)

    prompts = load_fixed_prompts(args.n_prompts, hf_token)
    print(f"Benchmarking on {len(prompts)} fixed WildGuardTest prompts (seed={RANDOM_SEED}).")

    conditions = ["0", "B", "D"] if args.condition == "all" else [args.condition]
    results = {}
    for cond in conditions:
        results[cond] = run_condition(cond, prompts, args.batch_size, hf_token)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"latency_benchmark_{'_'.join(conditions)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if len(conditions) > 1:
        print("\n=== Summary ===")
        print(f"{'Condition':12s} {'P50 (s)':>8} {'P95 (s)':>8} {'Mean tok':>9} {'Prompts/s':>10} {'Peak GB':>8}")
        for cond in conditions:
            r = results[cond]
            mem = f"{r['peak_gpu_memory_gb']:.2f}" if r["peak_gpu_memory_gb"] is not None else "  n/a"
            print(f"{r['name']:12s} {r['single_request']['p50_latency_s']:8.3f} "
                  f"{r['single_request']['p95_latency_s']:8.3f} "
                  f"{r['single_request']['mean_generated_tokens']:9.1f} "
                  f"{r['batched_throughput']['prompts_per_sec']:10.2f} {mem:>8}")


if __name__ == "__main__":
    main()
