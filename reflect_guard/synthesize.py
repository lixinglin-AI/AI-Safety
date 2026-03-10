"""
Step 1: Synthesize training data from WildGuardMix, AdvBench, and HarmBench.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from tqdm import tqdm

from config import (
    DATASET_PATH, SKIP_SYNTHESIS, MAX_PER_SOURCE,
    OPENAI_MODEL, RANDOM_SEED,
    LLAMAGUARD_COT_INSTRUCTION, SYNTHESIS_SYSTEM,
)


def make_completion(reflection: str, label: str, categories: list) -> str:
    verdict = "unsafe" if label == "harmful" else "safe"
    lines = [f"<reflection>\n{reflection}\n</reflection>", verdict]
    if verdict == "unsafe" and categories:
        lines.append(", ".join(categories))
    return "\n".join(lines)


def synthesize_reflection(client, prompt: str, label: str, retries: int = 3) -> str:
    user_msg = (
        f"User message: {prompt[:1500]}\n\n"
        f"Ground-truth label: {label}\n\n"
        "Write the reflection:"
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"[synthesis error: {e}]"
            time.sleep(2 ** attempt)


def load_sources(token: str) -> list:
    random.seed(RANDOM_SEED)
    all_examples = []

    # Source 1: WildGuardMix
    print("Loading WildGuardMix (train)...")
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", token=token, trust_remote_code=True)
        rows = list(ds[list(ds.keys())[0]])
        examples = [
            {"prompt": r["prompt"], "label": r["prompt_harm_label"], "categories": [], "source": "wildguardmix"}
            for r in rows if r.get("prompt_harm_label") in ("harmful", "unharmful")
        ]
        random.shuffle(examples)
        all_examples.extend(examples[:MAX_PER_SOURCE])
        print(f"  -> {min(len(examples), MAX_PER_SOURCE)} examples")
    except Exception as e:
        print(f"  WildGuardMix failed: {e}")

    # Source 2: AdvBench
    print("Loading AdvBench...")
    try:
        ds = load_dataset("walledai/AdvBench", token=token, trust_remote_code=True)
        rows = list(ds[list(ds.keys())[0]])
        col = "prompt" if "prompt" in rows[0] else "instruction"
        examples = [
            {"prompt": r[col], "label": "harmful", "categories": ["S1"], "source": "advbench"}
            for r in rows if r.get(col)
        ]
        random.shuffle(examples)
        all_examples.extend(examples[:MAX_PER_SOURCE])
        print(f"  -> {min(len(examples), MAX_PER_SOURCE)} examples")
    except Exception as e:
        print(f"  AdvBench failed: {e}")

    # Source 3: HarmBench
    print("Loading HarmBench...")
    try:
        ds = load_dataset("cais/harmbench", "behaviors", token=token, trust_remote_code=True)
        rows = list(ds[list(ds.keys())[0]])
        col = next(
            (c for c in rows[0].keys() if c.lower() in ("behavior", "prompt", "instruction")),
            list(rows[0].keys())[0]
        )
        examples = [
            {"prompt": r[col], "label": "harmful", "categories": ["S1"], "source": "harmbench"}
            for r in rows if r.get(col)
        ]
        random.shuffle(examples)
        all_examples.extend(examples[:MAX_PER_SOURCE])
        print(f"  -> {min(len(examples), MAX_PER_SOURCE)} examples")
    except Exception as e:
        print(f"  HarmBench failed: {e}")

    random.shuffle(all_examples)
    print(f"\nTotal examples to process: {len(all_examples)}")
    return all_examples


def run(hf_token: str, openai_key: str | None = None):
    all_examples = load_sources(hf_token)

    client = None
    if not SKIP_SYNTHESIS:
        if not openai_key:
            raise ValueError("OpenAI key required when SKIP_SYNTHESIS=False")
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

    records = []
    desc = "Synthesizing" if not SKIP_SYNTHESIS else "Building placeholders"
    for ex in tqdm(all_examples, desc=desc):
        if SKIP_SYNTHESIS:
            reflection = (
                f"This prompt is labeled as {ex['label']}. "
                "The request appears straightforward with no adversarial framing detected. "
                "[Placeholder - set SKIP_SYNTHESIS=False to generate real reflections.]"
            )
        else:
            reflection = synthesize_reflection(client, ex["prompt"], ex["label"])

        records.append({
            "source": ex["source"],
            "label":  ex["label"],
            "messages": [
                {"role": "user",      "content": LLAMAGUARD_COT_INSTRUCTION.format(prompt=ex["prompt"])},
                {"role": "assistant", "content": make_completion(reflection, ex["label"], ex["categories"])},
            ],
        })

    with open(DATASET_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    label_counts = {}
    src_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        src_counts[r["source"]]  = src_counts.get(r["source"], 0) + 1

    print(f"\n✓ {len(records)} examples saved to {DATASET_PATH}")
    print("Label distribution:", label_counts)
    print("Source distribution:", src_counts)
