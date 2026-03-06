"""
Build CoT-annotated training dataset for Reflect-Guard finetuning.

Sources:
  1. allenai/wildguardmix (train split) - labeled harmful/unharmful prompts
  2. walledai/AdvBench                  - harmful behavior requests (all harmful)
  3. cais/harmbench                     - adversarial behavior prompts (all harmful)

For each prompt, calls the Anthropic API to synthesize a logical self-reflection:
  <reflection>
  ...analysis of the prompt's intent and adversarial signals...
  </reflection>
  unsafe
  S1

Output: data/reflect_guard_train.jsonl

Usage:
  python build_dataset.py [--max-per-source N] [--skip-synthesis]
  ANTHROPIC_API_KEY must be set in .env or environment.

  --max-per-source N   cap each dataset source at N examples (default: unlimited)
  --skip-synthesis     write placeholder reflections (for dry runs / testing)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Output ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "reflect_guard_train.jsonl"
CHECKPOINT_PATH = DATA_DIR / "reflect_guard_train.ckpt.jsonl"

# ── Llama-Guard 3 safety categories ─────────────────────────────────────────
SAFETY_CATEGORIES = (
    "S1: Violent Crimes.\n"
    "S2: Non-Violent Crimes.\n"
    "S3: Sex-Related Crimes.\n"
    "S4: Child Sexual Exploitation.\n"
    "S5: Defamation.\n"
    "S6: Specialized Advice.\n"
    "S7: Privacy.\n"
    "S8: Intellectual Property.\n"
    "S9: Indiscriminate Weapons.\n"
    "S10: Hate.\n"
    "S11: Suicide & Self-Harm.\n"
    "S12: Sexual Content.\n"
    "S13: Elections."
)

# ── Claude synthesis prompt ──────────────────────────────────────────────────
SYNTHESIS_SYSTEM = (
    "You are a safety analysis expert building training data for an AI safety classifier. "
    "For a given user message and its ground-truth safety label, write a concise logical "
    "self-reflection (2–4 sentences) that:\n"
    "1. Identifies whether the message uses adversarial techniques to bypass safety filters "
    "(e.g. roleplay framing, fictional scenarios, indirect harmful requests, technical "
    "obfuscation, authority appeals, social engineering).\n"
    "2. States whether the underlying intent is harmful or benign and why.\n"
    "3. Lists the key indicators that support the classification.\n\n"
    "Rules:\n"
    "- Be precise and analytical. Do NOT moralize or preach.\n"
    "- For harmful prompts: explain the harm category and any adversarial technique used.\n"
    "- For benign prompts: confirm safety and explain why.\n"
    "- Output ONLY the reflection text (2–4 sentences). No preamble, no verdict."
)

SYNTHESIS_USER_TMPL = (
    "User message: {prompt}\n\n"
    "Ground-truth label: {label}\n\n"
    "Write the reflection:"
)

# ── LLaMA-Guard CoT instruction template ────────────────────────────────────
# This is stored in each training example so the model knows what was asked.
LLAMAGUARD_COT_INSTRUCTION = (
    "Task: Check if there is unsafe content in 'User' messages in conversations "
    "according our safety policy with the below categories.\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    f"{SAFETY_CATEGORIES}\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\n"
    "User: {prompt}\n\n"
    "<END CONVERSATION>\n\n"
    "Provide your safety assessment for ONLY THE LAST User message in the above "
    "conversation:\n"
    " - First provide a logical self-reflection in <reflection>...</reflection> tags, "
    "analyzing whether the message uses adversarial techniques (roleplay bypass, fictional "
    "framing, indirect harmful requests, obfuscation, social engineering) and identifying "
    "key harm indicators.\n"
    " - After the reflection, the first line must read 'safe' or 'unsafe'.\n"
    " - If unsafe, a second line must include a comma-separated list of violated categories."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_completion(label: str, categories: list[str]) -> str:
    """Build the placeholder completion (no reflection text)."""
    verdict = "unsafe" if label == "harmful" else "safe"
    if verdict == "unsafe" and categories:
        return f"unsafe\n{', '.join(categories)}"
    return verdict


def make_full_completion(reflection: str, label: str, categories: list[str]) -> str:
    verdict = "unsafe" if label == "harmful" else "safe"
    lines = [f"<reflection>\n{reflection}\n</reflection>", verdict]
    if verdict == "unsafe" and categories:
        lines.append(", ".join(categories))
    return "\n".join(lines)


def synthesize_reflection(
    client: anthropic.Anthropic,
    prompt: str,
    label: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Call Claude to generate a reflection for the given prompt + label."""
    user_msg = SYNTHESIS_USER_TMPL.format(
        prompt=prompt[:1500],  # truncate very long prompts
        label=label,
    )
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = backoff * (2 ** attempt)
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
    return ""  # unreachable


def load_checkpoint(path: Path) -> set[str]:
    """Return set of prompt hashes already processed."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["_hash"])
                except Exception:
                    pass
    return done


def prompt_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_wildguardmix(token: str, max_samples: int | None) -> list[dict]:
    """Load WildGuardMix train split. Returns list of {prompt, label, categories}."""
    print("Loading allenai/wildguardmix (train)...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", token=token, trust_remote_code=True)
    split_name = list(ds.keys())[0]
    rows = ds[split_name]
    examples = []
    for row in rows:
        label = row.get("prompt_harm_label")
        if label not in ("harmful", "unharmful"):
            continue
        examples.append({
            "prompt": row["prompt"],
            "label": label,
            "categories": [],  # WildGuardMix doesn't provide LLaMA-Guard category codes
            "source": "wildguardmix",
        })
    if max_samples:
        random.shuffle(examples)
        examples = examples[:max_samples]
    print(f"  Loaded {len(examples)} examples from WildGuardMix.")
    return examples


def load_advbench(token: str, max_samples: int | None) -> list[dict]:
    """Load AdvBench harmful behaviors. All examples are labeled harmful."""
    print("Loading walledai/AdvBench...")
    try:
        ds = load_dataset("walledai/AdvBench", token=token, trust_remote_code=True)
        split_name = list(ds.keys())[0]
        rows = ds[split_name]
        # AdvBench has a 'prompt' or 'instruction' column
        prompt_col = "prompt" if "prompt" in rows.column_names else "instruction"
        examples = [
            {"prompt": row[prompt_col], "label": "harmful", "categories": ["S1"], "source": "advbench"}
            for row in rows if row.get(prompt_col)
        ]
    except Exception as e:
        print(f"  Could not load walledai/AdvBench: {e}")
        print("  Skipping AdvBench.")
        return []
    if max_samples:
        random.shuffle(examples)
        examples = examples[:max_samples]
    print(f"  Loaded {len(examples)} examples from AdvBench.")
    return examples


def load_harmbench(token: str, max_samples: int | None) -> list[dict]:
    """Load HarmBench behaviors. All examples are labeled harmful."""
    print("Loading cais/harmbench...")
    try:
        ds = load_dataset("cais/harmbench", "behaviors", token=token, trust_remote_code=True)
        split_name = list(ds.keys())[0]
        rows = ds[split_name]
        # HarmBench has a 'Behavior' or 'behavior' column
        prompt_col = next(
            (c for c in rows.column_names if c.lower() in ("behavior", "prompt", "instruction")),
            rows.column_names[0],
        )
        examples = [
            {"prompt": row[prompt_col], "label": "harmful", "categories": ["S1"], "source": "harmbench"}
            for row in rows if row.get(prompt_col)
        ]
    except Exception as e:
        print(f"  Could not load cais/harmbench: {e}")
        print("  Skipping HarmBench.")
        return []
    if max_samples:
        random.shuffle(examples)
        examples = examples[:max_samples]
    print(f"  Loaded {len(examples)} examples from HarmBench.")
    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Reflect-Guard CoT training dataset.")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Cap each dataset source at this many examples.")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="Skip API calls; write placeholder reflections (for dry runs).")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if not args.skip_synthesis and not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or use --skip-synthesis.")

    DATA_DIR.mkdir(exist_ok=True)

    client = anthropic.Anthropic(api_key=api_key) if not args.skip_synthesis else None

    # Load all sources
    all_examples = []
    all_examples.extend(load_wildguardmix(hf_token, args.max_per_source))
    all_examples.extend(load_advbench(hf_token, args.max_per_source))
    all_examples.extend(load_harmbench(hf_token, args.max_per_source))

    random.seed(42)
    random.shuffle(all_examples)
    print(f"\nTotal examples to process: {len(all_examples)}")

    # Load checkpoint (skip already-processed prompts)
    done_hashes = load_checkpoint(CHECKPOINT_PATH)
    print(f"Already processed: {len(done_hashes)} (from checkpoint)")

    checkpoint_f = open(CHECKPOINT_PATH, "a")
    written = len(done_hashes)

    try:
        for ex in tqdm(all_examples, desc="Synthesizing CoT"):
            h = prompt_hash(ex["prompt"])
            if h in done_hashes:
                continue

            if args.skip_synthesis:
                reflection = (
                    f"This prompt is labeled as {ex['label']}. "
                    "[Placeholder reflection — run without --skip-synthesis to generate real reflections.]"
                )
            else:
                reflection = synthesize_reflection(client, ex["prompt"], ex["label"])

            completion = make_full_completion(reflection, ex["label"], ex["categories"])
            instruction = LLAMAGUARD_COT_INSTRUCTION.format(prompt=ex["prompt"])

            record = {
                "_hash": h,
                "source": ex["source"],
                "label": ex["label"],
                # SFTTrainer-compatible "messages" format
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ],
            }
            line = json.dumps(record, ensure_ascii=False)
            checkpoint_f.write(line + "\n")
            checkpoint_f.flush()
            done_hashes.add(h)
            written += 1

    finally:
        checkpoint_f.close()

    # Consolidate checkpoint -> final output
    with open(CHECKPOINT_PATH) as src, open(OUTPUT_PATH, "w") as dst:
        seen = set()
        for line in src:
            try:
                obj = json.loads(line)
                h = obj.pop("_hash", None)
                if h and h not in seen:
                    seen.add(h)
                    dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception:
                pass

    print(f"\nDone. {written} examples written to {OUTPUT_PATH}")

    # Print label distribution
    labels = {"harmful": 0, "unharmful": 0}
    sources: dict[str, int] = {}
    with open(OUTPUT_PATH) as f:
        for line in f:
            obj = json.loads(line)
            labels[obj.get("label", "unknown")] = labels.get(obj.get("label", "unknown"), 0) + 1
            src = obj.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
    print("\nLabel distribution:")
    for k, v in labels.items():
        print(f"  {k}: {v}")
    print("\nSource distribution:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
