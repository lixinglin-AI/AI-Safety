"""
Generate ablation training data:
  - Ablation B: labels only (no reflection)
  - Ablation C: blind teacher reflections (no ground-truth label given to GPT-4o-mini)

Requires the original training JSONL (reflect_guard_train.jsonl) as source.
Ablation C requires OPENAI_API_KEY.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

from config import (
    DATASET_PATH,
    ABLATION_B_DATASET_PATH,
    ABLATION_C_DATASET_PATH,
    LLAMAGUARD_COT_INSTRUCTION,
    LLAMAGUARD_STANDARD_INSTRUCTION,
    OPENAI_MODEL,
    SYNTHESIS_SYSTEM_BLIND,
)


def load_original_data() -> list[dict]:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} original training examples from {DATASET_PATH}")
    return records


def extract_prompt_from_message(user_content: str) -> str:
    """Extract the raw user prompt from the formatted instruction."""
    marker_start = "<BEGIN CONVERSATION>\nUser: "
    marker_end = "\n\n<END CONVERSATION>"
    start = user_content.find(marker_start)
    end = user_content.find(marker_end)
    if start == -1 or end == -1:
        return user_content
    return user_content[start + len(marker_start):end]


def make_label_only_completion(label: str, raw_assistant: str) -> str:
    """Create completion with just verdict + categories (no reflection)."""
    verdict = "unsafe" if label == "harmful" else "safe"
    # Extract categories from original assistant content
    lines = raw_assistant.strip().split("\n")
    categories = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("S") and "," in stripped or (stripped.startswith("S") and len(stripped) < 30):
            categories = stripped
            break
    result = [verdict]
    if verdict == "unsafe" and categories:
        result.append(categories)
    return "\n".join(result)


def synthesize_blind_reflection(client, prompt: str, retries: int = 3) -> str:
    """Generate reflection WITHOUT ground-truth label."""
    user_msg = (
        f"User message: {prompt[:1500]}\n\n"
        "Analyze this message and write the reflection:"
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_BLIND},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"[synthesis error: {e}]"
            time.sleep(2 ** attempt)


def generate_ablation_b(records: list[dict]) -> None:
    """Ablation B: SFT with labels only (no reflection).
    Uses LLAMAGUARD_STANDARD_INSTRUCTION (no reflection prompt)."""
    output_records = []
    for rec in tqdm(records, desc="Ablation B (labels only)"):
        prompt = extract_prompt_from_message(rec["messages"][0]["content"])
        label = rec["label"]
        assistant_content = rec["messages"][1]["content"]
        completion = make_label_only_completion(label, assistant_content)

        output_records.append({
            "source": rec["source"],
            "label": label,
            "messages": [
                {"role": "user",      "content": LLAMAGUARD_STANDARD_INSTRUCTION.format(prompt=prompt)},
                {"role": "assistant", "content": completion},
            ],
        })

    with open(ABLATION_B_DATASET_PATH, "w") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Ablation B: {len(output_records)} examples saved to {ABLATION_B_DATASET_PATH}")


def generate_ablation_c(records: list[dict], openai_key: str) -> None:
    """Ablation C: Blind teacher reflections (GPT-4o-mini without ground-truth label).
    Uses LLAMAGUARD_COT_INSTRUCTION (with reflection prompt)."""
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    output_records = []
    for rec in tqdm(records, desc="Ablation C (blind reflections)"):
        prompt = extract_prompt_from_message(rec["messages"][0]["content"])
        label = rec["label"]
        assistant_content = rec["messages"][1]["content"]

        # Generate blind reflection
        reflection = synthesize_blind_reflection(client, prompt)

        # Use ground-truth label for verdict (only reflection is blind)
        verdict = "unsafe" if label == "harmful" else "safe"
        # Extract categories from original
        lines = assistant_content.strip().split("\n")
        categories = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("S") and ("," in stripped or len(stripped) < 30):
                categories = stripped
                break

        completion_parts = [f"<reflection>\n{reflection}\n</reflection>", verdict]
        if verdict == "unsafe" and categories:
            completion_parts.append(categories)
        completion = "\n".join(completion_parts)

        output_records.append({
            "source": rec["source"],
            "label": label,
            "messages": [
                {"role": "user",      "content": LLAMAGUARD_COT_INSTRUCTION.format(prompt=prompt)},
                {"role": "assistant", "content": completion},
            ],
        })

    with open(ABLATION_C_DATASET_PATH, "w") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Ablation C: {len(output_records)} examples saved to {ABLATION_C_DATASET_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ablation training data")
    parser.add_argument("--ablation", choices=["b", "c", "both"], default="both",
                        help="Which ablation data to generate")
    args = parser.parse_args()

    records = load_original_data()

    if args.ablation in ("b", "both"):
        generate_ablation_b(records)

    if args.ablation in ("c", "both"):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("ERROR: OPENAI_API_KEY required for ablation C", file=sys.stderr)
            sys.exit(1)
        generate_ablation_c(records, openai_key)


if __name__ == "__main__":
    main()
