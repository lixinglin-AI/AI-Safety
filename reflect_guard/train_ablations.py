"""
Train ablation LoRA adapters:
  - Ablation B: SFT with labels only (no reflection)
  - Ablation C: SFT with blind teacher reflections

Uses the same hyperparameters and LoRA config as the main Reflect-Guard training.
"""

import gc
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import (
    MODEL_ID,
    ABLATION_B_DATASET_PATH, ABLATION_C_DATASET_PATH,
    ABLATION_B_ADAPTER_PATH, ABLATION_C_ADAPTER_PATH,
    ABLATION_B_CKPT_DIR, ABLATION_C_CKPT_DIR,
    NUM_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS, LEARNING_RATE,
    LORA_R, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT,
)


def load_dataset_from_disk(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print(f"Loaded {len(records)} examples from {path}")
    print(f"  Label distribution: {label_counts}")
    return records


def load_model_and_tokenizer(hf_token: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    print("Model loaded.")
    return model, tokenizer


def apply_lora(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_dataset(records: list[dict], tokenizer) -> Dataset:
    def format_example(record: dict) -> dict:
        user_text = tokenizer.apply_chat_template(
            [record["messages"][0]],
            tokenize=False,
            add_generation_prompt=True,
        )
        assistant_content = record["messages"][1]["content"]
        text = user_text + assistant_content + "<|eot_id|>"
        return {"text": text}

    raw_dataset = Dataset.from_list(records)
    train_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    print(f"  Formatted {len(train_dataset)} training examples")
    return train_dataset


def seeded_path(base_path: str, seed: int | None) -> str:
    """Suffix an adapter/checkpoint path with _seed{N}, unless seed is the default
    (42, matching config.RANDOM_SEED) or unset — the existing unsuffixed adapter
    from the original run already IS the seed=42 data point, so it's reused as-is
    rather than retrained, matching reflect_guard/train_multiseed_job.sh's design."""
    if seed is None or seed == 42:
        return base_path
    return f"{base_path}_seed{seed}"


def train_adapter(records, tokenizer, model, output_dir, adapter_path):
    train_dataset = format_dataset(records, tokenizer)

    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        dataset_text_field="text",
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )

    # scavenge_gpu is preemptible and jobs get killed mid-training in practice
    # (observed: several runs PREEMPTED after 15-55 minutes) — resume from the
    # latest checkpoint in output_dir if one exists (save_steps=50 above), so a
    # requeued job continues instead of restarting from step 0.
    resume = bool(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if resume:
        print(f"Found existing checkpoint(s) in {output_dir}, resuming training from the latest one.")
    print(f"Training on {len(train_dataset)} examples x {NUM_EPOCHS} epochs...")
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(adapter_path, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    return trainer


def main():
    import argparse
    import getpass
    from huggingface_hub import login

    parser = argparse.ArgumentParser(description="Train ablation LoRA adapters")
    parser.add_argument("--ablation", choices=["b", "c", "both"], default="both",
                        help="Which ablation to train")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for weight init and data-loader shuffling "
                              "(transformers.set_seed). 42 is the original run's implicit "
                              "seed and writes to the default (unsuffixed) adapter path; any "
                              "other value writes to {adapter_path}_seed{N} instead, so "
                              "additional seeds for statistical-significance testing "
                              "(reflect_guard/compute_seed_variance.py) don't overwrite it.")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or getpass.getpass("HF token: ").strip()
    login(token=hf_token)

    if args.ablation in ("b", "both"):
        print("\n" + "=" * 60)
        print(f"TRAINING ABLATION B: SFT Labels Only (no reflection), seed={args.seed}")
        print("=" * 60)
        records_b = load_dataset_from_disk(ABLATION_B_DATASET_PATH)
        model, tokenizer = load_model_and_tokenizer(hf_token)
        model = apply_lora(model)
        adapter_path = seeded_path(ABLATION_B_ADAPTER_PATH, args.seed)
        ckpt_dir = seeded_path(ABLATION_B_CKPT_DIR, args.seed)
        train_adapter(records_b, tokenizer, model, ckpt_dir, adapter_path)
        # Free GPU memory before next training
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.ablation in ("c", "both"):
        print("\n" + "=" * 60)
        print(f"TRAINING ABLATION C: Blind Teacher Reflections, seed={args.seed}")
        print("=" * 60)
        records_c = load_dataset_from_disk(ABLATION_C_DATASET_PATH)
        model, tokenizer = load_model_and_tokenizer(hf_token)
        model = apply_lora(model)
        adapter_path = seeded_path(ABLATION_C_ADAPTER_PATH, args.seed)
        ckpt_dir = seeded_path(ABLATION_C_CKPT_DIR, args.seed)
        train_adapter(records_c, tokenizer, model, ckpt_dir, adapter_path)


if __name__ == "__main__":
    main()
