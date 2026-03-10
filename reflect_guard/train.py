"""
Step 2-5: Load model, configure LoRA, train, and save adapters.
"""

import gc
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
    MODEL_ID, DATASET_PATH, OUTPUT_DIR, ADAPTER_SAVE_PATH,
    NUM_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS, LEARNING_RATE,
    LORA_R, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT,
)


def load_dataset_from_disk() -> list:
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"Total training examples: {len(records)}")
    print("Label distribution:", label_counts)
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


def format_dataset(records: list, tokenizer) -> Dataset:
    def format_example(record: dict) -> dict:
        # Llama Guard 3's chat template drops assistant content when passed as a turn.
        # Fix: apply template for user turn only (with generation prompt), then
        # manually append the assistant content + end-of-turn token.
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
    print(f"Training examples: {len(train_dataset)}")
    return train_dataset


def run(hf_token: str):
    # Load data
    records = load_dataset_from_disk()

    # Load model
    model, tokenizer = load_model_and_tokenizer(hf_token)
    model = apply_lora(model)

    # Format dataset
    train_dataset = format_dataset(records, tokenizer)

    # Free memory before training
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Train
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=5,
        save_steps=200,
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

    print(f"Training on {len(train_dataset)} examples x {NUM_EPOCHS} epoch(s)...")
    trainer.train()

    # Save adapters
    os.makedirs(ADAPTER_SAVE_PATH, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
    print(f"LoRA adapters saved to {ADAPTER_SAVE_PATH}")

    # Reset model to inference mode
    trainer.model.config.use_cache = True
    trainer.model.gradient_checkpointing_disable()
    trainer.model.eval()

    return trainer.model, tokenizer
