"""
Shared configuration and constants for Reflect-Guard training pipeline.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = "/vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse/reflect_guard"
DATASET_PATH    = os.path.join(BASE_DIR, "reflect_guard_train.jsonl")
OUTPUT_DIR      = os.path.join(BASE_DIR, "checkpoints")
ADAPTER_SAVE_PATH = os.path.join(BASE_DIR, "reflect_guard_lora")

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-Guard-3-8B"

# ── Dataset synthesis ──────────────────────────────────────────────────────────
SKIP_SYNTHESIS  = False      # set False to call GPT-4o-mini for real reflections
MAX_PER_SOURCE  = 500        # examples per source dataset
OPENAI_MODEL    = "gpt-4o-mini"
RANDOM_SEED     = 42

# ── Training hyperparameters ───────────────────────────────────────────────────
NUM_EPOCHS                  = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_STEPS                = 5
LEARNING_RATE               = 2e-4

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT        = 0.05

# ── Prompts ────────────────────────────────────────────────────────────────────
SAFETY_CATEGORIES = (
    "S1: Violent Crimes.\nS2: Non-Violent Crimes.\nS3: Sex-Related Crimes.\n"
    "S4: Child Sexual Exploitation.\nS5: Defamation.\nS6: Specialized Advice.\n"
    "S7: Privacy.\nS8: Intellectual Property.\nS9: Indiscriminate Weapons.\n"
    "S10: Hate.\nS11: Suicide & Self-Harm.\nS12: Sexual Content.\nS13: Elections."
)

LLAMAGUARD_COT_INSTRUCTION = (
    "Task: Check if there is unsafe content in 'User' messages in conversations "
    "according our safety policy with the below categories.\n\n"
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n" + SAFETY_CATEGORIES + "\n"
    "<END UNSAFE CONTENT CATEGORIES>\n\n"
    "<BEGIN CONVERSATION>\nUser: {prompt}\n\n<END CONVERSATION>\n\n"
    "Provide your safety assessment for ONLY THE LAST User message in the above conversation:\n"
    " - First provide a logical self-reflection in <reflection>...</reflection> tags, "
    "analyzing whether the message uses adversarial techniques (roleplay bypass, fictional "
    "framing, indirect harmful requests, obfuscation, social engineering) and identifying "
    "key harm indicators.\n"
    " - After the reflection, the first line must read 'safe' or 'unsafe'.\n"
    " - If unsafe, a second line must include a comma-separated list of violated categories."
)

SYNTHESIS_SYSTEM = (
    "You are a safety analysis expert building training data for an AI safety classifier. "
    "For a given user message and its ground-truth safety label, write a concise logical "
    "self-reflection (2-4 sentences) that:\n"
    "1. Identifies whether the message uses adversarial techniques to bypass safety filters "
    "(e.g. roleplay framing, fictional scenarios, indirect harmful requests, technical "
    "obfuscation, authority appeals, social engineering).\n"
    "2. States whether the underlying intent is harmful or benign and why.\n"
    "3. Lists the key indicators that support the classification.\n\n"
    "Rules:\n"
    "- Be precise and analytical. Do NOT moralize or preach.\n"
    "- For harmful prompts: explain the harm category and any adversarial technique used.\n"
    "- For benign prompts: confirm safety and explain why.\n"
    "- Output ONLY the reflection text (2-4 sentences). No preamble, no verdict."
)
