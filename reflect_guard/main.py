"""
Entry point for Reflect-Guard training pipeline on HPC.

Usage:
    python main.py
    python main.py --skip-smoke-test
    python main.py --skip-synthesis    # override config.SKIP_SYNTHESIS
"""

import argparse
import getpass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from huggingface_hub import login

import config


def prompt_credentials(skip_synthesis: bool, hf_token_arg: str = None, openai_key_arg: str = None) -> tuple[str, str | None]:
    print("=" * 60)
    print("Reflect-Guard Training Pipeline")
    print("=" * 60)

    hf_token = hf_token_arg or os.environ.get("HF_TOKEN") or getpass.getpass("Enter your HuggingFace token (HF_TOKEN): ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token is required.")
        sys.exit(1)

    openai_key = None
    if not skip_synthesis:
        openai_key = openai_key_arg or os.environ.get("OPENAI_API_KEY") or getpass.getpass("Enter your OpenAI API key: ").strip()
        if not openai_key:
            print("ERROR: OpenAI API key is required when SKIP_SYNTHESIS=False.")
            sys.exit(1)
    else:
        print("SKIP_SYNTHESIS=True: OpenAI key not needed.")

    return hf_token, openai_key


def main():
    parser = argparse.ArgumentParser(description="Reflect-Guard training pipeline")
    parser.add_argument("--skip-smoke-test",  action="store_true", help="Skip smoke test after training")
    parser.add_argument("--skip-synthesis",   action=argparse.BooleanOptionalAction, default=None, help="Override config SKIP_SYNTHESIS (--skip-synthesis / --no-skip-synthesis)")
    parser.add_argument("--synthesis-only",   action="store_true", help="Only run data synthesis, then exit")
    parser.add_argument("--train-only",       action="store_true", help="Skip synthesis, only train (dataset must exist)")
    parser.add_argument("--hf-token",        default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--openai-key",      default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # Allow CLI flag to override config (None means not specified, use config default)
    if args.skip_synthesis is not None:
        config.SKIP_SYNTHESIS = args.skip_synthesis

    skip_synthesis = config.SKIP_SYNTHESIS
    hf_token, openai_key = prompt_credentials(skip_synthesis, args.hf_token, args.openai_key)

    # Login to HuggingFace
    login(token=hf_token)
    print("Logged in to HuggingFace.\n")

    # Step 1: Synthesize data
    if not args.train_only:
        print("[Step 1] Synthesizing training data...")
        import synthesize
        synthesize.run(hf_token=hf_token, openai_key=openai_key)
    else:
        print("[Step 1] Skipped (--train-only flag set).")

    if args.synthesis_only:
        print("Synthesis complete. Exiting (--synthesis-only flag set).")
        return

    # Step 2-5: Train
    print("\n[Step 2-5] Loading model and training...")
    import train
    model, tokenizer = train.run(hf_token=hf_token)

    # Step 6: Smoke test (in-memory model)
    if not args.skip_smoke_test:
        print("\n[Step 6a] Running smoke test (in-memory model)...")
        import smoke_test
        smoke_test.run(model, tokenizer)

        # Step 6b: Verify the saved adapter loads correctly from disk
        print("\n[Step 6b] Verifying saved adapter loads correctly from disk...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        del model
        torch.cuda.empty_cache()
        loaded_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, token=hf_token)
        loaded_base = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
        )
        loaded_model = PeftModel.from_pretrained(loaded_base, config.ADAPTER_SAVE_PATH)
        loaded_model.eval()
        loaded_model.config.use_cache = True
        if hasattr(loaded_model, "base_model"):
            loaded_model.base_model.config.use_cache = True
        print("Saved adapter loaded. Running smoke test on loaded model...")
        smoke_test.run(loaded_model, loaded_tokenizer)
    else:
        print("[Step 6] Skipped (--skip-smoke-test flag set).")

    print("\nDone.")


if __name__ == "__main__":
    main()
