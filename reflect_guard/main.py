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


def prompt_credentials(skip_synthesis: bool) -> tuple[str, str | None]:
    print("=" * 60)
    print("Reflect-Guard Training Pipeline")
    print("=" * 60)

    hf_token = getpass.getpass("Enter your HuggingFace token (HF_TOKEN): ").strip()
    if not hf_token:
        print("ERROR: HuggingFace token is required.")
        sys.exit(1)

    openai_key = None
    if not skip_synthesis:
        openai_key = getpass.getpass("Enter your OpenAI API key: ").strip()
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
    args = parser.parse_args()

    # Allow CLI flag to override config (None means not specified, use config default)
    if args.skip_synthesis is not None:
        config.SKIP_SYNTHESIS = args.skip_synthesis

    skip_synthesis = config.SKIP_SYNTHESIS
    hf_token, openai_key = prompt_credentials(skip_synthesis)

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

    # Step 6: Smoke test
    if not args.skip_smoke_test:
        print("\n[Step 6] Running smoke test...")
        import smoke_test
        smoke_test.run(model, tokenizer)
    else:
        print("[Step 6] Skipped (--skip-smoke-test flag set).")

    print("\nDone.")


if __name__ == "__main__":
    main()
