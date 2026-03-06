def parse_label(raw_output: str) -> str:
    """
    Parse Llama-Guard-3-8B output into a binary label.

    Args:
        raw_output: Raw model output, e.g. "safe", "unsafe", "unsafe\nS1".

    Returns:
        "harmful" if the output indicates unsafe content, "unharmful" otherwise.
    """
    if raw_output.lower().startswith("unsafe"):
        return "harmful"
    return "unharmful"


def parse_violated_categories(raw_output: str) -> list[str]:
    """
    Extract violated safety category codes from Llama-Guard output.

    Args:
        raw_output: Raw model output, e.g. "unsafe\nS1\nS4".

    Returns:
        List of category codes, e.g. ["S1", "S4"]. Empty if safe.
    """
    lines = raw_output.strip().splitlines()
    if not lines or not lines[0].lower().startswith("unsafe"):
        return []
    return [line.strip() for line in lines[1:] if line.strip()]
