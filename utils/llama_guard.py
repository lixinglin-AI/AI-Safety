import re


def _strip_reflection(text: str) -> str:
    """Remove <reflection>...</reflection> block if present, returning the verdict portion."""
    if "<reflection>" in text:
        text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_label(raw_output: str) -> str:
    """
    Parse Llama-Guard-3-8B output into a binary label.

    Handles both standard output and Reflect-Guard CoT format:
      Standard:       "safe" / "unsafe\nS1"
      Reflect-Guard:  "<reflection>...</reflection>\nsafe"

    Returns:
        "harmful" if the output indicates unsafe content, "unharmful" otherwise.
    """
    verdict = _strip_reflection(raw_output)
    if verdict.lower().startswith("unsafe"):
        return "harmful"
    return "unharmful"


def parse_reflection(raw_output: str) -> str | None:
    """
    Extract the reflection text from a Reflect-Guard CoT output.

    Returns:
        The reflection string, or None if no reflection block is present.
    """
    match = re.search(r"<reflection>(.*?)</reflection>", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_violated_categories(raw_output: str) -> list[str]:
    """
    Extract violated safety category codes from Llama-Guard output.

    Handles both standard and Reflect-Guard CoT format.

    Returns:
        List of category codes, e.g. ["S1", "S4"]. Empty if safe.
    """
    verdict = _strip_reflection(raw_output)
    lines = verdict.splitlines()
    if not lines or not lines[0].lower().startswith("unsafe"):
        return []
    return [line.strip() for line in lines[1:] if line.strip()]
