import requests
from config import OLLAMA_BASE_URL, MODEL_NAME


def classify(prompt: str, response: str | None = None, timeout: int = 300) -> str:
    """
    Send a prompt (and optional model response) to Llama-Guard-3-8B via Ollama.

    Args:
        prompt: The user prompt to classify.
        response: Optional assistant response to include for response-level classification.
        timeout: Request timeout in seconds.

    Returns:
        Raw model output string, e.g. "safe" or "unsafe\nS1".
    """
    messages = [{"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()
