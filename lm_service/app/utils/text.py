import json
import re


def _clean(raw_text: str) -> str:
    cleaned = re.sub(r"<\|[a-zA-Z0-9_]+\|>", "", raw_text)
    return re.sub(r"```(?:json)?", "", cleaned)


def extract_json_from_text(raw_text: str):
    cleaned_text = _clean(raw_text)
    match = re.search(r'\[.*\]', cleaned_text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found in the text")

    return json.loads(match.group(0))


def extract_json_object(raw_text: str) -> dict:
    """Extracts the first JSON object ({...}) from the model response."""
    cleaned_text = _clean(raw_text)
    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the text")

    return json.loads(cleaned_text[start:end + 1])
