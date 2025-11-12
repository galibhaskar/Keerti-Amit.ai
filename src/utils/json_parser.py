"""JSON parsing utilities."""

import json
import re
from typing import Dict, Any


def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Remove common LLM formatting (like ```json) and parse JSON.

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON dictionary
    """
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*```', '', text).strip()
    return json.loads(text)

