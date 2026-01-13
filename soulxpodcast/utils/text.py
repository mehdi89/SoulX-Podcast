import re
from typing import List


def normalize_text(current_text):
    """
    Normalize text by ensuring proper punctuation at the end.

    For English text, adds a period if no punctuation exists.
    """
    current_text = current_text.strip()

    if not current_text:
        return current_text

    # Check if the text ends with an English character
    if re.search(r'[a-zA-Z]$', current_text):
        # If the last character is not a punctuation mark, add a period
        if current_text[-1] not in ".!?":
            current_text += "."

    return current_text


def check_monologue_text(text: str, prefix: str = None) -> bool:
    """Check if monologue text is valid."""
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialogue_text(text_list: List[str]) -> bool:
    """Check if dialogue text list is valid."""
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True
