import re

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

def detect_language(text: str) -> str:
    return "ar" if ARABIC_RE.search(text) else "en"