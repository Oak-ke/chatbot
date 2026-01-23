import re
from langchain_core.messages import HumanMessage

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

def detect_language(text: str) -> str:
    return "ar" if ARABIC_RE.search(text) else "en"

def translate_text(text: str, llm, target_lang: str | None = None,) -> tuple[str, str]:
    """
    Returns (translated_text, source_language)
    If target_lang is None, it auto-flips en <-> ar.
    """

    source_lang = detect_language(text)

    if target_lang is None:
        target_lang = "English" if source_lang == "ar" else "Arabic"

    if (
        source_lang == "en" and target_lang.lower() == "english"
    ) or (
        source_lang == "ar" and target_lang.lower() == "arabic"
    ):
        return text, source_lang  # no-op

    prompt = (
        f"Translate the following text to {target_lang}.\n"
        "Return ONLY the translation.\n\n"
        f"Text:\n{text}"
    )

    translated = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    return translated, source_lang