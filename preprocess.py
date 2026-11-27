"""Shared text cleaning. Used by BOTH training and serving so they match.

Emoji are removed using Python's built-in Unicode database (unicodedata), not a hand-written list:
  So  other symbols (emoji, hearts, stars, flags, pictographs)
  Cf  format characters (zero-width joiner, tag characters)
  Me  enclosing marks (keycap)
  Cn  unassigned code points (emoji newer than this Python's Unicode version)
  Co/Cs  private use / surrogates
plus skin-tone modifiers and variation selectors, found by their Unicode names.
"""
import re
import unicodedata

_DROP_CATEGORIES = {"So", "Cf", "Me", "Cn", "Co", "Cs"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{3,}")  # "sooooo" -> "sooo"


def _is_emoji_like(ch: str) -> bool:
    cat = unicodedata.category(ch)
    if cat in _DROP_CATEGORIES:
        return True
    if cat == "Sk":   # only skin-tone modifiers; plain accents like ` and ^ stay
        return "EMOJI MODIFIER" in unicodedata.name(ch, "")
    if cat == "Mn":   # only variation selectors; real combining marks stay
        return unicodedata.name(ch, "").startswith("VARIATION SELECTOR")
    return False


def non_latin_ratio(text: str) -> float:
    """Share of letters outside Latin script (Devanagari, Arabic, CJK ...). Used to skip non-English comments."""
    letters = [c for c in text if c.isalpha()]
    return sum(ord(c) > 591 for c in letters) / len(letters) if letters else 0.0


def strip_emoji(text) -> str:
    """Remove emoji and emoji-like symbols. Safe to use for text shown to people."""
    if text is None:
        return ""
    out = [" " if _is_emoji_like(ch) else ch for ch in str(text)]
    return _WS.sub(" ", "".join(out)).strip()


def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _TIMESTAMP.sub(" ", t)
    t = strip_emoji(t)
    t = _REPEAT.sub(r"\1\1\1", t)
    t = _WS.sub(" ", t).strip()
    return t
