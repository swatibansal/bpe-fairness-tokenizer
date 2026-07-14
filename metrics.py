"""Per-language and aggregate metrics for the BPE tokenizer.

Two metric families live here. The word-based family (`count_words`,
`language_metrics`, `aggregate_metrics`) measures fertility as tokens/word and drives
the default optimizer and the widget. The faithful family (`faithful_units`,
`faithful_language_metrics`, `faithful_aggregate`) measures fertility against faithful
units and reproduces the `bpe-tokenizer-markdown` scoring (spread score + Hindi
penalty) so the two engines can be compared on identical rules.
"""

import math
import re
import unicodedata


def count_words(text):
    return len(re.findall(r"\S+", text))


def faithful_units(text):
    """Count faithful units in `text`.

    Each maximal run of Unicode letter/mark/number characters counts as one unit, and
    each other visible (non-whitespace) character — punctuation or symbol — counts as
    its own unit. This is the stdlib reproduction of the sibling project's regex
    ``[\\p{L}\\p{M}\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]``.
    """
    count, in_run = 0, False
    for ch in text:
        if ch.isspace():
            in_run = False
        elif unicodedata.category(ch)[0] in ("L", "M", "N"):
            if not in_run:
                count += 1
                in_run = True
        else:
            count += 1
            in_run = False
    return count


def faithful_language_metrics(tokenizer, text, vocab_used=None):
    """Faithful-unit fertility for one language: X = tokens / faithful_units."""
    units = faithful_units(text)
    counter = getattr(tokenizer, "count_tokens", None)  # fast dedup path when available
    tokens = counter(text) if counter else len(tokenizer.encode(text))
    X = tokens / units if units else 0.0
    return {
        "units": units,
        "tokens": tokens,
        "X": round(X, 6),
        "within_1_2": X <= 1.2,
        "vocab_used": vocab_used,
    }


def faithful_aggregate(per_lang, hindi_key="hi"):
    """Spread score with the Hindi penalty, matching bpe-tokenizer-markdown.

    score = 1000 / (X_max - X_min); hindi_penalty = exp(max(0, hi/1.2 - 1)); the
    Hindi-adjusted score divides the two. `build_score` mirrors the raw score so callers
    that key on it keep working.
    """
    xs = [m["X"] for m in per_lang.values()]
    x_max, x_min = max(xs), min(xs)
    spread = x_max - x_min
    score = float("inf") if spread == 0 else 1000 / spread
    hi = per_lang.get(hindi_key, {}).get("X", 0.0)
    penalty = math.exp(max(0.0, hi / 1.2 - 1.0))
    adjusted = score / penalty if penalty else score
    return {
        "X_max": x_max,
        "X_min": x_min,
        "spread": round(spread, 6),
        "sorted_X": sorted(xs, reverse=True),
        "score": score,
        "build_score": score,
        "hindi_penalty": penalty,
        "hindi_adjusted_score": adjusted,
    }


def language_metrics(tokenizer, text, vocab_used=None):
    words = count_words(text)
    tokens = len(tokenizer.encode(text))
    X = tokens / words if words else 0.0
    return {
        "words": words,
        "tokens": tokens,
        "X": round(X, 6),
        "within_1_2": X <= 1.2,
        "vocab_used": vocab_used,
    }


def aggregate_metrics(per_lang):
    xs = [m["X"] for m in per_lang.values()]
    x_max, x_min = max(xs), min(xs)
    spread = x_max - x_min
    build_score = float("inf") if spread == 0 else 1000 / spread
    return {
        "X_max": x_max,
        "X_min": x_min,
        "spread": round(spread, 6),
        "sorted_X": sorted(xs, reverse=True),
        "build_score": build_score,
    }
