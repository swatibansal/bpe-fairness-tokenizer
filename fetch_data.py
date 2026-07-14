"""Fetch and cache the Wikipedia 'India' article in four languages."""

import os
import re
import json
import time
import urllib.request
import urllib.parse

DATA_DIR = "data"

# Each language lists article titles, fetched in order and concatenated until
# WORD_LIMIT words are reached (missing/short titles are skipped). Native "India"
# articles vary widely in length, so each language is topped up with the same
# roster of large India-related articles to keep the four corpora comparably sized.
LANGUAGES = {
    "en": {"name": "English", "titles": ["India", "History of India", "Mahatma Gandhi"]},
    "hi": {"name": "Hindi", "titles": ["भारत", "महात्मा गांधी", "भारत का इतिहास"]},
    "te": {"name": "Telugu", "titles": ["భారత దేశం", "భారతదేశ చరిత్ర", "మహాత్మా గాంధీ",
                                        "భారత జాతీయ కాంగ్రెస్"]},
    "bn": {"name": "Bengali", "titles": ["ভারত", "ভারতের ইতিহাস", "মহাত্মা গান্ধী"]},
}

# ~5,000 words/language. Larger corpora were tested but hurt the result: with a
# fixed 10k shared vocab, English needs far more merges to keep its fertility <= 1.2
# on a bigger corpus, starving the other three languages (see the design doc).
WORD_LIMIT = 5000
FETCH_DELAY = 1.0  # seconds between requests; Wikipedia rate-limits rapid bursts


def clean_text(raw):
    raw = re.sub(r"\[\d+\]", "", raw)          # drop [1]-style reference marks
    raw = re.sub(r"\s+", " ", raw)             # collapse whitespace
    return raw.strip()


def take_words(text, limit=WORD_LIMIT):
    return " ".join(text.split()[:limit])


def fetch_page(lang, title):
    """Fetch the plain-text extract of a Wikipedia article."""
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "format": "json", "titles": title, "redirects": "1",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "bpe-tokenizer/1.0 (educational)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data["query"]["pages"]
    # A missing page has no "extract"; return None so callers can skip it.
    return next(iter(pages.values())).get("extract")


def accumulate_extracts(titles, fetch, limit=WORD_LIMIT):
    """Fetch titles in order, skip empties, concatenate, and cap at `limit` words.

    `fetch(title)` returns the raw extract or a falsy value for a missing article.
    Stops early once `limit` words are reached so we never fetch more than needed.
    """
    pieces = []
    for title in titles:
        raw = fetch(title)
        if not raw:
            continue
        pieces.append(clean_text(raw))
        if len(" ".join(pieces).split()) >= limit:
            break
    return take_words(" ".join(pieces), limit)


def load_corpus(lang, refresh=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{lang}.txt")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def fetch(title):
        page = fetch_page(lang, title)
        time.sleep(FETCH_DELAY)
        return page

    text = accumulate_extracts(LANGUAGES[lang]["titles"], fetch)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    for lang in LANGUAGES:
        text = load_corpus(lang, refresh=True)
        print(f"{lang}: {len(text.split())} words cached")
