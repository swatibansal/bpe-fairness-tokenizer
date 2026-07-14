#!/usr/bin/env python3
"""Fetch the India Wikipedia pages and convert them to a faithful Markdown corpus.

Faithful means visible article content — links, tables, references, image links,
navboxes, categories — is preserved wherever the HTML-to-Markdown converter emits it;
only script/style/meta/link machinery is stripped. This mirrors the sibling
`bpe-tokenizer-markdown` corpus, but for this repo's four languages (Bengali retained).

This is the ONLY part of the project that uses third-party libraries; they build the
corpus and are listed in requirements-corpus.txt. The tokenizer itself
(bpe.py / metrics.py / train.py) stays standard-library only.

    pip install -r requirements-corpus.txt
    python3 fetch_faithful.py     # writes corpus/<lang>.faithful.{md,txt}
"""
from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import quote, urljoin

from metrics import faithful_units

# requests / bs4 / markdownify are imported lazily inside the functions that fetch and
# convert, so importing this module's constants (PAGES, CORPUS_DIR) never requires the
# third-party corpus libraries to be installed.

CORPUS_DIR = "corpus"
USER_AGENT = "ai-tokenizer faithful-markdown/1.0 (educational)"

# (language name, Wikipedia article title). Same four languages as the tokenizer.
PAGES = {
    "en": ("English", "India"),
    "hi": ("Hindi", "भारत"),
    "te": ("Telugu", "భారతదేశం"),
    "bn": ("Bengali", "ভারত"),
}


def get(url):
    import requests
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))


def absolutize_links(soup, lang):
    """Rewrite relative/protocol-relative hrefs and srcs to absolute URLs."""
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_only_technical_noise(node, soup):
    """Remove script/style/meta and link machinery, keeping category links as text."""
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown):
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def build_one(lang, title):
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    os.makedirs(CORPUS_DIR, exist_ok=True)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    res = get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")
    body = soup.find("body") or soup
    strip_only_technical_noise(body, soup)
    absolutize_links(body, lang)
    markdown = normalize_markdown(
        md(str(body), heading_style="ATX", bullets="-", strip=["span"])
    )

    md_path = os.path.join(CORPUS_DIR, f"{lang}.faithful.md")
    txt_path = os.path.join(CORPUS_DIR, f"{lang}.faithful.txt")
    meta_path = os.path.join(CORPUS_DIR, f"{lang}.meta.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    meta = {
        "lang": lang,
        "title": title,
        "source_url": url,
        "variant": "wiki_faithful_markdown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chars": len(markdown),
        "faithful_units": faithful_units(markdown),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    for code, (name, title) in PAGES.items():
        meta = build_one(code, title)
        print(f"{code} {name}: {meta['faithful_units']} faithful units")
        time.sleep(1.0)  # be polite to the Wikipedia API between requests
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
