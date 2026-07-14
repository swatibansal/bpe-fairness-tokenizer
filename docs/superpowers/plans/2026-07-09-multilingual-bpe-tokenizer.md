# Multilingual BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one shared 10,000-token from-scratch BPE tokenizer over the Wikipedia "India" articles in English, Hindi, Telugu, and Spanish, report per-language tokens/word ratios, maximize build score = 1000/(X_max − X_min), and emit a self-contained HTML widget with JSON export.

**Architecture:** Pure-Python pipeline: `fetch_data.py` caches ~5,000 words/language to `data/`; `bpe.py` is a from-scratch BPE (train/encode/decode/serialize); `train.py` allocates the 10k vocab across languages with an auto-adjust loop that shrinks X_max − X_min, writes `tokenizer.json`, and renders a static `report.html` widget via a string template. `metrics.py` computes ratios and build score.

**Tech Stack:** Python 3 (standard library only — `urllib`, `json`, `re`, `html`, `collections`), `pytest` for tests. No external tokenizer or ML libraries.

---

## File Structure

- `bpe.py` — from-scratch BPE: pre-tokenization, `train`, `encode`, `decode`, `to_dict`/`from_dict`.
- `metrics.py` — per-language and aggregate metrics + build score.
- `fetch_data.py` — Wikipedia fetch + clean + cache to `data/<lang>.txt`.
- `train.py` — orchestration: budget allocation, auto-adjust, write `tokenizer.json`, render `report.html`.
- `report_template.py` — HTML template string + render function.
- `tests/test_bpe.py`, `tests/test_metrics.py`, `tests/test_train.py` — unit tests.
- `requirements.txt`, `README.md`.

Test fixtures use small inline strings so tests never hit the network.

---

## Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `requirements.txt`**

```
pytest>=7.0
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create `tests/__init__.py`** (empty file)

- [ ] **Step 4: Create `README.md`**

```markdown
# Multilingual BPE Tokenizer

A from-scratch 10,000-token BPE tokenizer over the Wikipedia "India" article in
English, Hindi, Telugu, and Spanish.

## Usage

```bash
pip install -r requirements.txt
python fetch_data.py      # caches ~5000 words/language to data/
python train.py           # writes tokenizer.json and report.html
open report.html          # view ratios, stats, build score; export JSON
```

## Metrics

For each language, X = total tokens / total words. Build score = 1000 / (X_max − X_min).
```

- [ ] **Step 5: Verify pytest runs**

Run: `python -m pytest -q`
Expected: "no tests ran" (exit code 5) — confirms pytest is installed.

- [ ] **Step 6: Commit**

```bash
git init && git add -A && git commit -m "chore: project scaffold"
```

---

## Task 2: BPE pre-tokenization

**Files:**
- Create: `bpe.py`
- Test: `tests/test_bpe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bpe.py
from bpe import pre_tokenize

def test_pre_tokenize_splits_words_with_boundary_marker():
    # Each word is prefixed with a boundary marker so spacing is reconstructable.
    assert pre_tokenize("hola mundo") == [["▁", "h", "o", "l", "a"], ["▁", "m", "u", "n", "d", "o"]]

def test_pre_tokenize_handles_unicode_scripts():
    # Devanagari word stays intact as characters.
    assert pre_tokenize("भारत") == [["▁", "भ", "ा", "र", "त"]]

def test_pre_tokenize_empty_string():
    assert pre_tokenize("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: FAIL with "cannot import name 'pre_tokenize'".

- [ ] **Step 3: Write minimal implementation**

```python
# bpe.py
import re

BOUNDARY = "▁"

def pre_tokenize(text):
    """Split text into words; each word becomes a list of chars prefixed with BOUNDARY."""
    words = re.findall(r"\S+", text)
    return [[BOUNDARY] + list(word) for word in words]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add bpe.py tests/test_bpe.py && git commit -m "feat: BPE pre-tokenization"
```

---

## Task 3: BPE training (merge loop)

**Files:**
- Modify: `bpe.py`
- Test: `tests/test_bpe.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bpe.py
from bpe import train_merges

def test_train_merges_learns_most_frequent_pair():
    # "ab" appears 3 times across words; first merge must be ('a','b').
    corpus = ["ab", "ab", "ab", "ac"]
    merges = train_merges(corpus, num_merges=1)
    assert merges == [("▁a", "b")] or merges == [("a", "b")]

def test_train_merges_respects_budget():
    corpus = ["banana", "banana"]
    merges = train_merges(corpus, num_merges=3)
    assert len(merges) <= 3

def test_train_merges_zero_budget():
    assert train_merges(["abc"], num_merges=0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -k train_merges -v`
Expected: FAIL with "cannot import name 'train_merges'".

- [ ] **Step 3: Write minimal implementation**

```python
# add to bpe.py
from collections import Counter

def _get_pairs(word_tokens):
    return [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]

def _merge_word(word_tokens, pair):
    merged, i = [], 0
    while i < len(word_tokens):
        if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
            merged.append(word_tokens[i] + word_tokens[i + 1])
            i += 2
        else:
            merged.append(word_tokens[i])
            i += 1
    return merged

def train_merges(corpus, num_merges):
    """Learn up to num_merges BPE merges from a list of raw word strings."""
    words = [tok for text in corpus for tok in pre_tokenize(text)]
    merges = []
    for _ in range(num_merges):
        pair_counts = Counter()
        for w in words:
            for p in _get_pairs(w):
                pair_counts[p] += 1
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        words = [_merge_word(w, best) for w in words]
        merges.append(best)
    return merges
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bpe.py -k train_merges -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add bpe.py tests/test_bpe.py && git commit -m "feat: BPE merge training"
```

---

## Task 4: Tokenizer class (encode/decode/serialize)

**Files:**
- Modify: `bpe.py`
- Test: `tests/test_bpe.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bpe.py
from bpe import Tokenizer

def test_encode_decode_roundtrip():
    tok = Tokenizer()
    tok.train(["banana banana bandana"], num_merges=10)
    ids = tok.encode("banana")
    assert tok.decode(ids) == "banana"

def test_encode_returns_ids_within_vocab():
    tok = Tokenizer()
    tok.train(["hola mundo"], num_merges=5)
    ids = tok.encode("hola")
    assert all(0 <= i < len(tok.vocab) for i in ids)

def test_serialize_roundtrip():
    tok = Tokenizer()
    tok.train(["hola mundo hola"], num_merges=5)
    d = tok.to_dict()
    tok2 = Tokenizer.from_dict(d)
    assert tok2.encode("hola") == tok.encode("hola")

def test_unknown_char_uses_unk():
    tok = Tokenizer()
    tok.train(["abc"], num_merges=2)
    ids = tok.encode("z")  # 'z' never seen
    assert tok.unk_id in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -k "Tokenizer or roundtrip or unk" -v`
Expected: FAIL with "cannot import name 'Tokenizer'".

- [ ] **Step 3: Write minimal implementation**

```python
# add to bpe.py
SPECIAL_TOKENS = ["<pad>", "<unk>"]

class Tokenizer:
    def __init__(self):
        self.merges = []          # list of (str, str)
        self.vocab = {}           # token -> id
        self.id_to_token = {}     # id -> token
        self.merge_ranks = {}     # (a,b) -> rank
        self._build_specials()

    def _build_specials(self):
        for t in SPECIAL_TOKENS:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        self.unk_id = self.vocab["<unk>"]

    def _rebuild_index(self):
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}

    def train(self, corpus, num_merges, seed_chars=None):
        # Seed base characters (all languages share this set).
        chars = set(seed_chars or [])
        for text in corpus:
            for word in pre_tokenize(text):
                chars.update(word)
        for c in sorted(chars):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        self.merges = train_merges(corpus, num_merges)
        for a, b in self.merges:
            merged = a + b
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
        self._rebuild_index()

    def _apply_merges(self, word_tokens):
        while True:
            pairs = _get_pairs(word_tokens)
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            _, best = min(ranked)
            word_tokens = _merge_word(word_tokens, best)
        return word_tokens

    def encode(self, text):
        ids = []
        for word in pre_tokenize(text):
            for tok in self._apply_merges(word):
                ids.append(self.vocab.get(tok, self.unk_id))
        return ids

    def decode(self, ids):
        toks = [self.id_to_token.get(i, "") for i in ids]
        return "".join(toks).replace(BOUNDARY, " ").strip()

    def to_dict(self):
        return {
            "version": 1,
            "vocab_size": len(self.vocab),
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.vocab = dict(d["vocab"])
        tok.merges = [tuple(m) for m in d["merges"]]
        tok.unk_id = tok.vocab["<unk>"]
        tok._rebuild_index()
        return tok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: PASS (all bpe tests).

- [ ] **Step 5: Commit**

```bash
git add bpe.py tests/test_bpe.py && git commit -m "feat: Tokenizer encode/decode/serialize"
```

---

## Task 5: Metrics and build score

**Files:**
- Create: `metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from metrics import language_metrics, aggregate_metrics

def test_language_metrics_computes_ratio():
    # 2 words; encoder returns 3 tokens -> X = 1.5
    class FakeTok:
        def encode(self, text):
            return [0, 1, 2] if text == "one two" else []
    m = language_metrics(FakeTok(), "one two")
    assert m["words"] == 2
    assert m["tokens"] == 3
    assert m["X"] == 1.5
    assert m["within_1_2"] is False

def test_aggregate_computes_build_score():
    per_lang = {
        "en": {"X": 1.0}, "hi": {"X": 1.2}, "te": {"X": 1.1}, "es": {"X": 1.05},
    }
    agg = aggregate_metrics(per_lang)
    assert agg["X_max"] == 1.2
    assert agg["X_min"] == 1.0
    assert abs(agg["build_score"] - 1000 / (1.2 - 1.0)) < 1e-6

def test_aggregate_handles_zero_spread():
    per_lang = {"en": {"X": 1.1}, "es": {"X": 1.1}}
    agg = aggregate_metrics(per_lang)
    assert agg["build_score"] == float("inf")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with "No module named 'metrics'".

- [ ] **Step 3: Write minimal implementation**

```python
# metrics.py
import re

def count_words(text):
    return len(re.findall(r"\S+", text))

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
        "spread": spread,
        "sorted_X": sorted(xs, reverse=True),
        "build_score": build_score,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py && git commit -m "feat: metrics and build score"
```

---

## Task 6: Data fetching with cache

**Files:**
- Create: `fetch_data.py`
- Test: `tests/test_train.py` (fetch-cache test only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import os
from fetch_data import clean_text, take_words, LANGUAGES

def test_clean_text_strips_references_and_whitespace():
    raw = "India[1] is a  country.\n\nSee also"
    cleaned = clean_text(raw)
    assert "[1]" not in cleaned
    assert "  " not in cleaned

def test_take_words_limits_count():
    text = " ".join(["word"] * 10000)
    assert len(take_words(text, 5000).split()) == 5000

def test_languages_config_has_four_entries():
    assert set(LANGUAGES.keys()) == {"en", "hi", "te", "es"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with "No module named 'fetch_data'".

- [ ] **Step 3: Write minimal implementation**

```python
# fetch_data.py
import os, re, json, urllib.request, urllib.parse

DATA_DIR = "data"

LANGUAGES = {
    "en": {"name": "English", "title": "India"},
    "hi": {"name": "Hindi", "title": "भारत"},
    "te": {"name": "Telugu", "title": "భారత దేశం"},
    "es": {"name": "Spanish", "title": "India"},
}

WORD_LIMIT = 5000

def clean_text(raw):
    raw = re.sub(r"\[\d+\]", "", raw)          # drop [1] style refs
    raw = re.sub(r"\s+", " ", raw)             # collapse whitespace
    return raw.strip()

def take_words(text, limit=WORD_LIMIT):
    return " ".join(text.split()[:limit])

def fetch_page(lang, title):
    """Fetch plain-text extract of a Wikipedia article."""
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "format": "json", "titles": title, "redirects": "1",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "bpe-tokenizer/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = data["query"]["pages"]
    return next(iter(pages.values()))["extract"]

def load_corpus(lang, refresh=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{lang}.txt")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return f.read()
    raw = fetch_page(lang, LANGUAGES[lang]["title"])
    text = take_words(clean_text(raw))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

if __name__ == "__main__":
    for lang in LANGUAGES:
        text = load_corpus(lang, refresh=True)
        print(f"{lang}: {len(text.split())} words cached")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS (3 passed). Network is not touched by these tests.

- [ ] **Step 5: Fetch real data (manual, requires network)**

Run: `python fetch_data.py`
Expected: four lines like "en: 5000 words cached". Creates `data/*.txt`.

- [ ] **Step 6: Commit**

```bash
git add fetch_data.py tests/test_train.py data/ && git commit -m "feat: Wikipedia fetch with cache"
```

---

## Task 7: Vocab allocation with auto-adjust

**Files:**
- Create: `train.py`
- Test: `tests/test_train.py` (allocation test)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_train.py
from train import build_tokenizer, VOCAB_SIZE

def test_build_tokenizer_respects_total_vocab():
    corpora = {
        "en": "the quick brown fox jumps over the lazy dog " * 20,
        "es": "el rapido zorro marron salta sobre el perro " * 20,
        "hi": "भारत एक देश है भारत महान है " * 20,
        "te": "భారత దేశం గొప్ప దేశం భారత " * 20,
    }
    tok, per_lang, agg = build_tokenizer(corpora, rounds=2)
    assert len(tok.vocab) <= VOCAB_SIZE
    assert set(per_lang.keys()) == set(corpora.keys())
    assert "build_score" in agg

def test_build_tokenizer_reports_ratios_per_language():
    corpora = {k: ("word " * 100) for k in ["en", "es", "hi", "te"]}
    tok, per_lang, agg = build_tokenizer(corpora, rounds=1)
    for m in per_lang.values():
        assert m["X"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -k build_tokenizer -v`
Expected: FAIL with "cannot import name 'build_tokenizer'".

- [ ] **Step 3: Write minimal implementation**

```python
# train.py
import json
from bpe import Tokenizer, pre_tokenize, SPECIAL_TOKENS
from metrics import language_metrics, aggregate_metrics

VOCAB_SIZE = 10000

def _all_base_chars(corpora):
    chars = set()
    for text in corpora.values():
        for word in pre_tokenize(text):
            chars.update(word)
    return chars

def _train_shared(corpora, budgets, base_chars):
    """Train one shared tokenizer: base chars + per-language merges up to budgets."""
    tok = Tokenizer()
    for c in sorted(base_chars):
        if c not in tok.vocab:
            tok.vocab[c] = len(tok.vocab)
    # Accumulate merges per language, appending new ones to the shared vocab.
    from bpe import train_merges
    seen = set()
    for lang, text in corpora.items():
        merges = train_merges([text], budgets[lang])
        for m in merges:
            if m not in seen:
                seen.add(m)
                tok.merges.append(m)
                merged = m[0] + m[1]
                if merged not in tok.vocab:
                    tok.vocab[merged] = len(tok.vocab)
    tok._rebuild_index()
    return tok

def build_tokenizer(corpora, rounds=5):
    base_chars = _all_base_chars(corpora)
    reserved = len(base_chars) + len(SPECIAL_TOKENS)
    total_merges = VOCAB_SIZE - reserved
    langs = list(corpora.keys())
    budgets = {l: total_merges // len(langs) for l in langs}
    # Auto-adjust: move budget from lowest-X language to highest-X language.
    per_lang, agg, tok = None, None, None
    for _ in range(max(1, rounds)):
        tok = _train_shared(corpora, budgets, base_chars)
        per_lang = {l: language_metrics(tok, corpora[l]) for l in langs}
        agg = aggregate_metrics(per_lang)
        hi = max(langs, key=lambda l: per_lang[l]["X"])
        lo = min(langs, key=lambda l: per_lang[l]["X"])
        if hi == lo:
            break
        step = max(1, total_merges // 100)
        if budgets[lo] - step >= 1:
            budgets[lo] -= step
            budgets[hi] += step
    return tok, per_lang, agg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -k build_tokenizer -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py && git commit -m "feat: shared-vocab allocation with auto-adjust"
```

---

## Task 8: HTML report template

**Files:**
- Create: `report_template.py`
- Test: `tests/test_train.py` (render test)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_train.py
import json as _json
from report_template import render_report

def test_render_report_contains_build_score_and_table():
    per_lang = {"en": {"words": 100, "tokens": 110, "X": 1.1, "within_1_2": True, "vocab_used": 500}}
    agg = {"X_max": 1.1, "X_min": 1.1, "spread": 0.0, "sorted_X": [1.1], "build_score": float("inf")}
    tokenizer_dict = {"version": 1, "vocab": {"a": 0}, "merges": []}
    html = render_report(per_lang, agg, tokenizer_dict, {"en": "English"})
    assert "Build Score" in html
    assert "English" in html
    assert "Export tokenizer JSON" in html
    # Tokenizer JSON is embedded for offline export.
    assert "1.1" in html

def test_render_report_embeds_valid_json():
    tokenizer_dict = {"version": 1, "vocab": {"a": 0}, "merges": [["a", "b"]]}
    html = render_report({}, {"build_score": 5.0, "sorted_X": [], "X_max": 0, "X_min": 0, "spread": 0}, tokenizer_dict, {})
    # The embedded blob must be parseable back out.
    start = html.index("id=\"tokenizer-data\">") + len("id=\"tokenizer-data\">")
    end = html.index("</script>", start)
    blob = html[start:end].strip()
    assert _json.loads(blob)["version"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -k render_report -v`
Expected: FAIL with "No module named 'report_template'".

- [ ] **Step 3: Write minimal implementation**

```python
# report_template.py
import json, html as html_lib

def _rows(per_lang, lang_names):
    out = []
    for lang, m in per_lang.items():
        name = html_lib.escape(lang_names.get(lang, lang))
        ok = "✅" if m.get("within_1_2") else "❌"
        out.append(
            f"<tr><td>{name}</td><td>{m['words']}</td><td>{m['tokens']}</td>"
            f"<td>{m['X']}</td><td>{ok}</td><td>{m.get('vocab_used', '')}</td></tr>"
        )
    return "\n".join(out)

def render_report(per_lang, agg, tokenizer_dict, lang_names):
    rows = _rows(per_lang, lang_names)
    score = agg["build_score"]
    score_str = "∞" if score == float("inf") else f"{score:.2f}"
    sorted_x = ", ".join(str(x) for x in agg.get("sorted_X", []))
    blob = json.dumps(tokenizer_dict, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>BPE Tokenizer Report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#0f172a;color:#e2e8f0}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #334155;padding:.5rem .75rem;text-align:left}}
th{{background:#1e293b}}
.score{{font-size:3rem;font-weight:700;color:#38bdf8}}
.card{{background:#1e293b;padding:1rem 1.5rem;border-radius:8px;margin:1rem 0}}
button{{background:#38bdf8;border:0;color:#0f172a;padding:.6rem 1rem;border-radius:6px;font-weight:600;cursor:pointer}}
</style></head><body>
<h1>Multilingual BPE Tokenizer</h1>
<div class="card"><div>Build Score = 1000 / (X_max − X_min)</div>
<div class="score">{score_str}</div>
<div>X_max = {agg.get('X_max')} &nbsp; X_min = {agg.get('X_min')} &nbsp; spread = {agg.get('spread')}</div>
<div>Sorted X (desc): {sorted_x}</div></div>
<h2>Per-language statistics</h2>
<table><thead><tr><th>Language</th><th>Words</th><th>Tokens</th><th>X (tokens/word)</th><th>≤ 1.2</th><th>Vocab used</th></tr></thead>
<tbody>
{rows}
</tbody></table>
<button onclick="exportJSON()">Export tokenizer JSON</button>
<script type="application/json" id="tokenizer-data">{blob}</script>
<script>
function exportJSON() {{
  const data = document.getElementById('tokenizer-data').textContent;
  const blob = new Blob([data], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'tokenizer.json'; a.click();
  URL.revokeObjectURL(url);
}}
</script>
</body></html>"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -k render_report -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add report_template.py tests/test_train.py && git commit -m "feat: HTML report widget with JSON export"
```

---

## Task 9: End-to-end main entrypoint

**Files:**
- Modify: `train.py`
- Test: `tests/test_train.py` (main writes files)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_train.py
import os, json
from train import run

def test_run_writes_tokenizer_and_report(tmp_path):
    corpora = {
        "en": "the quick brown fox " * 30,
        "es": "el rapido zorro marron " * 30,
        "hi": "भारत एक देश है " * 30,
        "te": "భారత దేశం గొప్ప " * 30,
    }
    tj = tmp_path / "tokenizer.json"
    rep = tmp_path / "report.html"
    run(corpora, tokenizer_path=str(tj), report_path=str(rep), rounds=2)
    assert tj.exists() and rep.exists()
    data = json.loads(tj.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "metrics" in data
    assert data["metrics"]["build_score"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -k run_writes -v`
Expected: FAIL with "cannot import name 'run'".

- [ ] **Step 3: Write minimal implementation**

```python
# add to train.py
import os
from fetch_data import LANGUAGES, load_corpus
from report_template import render_report

def _count_vocab_used(tok, text):
    return len(set(tok.encode(text)))

def run(corpora, tokenizer_path="tokenizer.json", report_path="report.html", rounds=5):
    tok, per_lang, agg = build_tokenizer(corpora, rounds=rounds)
    for lang in per_lang:
        per_lang[lang]["vocab_used"] = _count_vocab_used(tok, corpora[lang])
    tokenizer_dict = tok.to_dict()
    # Build score is infinite when spread is 0; JSON can't hold inf, so store null.
    bs = agg["build_score"]
    tokenizer_dict["metrics"] = {
        **{l: per_lang[l] for l in per_lang},
        "build_score": None if bs == float("inf") else bs,
        "X_max": agg["X_max"], "X_min": agg["X_min"], "spread": agg["spread"],
        "sorted_X": agg["sorted_X"],
    }
    lang_names = {l: LANGUAGES[l]["name"] for l in per_lang if l in LANGUAGES}
    for l in per_lang:
        lang_names.setdefault(l, l)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)
    html = render_report(per_lang, agg, tokenizer_dict, lang_names)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return tok, per_lang, agg

def main():
    corpora = {lang: load_corpus(lang) for lang in LANGUAGES}
    tok, per_lang, agg = run(corpora)
    bs = agg["build_score"]
    print("Build score:", "inf" if bs == float("inf") else round(bs, 2))
    for l, m in per_lang.items():
        print(f"  {l}: X={m['X']} words={m['words']} tokens={m['tokens']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -k run_writes -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py && git commit -m "feat: end-to-end train entrypoint"
```

---

## Task 10: Full pipeline run and verification

**Files:**
- Modify: `README.md` (add results note)

- [ ] **Step 1: Run fetch + train against real data (requires network)**

Run: `python fetch_data.py && python train.py`
Expected: prints a build score and four per-language X values; creates `tokenizer.json` and `report.html`.

- [ ] **Step 2: Verify the vocab size cap**

Run: `python -c "import json; d=json.load(open('tokenizer.json')); print(len(d['vocab']))"`
Expected: a number ≤ 10000.

- [ ] **Step 3: Open the widget**

Run: `open report.html`
Expected: build score, per-language table with ≤1.2 flags, and a working "Export tokenizer JSON" button that downloads `tokenizer.json`.

- [ ] **Step 4: Note results in README and commit**

Add a short "## Results" section to `README.md` with the achieved build score and the four X values, then:

```bash
git add README.md tokenizer.json report.html data/ && git commit -m "docs: record pipeline results"
```

---

## Self-Review Notes

- **Spec coverage:** fetch (T6), from-scratch BPE train/encode/decode/serialize (T2–T4), shared 10k allocation with auto-adjust (T7), metrics + build score (T5), HTML widget with JSON export (T8), tokenizer.json + end-to-end (T9), real run (T10). All spec sections covered.
- **Placeholder scan:** every code step contains full code; no TBD/TODO.
- **Type consistency:** `Tokenizer.train/encode/decode/to_dict/from_dict`, `train_merges`, `pre_tokenize`, `SPECIAL_TOKENS`, `BOUNDARY`, `language_metrics`, `aggregate_metrics`, `build_tokenizer`, `run`, `render_report`, `load_corpus`, `LANGUAGES` are used with consistent signatures across tasks.
