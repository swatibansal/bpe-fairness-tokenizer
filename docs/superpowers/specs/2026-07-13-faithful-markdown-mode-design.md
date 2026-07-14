# Faithful-Markdown mode for the stdlib tokenizer

Date: 2026-07-13

## Goal

Make this repo's from-scratch, standard-library-only BPE tokenizer train and be
evaluated on a **faithful Markdown** corpus (the same honest framing used by the
sibling `bpe-tokenizer-markdown` project), and produce a head-to-head comparison of
our akshara-aware engine against a HuggingFace BPE on the **same corpus and the same
metric**.

## Principles kept vs. changed

Keep (the repo's identity):
- Tokenizer engine stays **standard-library only** — no ML/tokenizer libs in
  `encode`/`decode`/`train`.
- **Akshara-aware Brahmic segmentation** (grapheme clusters, virama conjuncts,
  ZWJ/ZWNJ binding).
- **Frequency-thresholded seeded clusters** + **code-point fallback → no `<unk>`**.
- **Hill-climb budget optimizer** over a shared 10k vocab.
- Languages **en / hi / te / bn** (Bengali retained).

Change (to drive on faithful Markdown):
- Corpus becomes **faithful Markdown**, built by our own fetcher.
- Metric becomes **faithful-unit fertility** + spread score + **Hindi penalty**,
  matching `bpe-tokenizer-markdown` exactly.
- **Full visible-character faithfulness** enforced and tested.
- Optimizer retargets from the English-≤1.2 gate to **maximize the Hindi-adjusted
  spread score** on faithful-unit fertility.

## Components

### 1. Corpus builder — `fetch_faithful.py` (new)
Fetches Wikipedia REST HTML for India (en), भारत (hi), భారతదేశం (te), ভারত (bn);
strips only script/style/meta/link noise; absolutizes links; converts to Markdown via
`markdownify`; normalizes; writes `corpus/<lang>.faithful.md` and `.txt`. External libs
(`requests`, `beautifulsoup4`, `lxml`, `markdownify`) live in a separate
`requirements-corpus.txt`; the tokenizer core stays dependency-free.

### 2. Faithful-unit metric — extend `metrics.py`
Implement `faithful_units()` with `unicodedata.category` (stdlib), reproducing the
`regex` pattern `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`: a maximal run of chars whose
category starts with `L`/`M`/`N` is one unit; every other non-whitespace char is its
own unit. Add `faithful_language_metrics()` (fertility = tokens/faithful_units) and
`faithful_aggregate()` (score = 1000/spread, `hindi_penalty = exp(max(0, hi/1.2 − 1))`,
`hindi_adjusted_score = score / hindi_penalty`). Existing word-based functions are
untouched.

### 3. Engine faithfulness + Markdown-aware pre-tokenization — `bpe.py`
- **Markdown-aware split:** apply the GPT-4-style chunker (`_split_latin`, renamed
  `_split_word`) to **every** word regardless of script, instead of keeping any
  Brahmic-containing word whole. Grapheme clustering already runs first, so matras and
  conjuncts stay intact inside a cluster; only separate punctuation clusters (e.g. the
  danda `।`, brackets, pipes, URL characters) now split off — which is what faithful
  Markdown needs. Removes the `_is_indic` branch.
- **Faithfulness:** define it against the NFC-normalized input — `decode(encode(t))`
  preserves the same non-whitespace characters as `normalize(t)`. Holds by construction
  (every corpus code point is seeded, every cluster is emitted, decode only normalizes
  whitespace). Add `visible_chars()` helper + a corpus test asserting it. Documented
  shared limitation: a literal `▁` (U+2581) in input round-trips to a space.

### 4. Optimizer retarget — `train.py`
Parameterize `build_tokenizer(corpora, rounds=None, objective=None)` with a pluggable
objective `{metrics: fn(tok, corpora, langs) -> (per_lang, agg), score_key: fn(agg,
per_lang) -> tuple}`. Default objective = current word-based + English-≤1.2 gate, so the
existing widget path and tests are unchanged. A faithful objective (faithful-unit
fertility, ranked by Hindi-adjusted score) is passed by `compare.py`. The hill-climb
orders donors/recipients by each language's `X`, which both objectives expose.

### 5. Deliverable — `compare.py` (new) + tests
Decision (updated 2026-07-13): compare against `bpe-tokenizer-markdown`'s **published**
ratios rather than retraining HuggingFace locally, so no `tokenizers` install is needed.
`compare.py`:
- loads the faithful corpus from `corpus/<lang>.faithful.txt` (errors with guidance if
  missing),
- trains our engine via `build_tokenizer(..., objective=FAITHFUL)`,
- evaluates it with the `faithful_units` metric,
- prints our per-language fertility, spread, raw/Hindi-adjusted score, vocab size, and a
  visible-character faithfulness check **side by side with the published ratios** from
  the sibling repo's `SOLUTION.md` (en/hi/te are 1:1; their 4th language Maithili vs our
  Bengali is flagged as indicative, not strictly comparable).

Tests: add `faithful_units` test, faithful-metric test, visible-character faithfulness
test on mixed-script text, and update the two Brahmic-whole pre-tokenize tests to the new
punctuation-splitting behavior.

## Addendum (2026-07-13): metaspace made the shipped default

After comparing to the published ratios, the GPT4-style split floored fertility near 1.0
(punctuation boundaries prevent cross-unit merges). Decisions taken:

- Added a **`WHITESPACE` (metaspace) pre-tokenization style** to `bpe.py` (whitespace-only
  split, punctuation merges in) alongside `GPT4`, kept akshara-aware, persisted as
  `pretok_style` in `tokenizer.json`. On the faithful corpus this drops fertility from
  ~0.97 to ~0.8 and beats the published score (11,787 vs 6,502) with a tighter spread.
- Made metaspace + the faithful objective the **shipped default** in `train.run()`/`main()`,
  training on `corpus/<lang>.faithful.txt`; the word-based/GPT4 path stays available.
- **Performance:** `train_merges` now uses incremental pair counts (~15× faster,
  output-identical, proved by a test); `Tokenizer.count_tokens` dedups word tokenization;
  `build_tokenizer(search_sample=)` runs the hill-climb on a corpus sample and reports on
  the full corpus. Full build dropped from >15 min to ~3 min.
- Updated `report_template.py` (and thus `index.html`): faithful-unit fertility, Hindi
  penalty card, metaspace worked example, and a JS port that honors `pretok_style` — token
  ids verified identical to Python via Node.

## Addendum 2 (2026-07-13): single shared BPE made the default

Exploring how close we could get to the published ~0.6 fertility showed objective tweaks
(compression, min-fertility) and freeing seeded budget only reached ~0.75–0.80 — because
the per-language pre-train-and-stitch approach gives English only ~2,300 of the 10k slots
and interleaves merges. Training **one shared BPE over the weighted concatenation**
(`build_shared_tokenizer`, weights en:3/hi:4/te:4/bn:2, seed_min_freq=50) — the way the
HuggingFace baseline is built — reached en 0.617 / hi 0.603 / te 0.694 / bn 0.714, matching
the published 0.598/0.579/0.673/0.733 and beating its spread score (9,042 vs 6,502).

Decisions:
- `run()` defaults to `build_shared_tokenizer` (shipped `tokenizer.json` uses it,
  `build: shared_bpe`); the spread-optimizing hill-climb stays available via
  `run(shared=False)`. Exploratory compression / min-fertility objectives were dropped.
- Widget/JS unchanged (style already honored); token ids re-verified identical to Python.

## Addendum 3 (2026-07-14): byte-level fallback for universal faithfulness

The grader rejected a submission because `decode(encode(text))` turned visible characters
into `<unk>` (e.g. `#`, `_`, and any character absent from our four-language corpus such
as `@`). Our no-`<unk>` guarantee was only corpus-scoped, but a faithful tokenizer must
round-trip ANY visible character.

Fix: **byte-level fallback** in `bpe.py`. The base vocabulary now seeds all 256 byte
tokens (`<0x00>`..`<0xFF>`); `encode` emits the UTF-8 bytes of any code point not in the
vocabulary, and `decode` buffers byte tokens and UTF-8-decodes them. So no input produces
`<unk>` and every visible character survives the round trip. The widget JS mirrors this
(TextEncoder/TextDecoder), verified identical to Python. Merge budget calculations account
for the 256 byte tokens; fertility is essentially unchanged (en 0.615→0.618). Adversarial
tests (`@`, emoji, CJK, the exact failing URL) were added.

## Out of scope (YAGNI)
- No changes to `index.html` / the JS port.
- No whitespace-structure round-tripping (visible-char faithfulness only).
- Not reproducing the sibling repo's exact corpus (we keep Bengali, fetch our own).
