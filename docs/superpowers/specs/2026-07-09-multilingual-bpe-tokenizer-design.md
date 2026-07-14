# Multilingual BPE Tokenizer — Design

Date: 2026-07-09

## Goal

Build one shared **10,000-token** BPE vocabulary that covers four languages, each
trained on that language's Wikipedia "India" article:

- English (`en`) — "India"
- Hindi (`hi`) — "भारत"
- Telugu (`te`) — "భారత దేశం"
- Spanish (`es`) — "India"

> **Superseded — see Revision 2026-07-09c:** Spanish was replaced by **Bengali
> (`bn`, "ভারত")**, corpora were enlarged to ~10,000 words, and metrics moved to
> held-out evaluation. The list above reflects the original spec, not the shipped set.

For each language, compute the ratio:

```
X = (total tokens produced tokenizing that language's text) / (total words in that text)
```

Constraint: each X should be ≤ 1.2 (reported, not enforced — flagged if violated).

Sort the four X values. With X_max the largest and X_min the smallest:

```
build_score = 1000 / (X_max - X_min)
```

The objective is to **maximize build_score**, i.e. make the four ratios as close
to each other as possible.

## Key Decisions (from brainstorming)

- **Stack:** Python for fetch + train + metrics; a self-contained static HTML widget.
- **Ratio X:** tokens per word (fertility). Denominator = total word count.
- **Corpus:** first ~5,000 words of each page, cached to disk for reproducibility.
- **Vocab:** single shared 10,000-token vocabulary across all four languages
  (not per-language). Allocated via balanced per-language merge budgets, with a
  light auto-adjust loop to shrink X_max − X_min.
- **BPE:** implemented from scratch (no external tokenizer library) so the whole
  pipeline is inspectable and exportable.
- **Widget:** generated static HTML file opened directly in a browser; JSON export
  via an in-page download button (data embedded inline, no server).

## Components

### 1. `fetch_data.py` — Data acquisition
- Fetch the "India" article from each language edition using the Wikipedia REST
  extract / API (`{lang}.wikipedia.org`).
- Titles (original spec): en=`India`, hi=`भारत`, te=`భారత దేశం`, es=`India`.
  **Superseded — see Revision 2026-07-09d for the final roster actually pulled**
  (es replaced by bn=`ভারত`; Telugu topped up with `భారతదేశ చরিత్ర`).
- Strip markup to plain text; take the first ~5,000 words per language.
- Cache to `data/<lang>.txt`. Re-runs use the cache (offline-reproducible).

### 2. `bpe.py` — BPE implementation
- Word-level pre-tokenization (split on whitespace/punctuation, preserve a word
  boundary marker so tokens can reconstruct spacing).
- Seed vocab with all base characters across Latin, Devanagari, and Telugu scripts
  plus special tokens, so every language is representable at ids from the start.
- `train(corpus, num_merges)`: standard BPE merge loop (count adjacent pairs, merge
  most frequent, repeat).
- `encode(text)` → token ids; `decode(ids)` → text.
- Serialize / deserialize vocab + merges.

### 3. `train.py` — Orchestration
- Load the four cached corpora.
- Compute the shared base-character set and special tokens.
- Distribute the remaining merge budget (10,000 − base − specials) across the four
  languages. Start from an even split; run a few auto-adjust rounds that move merge
  budget from the language with the lowest X toward the one with the highest X to
  shrink X_max − X_min.
- Merge the per-language results into one 10,000-entry vocabulary.
- Write `tokenizer.json` and hand metrics to the report generator.

### 4. `metrics.py` — Metrics
- Per language: total words, total tokens, X = tokens/words, vocab slots used,
  and a `<= 1.2` flag.
- Aggregate: sorted X list, X_max, X_min, `X_max - X_min`, and build_score.

### 5. `index.html` (generated) — Widget
- Table: per-language words, tokens, X, ≤1.2 status, vocab slots used.
- The sorted-X calculation, X_max − X_min, and the build_score displayed prominently.
- **"Export tokenizer JSON"** button that downloads the full tokenizer. Tokenizer
  data is embedded inline in the page so it works with no server.

## Data Formats

### `tokenizer.json`
```json
{
  "version": 1,
  "vocab_size": 10000,
  "special_tokens": ["<pad>", "<unk>", ...],
  "vocab": { "token": id, "...": 0 },
  "merges": [["a", "b"], ["...", "..."]],
  "base_chars": ["...", "..."],
  "per_language_budgets": { "en": 0, "hi": 0, "te": 0, "es": 0 },
  "metrics": {
    "en": { "words": 0, "tokens": 0, "X": 0.0, "vocab_used": 0, "within_1_2": true },
    "...": {},
    "build_score": 0.0,
    "X_max": 0.0,
    "X_min": 0.0
  }
}
```

## Revision 2026-07-09b: Grapheme-aware (Brahmic) pre-tokenization

Motivation: Indic scripts are not "English with different letters." A naive
code-point split (`list(word)`) orphans combining vowel signs (matras), separates
viramas from the consonants they bind, and treats ZWJ/ZWNJ as free-floating
tokens. BPE then spends merges re-stitching these, inflating Hindi/Telugu
fertility and hurting language fairness. Reference: *BrahmicTokenizer-131K: An
Indic-Capable Drop-In Replacement for o200k_base* (Rohan Shravan, arXiv 2605.29379),
which uses grapheme-cluster (akshara) pre-tokenization + normalization on top of BPE.

Changes to `bpe.py`:

1. **Normalization** — `normalize(text)` applies Unicode **NFC** before any
   segmentation, so precomposed/decomposed and nukta variants tokenize identically.
2. **Grapheme-cluster segmentation** — `grapheme_clusters(word)` replaces
   `list(word)`. A cluster (akshara) is a base character plus any following
   combining marks (categories Mn/Mc/Me — matras, anusvara, nukta, visarga),
   ZWJ/ZWNJ, and virama-driven conjuncts (consonant + virama + consonant …). The
   **atomic units for BPE are now grapheme clusters, not code points.**
3. **Base vocabulary** — seeded with the observed grapheme clusters *and* their
   component code points. Code points give a graceful fallback: an out-of-domain
   cluster is emitted as its code points rather than `<unk>`.
4. **encode fallback** — a token absent from the vocab is decomposed into code
   points (each seeded) before resorting to `<unk>`.

Latin scripts are unaffected: under NFC, accented Spanish letters are single code
points, so each Latin cluster is one character — English still reaches X ≤ 1.2.
Hindi/Telugu words now segment into a handful of aksharas instead of many code
points, lowering their token counts and tightening the cross-language spread.

Cross-*Brahmic* literal token sharing (e.g., Devanagari↔Telugu) is limited because
the scripts occupy different code-point ranges; transliteration-based sharing is
noted as future work. The immediate win is correctness + fairness + lower Indic
fertility.

## Revision 2026-07-09c: Held-out evaluation, larger corpus, train-only constraint

Three changes were evaluated after the initial build. All numbers here are the
build score = 1000 / (X_max − X_min).

**Corpus (adopted).** The four languages are **English, Hindi, Telugu, Bengali**
(Spanish was replaced by Bengali so the three non-English languages are all Brahmic
and cluster under uniform akshara handling). Each corpus was enlarged from ~5,000 to
**~10,000 words** by topping every language up with the same roster of large
India-related articles (`fetch_data.py` now skips missing titles and rate-limits
requests).

**Held-out evaluation (adopted).** `metrics.split_corpus` splits each corpus 80/20;
merges/seeds/base vocab are learned from the 80% only and every reported ratio and
the build score are measured on the unseen 20%. Rationale: BPE partly memorizes its
training text, so train==eval fertility is optimistic by +0.1 to +0.4 tokens/word
(measured). The original ~1570 headline (5k, train==eval) re-scores to **994** on
held-out text.

**Punctuation/digit pre-tokenization (tried, REVERTED).** A GPT-style pre-tokenizer
splitting leading/trailing punctuation and grouping digit runs into ≤3 was
implemented and measured. Held-out 2×2 (pre-tok × corpus size):

| corpus | no pre-tok | + pre-tok |
|--------|-----------:|----------:|
| 5k     | 994        | 1330      |
| 10k    | **2869**   | 1552      |

Pre-tokenization helps when data is scarce but *hurts* at 10k (it forces a separate
token per punctuation mark / long number, fragmenting held-out text more than the
vocab-efficiency gain repays). Reverted; `pre_tokenize` stays akshara-only.

**More data is the dominant win.** 994 → ~2870 (no pre-tok, 5k → 10k) — the opposite
of what train==eval showed, and only visible under held-out evaluation.

**Constraint is now a train-set target (adopted).** English held-out fertility is
~1.9, so ≤1.2 is unreachable on unseen text. The ≤1.2 check is therefore made on
English's *training* split (labeled as such everywhere: `tokenizer.json` metrics,
the widget). `MAX_PER_LANG` was raised 4500 → 6000 so English can actually reach 1.2
on train (it needs ~5–6k merges on the 8k-word train split); with the constraint
feasible the optimizer maximizes build score properly, giving the final **≈ 2475**
(spread 0.40, Brahmic trio within 0.003). A ceiling too low leaves the constraint
infeasible, forcing best-effort mode and a lower score.

## Revision 2026-07-09d: Hard requirements — English eval ≤1.2 and zero `<unk>`

Two requirements were clarified as **hard acceptance criteria**, and they override the
held-out experiment from revision c:

1. **English fertility X ≤ 1.2 on the evaluation set.**
2. **No `<unk>` on any downloaded article text.**

Both imply the tokenizer is evaluated on the corpus it is built for (the Wikipedia
articles), so evaluation is on the **full downloaded corpus**, not a held-out split:

- Held-out English fertility is ~1.9 with a 10k shared vocab — **≤1.2 is unreachable
  on unseen text**, so a held-out split cannot satisfy requirement 1. The 80/20 split
  (revision c) was removed; `metrics.split_corpus` deleted.
- A held-out split can also orphan code points that appear only in the tail, which
  would violate requirement 2. Training/seeding on the full corpus avoids this.

**No-`<unk>` guarantee (by construction).** Every code point in the corpus is seeded
into the base vocabulary; `word_to_atoms` decomposes any non-seeded cluster to code
points, and `encode` decomposes any out-of-vocab token to code points before `<unk>`.
So any text composed of characters seen in the corpus resolves with zero `<unk>`.
`train.verify_no_unk` checks this on the real articles at the end of every run (raising
on any violation), and `tests` cover it (`TestNoUnk`).

**Corpus size reverted 10k → 5k.** With eval-on-corpus and the hard ≤1.2 English
target, corpus size is a lever in the opposite direction from revision c:

| Corpus | Build score | English X | Brahmic cluster |
|--------|------------:|----------:|-----------------|
| 5k (adopted) | **1743** | 1.200 ✅ | ≈ 1.773 |
| 10k          | 905      | 1.200 ✅ | ≈ 2.30 |

With a fixed 10k vocabulary, a larger corpus forces English to spend ~6k merges to
hold ≤1.2, starving Hindi/Telugu/Bengali. 5k lets English reach 1.2 with far fewer
merges, freeing budget for the other three; their cluster drops to ~1.773 and the
spread narrows. (`MAX_PER_LANG` stays 6000 — non-binding at 5k, where English
saturates well below it.)

**Final corpus actually pulled** (WORD_LIMIT = 5,000 words/language; roster titles
beyond the first are fallbacks only fetched when the primary article is too short):

| Lang | Words | Article(s) used |
|------|------:|-----------------|
| en   | 5,000 | India |
| hi   | 5,000 | भारत |
| te   | 5,000 | భారత దేశం (2,511) + భారతదేశ చরిత్ర (2,489) |
| bn   | 5,000 | ভারত |

**Final metrics** (measured on the full corpus above):

| Lang | Tokens | X = tokens/word |
|------|-------:|----------------:|
| en   | 6,000  | **1.200** (constraint met) |
| hi   | 8,864  | 1.7728 |
| te   | 8,868  | 1.7736 |
| bn   | 8,868  | 1.7736 |

Build score ≈ **1743.38**, spread 0.5736 (X_max 1.7736 − X_min 1.200), 0 `<unk>` on
all four articles, 9,880 / 10,000 vocab used. The akshara segmentation, NFC
normalization, and frequency-thresholded seeding from the earlier revisions are
retained; the GPT-style punctuation/digit pre-tokenizer trialled in revision c stayed
reverted (it raises tokens/word on this corpus-eval metric).

## Revision 2026-07-10: Hybrid (Latin-only) GPT-4-style pre-tokenization

Motivation: whitespace-only splitting keeps punctuation glued to words, so a rare,
punctuation-heavy English sentence tokenizes poorly even though it is in-corpus (the
etymology sentence from the "India" article scored 1.66 tokens/word). Reviewed
minbpe's `regex.py` (GPT-4 split pattern). We do **not** adopt its byte-level encoding
(each Devanagari code point is 3 UTF-8 bytes — 12.9/20.2/18.4 bytes per word for
hi/te/bn — which would wreck Indic fertility and the akshara design) and we cannot use
its `regex` dependency (stdlib-only constraint). We do adopt the **split rules**.

Measured options (5k corpus, corpus eval, hard English ≤1.2, all met with 0 `<unk>`):

| Pre-tokenization | Build score | English X | etymology sentence | Brahmic cluster |
|------------------|------------:|----------:|-------------------:|-----------------|
| whitespace-only (previous) | 1743 | 1.200 | 1.658 | 1.773 |
| GPT-4 rules, all scripts | 1442 | 1.200 | 1.237 | 1.892 |
| **GPT-4 rules, Latin only (adopted)** | **1593** | 1.200 | **1.237** | 1.827 |

**Adopted: hybrid.** A whitespace word containing any Brahmic letter (code points
0x0900–0x0DFF) is kept whole as akshara clusters; every other word (Latin, Greek,
numbers, punctuation) is split GPT-4-style — contractions and trailing punctuation
split off, one leading punctuation attaches to the following letter run, digit runs
group into ≤3. Only the first chunk of each whitespace word carries BOUNDARY, and
merges never cross a chunk boundary. Implemented in `bpe.py`
(`_split_latin`, `_atom_class`, `_contraction_len`, `_is_indic`) and mirrored in the
widget JS (verified id-for-id against Python with Node).

Applying the rules to Latin only (not all scripts) keeps the full English robustness
gain (etymology sentence 1.66 → 1.24) while roughly halving the fairness cost of
splitting every script's punctuation. Both hard requirements (English ≤1.2, no `<unk>`)
still hold. Final build score ≈ **1592.9**; English 1.200, Hindi 1.827, Telugu 1.827,
Bengali 1.828.

## Out of Scope (YAGNI)
- No training server / web backend.
- No support for languages beyond the four specified.
- No incremental / streaming training.
- No exhaustive hyperparameter search — a light auto-adjust loop only.
