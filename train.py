"""Orchestrate training of the shared 10k-token tokenizer and render outputs."""

import json
import os
import re
import unicodedata

import bpe as _bpe
from bpe import (Tokenizer, pre_tokenize, train_merges, seeded_clusters_from,
                 cluster_frequencies, _is_virama, BOUNDARY, ZWJ, ZWNJ, SPECIAL_TOKENS,
                 BYTE_TOKENS, GPT4, WHITESPACE, CLUSTER_MIN_FREQ)
from metrics import (language_metrics, aggregate_metrics,
                     faithful_language_metrics, faithful_aggregate)
from fetch_data import LANGUAGES, load_corpus
from report_template import render_report

VOCAB_SIZE = 10000
MAX_PER_LANG = 6000   # ceiling each language is pre-trained to (prefix-stable slicing).
                      # Must exceed what English needs to hit its train-set <=1.2
                      # target (~5-6k merges on the 8k-word train split); a lower
                      # ceiling makes the constraint infeasible and forces the
                      # optimizer into best-effort mode, which also lowers the score.
MIN_PER_LANG = 100    # never starve a language below this many merges

# Hard constraint: English tokens/word must not exceed this (legacy word objective).
CONSTRAINT_LANG = "en"
CONSTRAINT_MAX_X = 1.2

# Per-language repetition weights for the shipped single shared BPE (mirrors the
# published bpe-tokenizer-markdown run). Upweighting the smaller corpora keeps their
# scripts from being starved of merges when one BPE is trained over the concatenation.
TRAIN_WEIGHTS = {"en": 3, "hi": 4, "te": 4, "bn": 2}
# Aksharas are seeded as whole atoms only above this frequency in the shared build; a
# higher threshold than the default frees vocabulary for merges (lower fertility) while
# rarer aksharas still round-trip via their seeded code points.
SHARED_SEED_MIN_FREQ = 50


def _base_atoms(corpora, seeded, style=GPT4):
    """Base vocabulary atoms: frequent aksharas (kept whole) plus all code points.

    Rare aksharas (not in `seeded`) are not given a slot; they decompose to code
    points at encode time, freeing vocabulary for merges.
    """
    chars = set()
    for text in corpora.values():
        for word in pre_tokenize(text, style):
            for cluster in word:
                if len(cluster) == 1 or cluster in seeded:
                    chars.add(cluster)
                chars.update(cluster)  # code points, for the encode fallback
    return chars


def _build_from_budgets(full_merges, base_chars, budgets, langs, seeded, style=GPT4):
    """Assemble one shared tokenizer from pre-trained per-language merge lists.

    Relies on BPE prefix-stability: full_merges[lang][:budget] is exactly the
    tokenizer that language would have with `budget` merges. Merges are deduped
    into a single shared vocab, so the total never exceeds VOCAB_SIZE.
    """
    tok = Tokenizer(pretok_style=style)
    tok.seeded_clusters = set(seeded)
    for c in sorted(base_chars):
        if c not in tok.vocab:
            tok.vocab[c] = len(tok.vocab)
    seen = set()
    # The caller keeps the total budget at/under total_merges, and cross-language
    # dedup only shrinks the vocab further, so the VOCAB_SIZE cap is never reached
    # in practice. A sequential build therefore includes each language's full
    # budget with no truncation; the cap check is a safety net only.
    for lang in langs:
        for m in full_merges[lang][:budgets[lang]]:
            if m in seen or len(tok.vocab) >= VOCAB_SIZE:
                continue
            seen.add(m)
            tok.merges.append(m)
            merged = m[0] + m[1]
            if merged not in tok.vocab:
                tok.vocab[merged] = len(tok.vocab)
    tok._rebuild_index()
    return tok


def _score_key(agg, per_lang):
    """Rank candidates: any allocation satisfying the English constraint beats any
    that violates it. Among feasible ones, higher build score wins (tie-broken by a
    lower worst-case ratio). Among infeasible ones, the one closest to feasibility
    (lowest English ratio) wins, so the search is always pulled toward feasibility.

    The constraint is the hard requirement English eval X <= 1.2 (agg["constraint_x"],
    measured on the full corpus like every other ratio).
    """
    en_x = agg["constraint_x"]
    feasible = en_x <= CONSTRAINT_MAX_X + 1e-9
    if feasible:
        return (1, agg["build_score"], -agg["X_max"])
    return (0, -en_x, 0.0)


def _word_metrics(tok, corpora, langs):
    """Default (word-based) objective: fertility = tokens/word, English <= 1.2 gate."""
    per_lang = {l: language_metrics(tok, corpora[l]) for l in langs}
    agg = aggregate_metrics(per_lang)
    agg["constraint_x"] = per_lang[CONSTRAINT_LANG]["X"]  # hard: English eval X <= 1.2
    return per_lang, agg


def _faithful_metrics(tok, corpora, langs):
    """Faithful-Markdown objective: fertility = tokens/faithful_unit, Hindi penalty."""
    per_lang = {l: faithful_language_metrics(tok, corpora[l]) for l in langs}
    agg = faithful_aggregate(per_lang, hindi_key="hi")
    return per_lang, agg


def _faithful_score_key(agg, per_lang):
    """Rank faithful candidates by Hindi-adjusted spread score (higher is better),
    tie-broken by a lower worst-case fertility."""
    return (agg["hindi_adjusted_score"], -agg["X_max"])


# An objective bundles how a candidate tokenizer is measured and ranked. The default
# reproduces the original word-based optimizer (with the English <= 1.2 gate); the
# faithful objective retargets the same hill-climb at faithful-unit fertility.
WORD_OBJECTIVE = {
    "metrics": _word_metrics,
    "score_key": _score_key,
    "constraint_lang": CONSTRAINT_LANG,
}
FAITHFUL_OBJECTIVE = {
    "metrics": _faithful_metrics,
    "score_key": _faithful_score_key,
    "constraint_lang": None,
}


def build_shared_tokenizer(corpora, style=WHITESPACE, weights=None,
                           seed_min_freq=SHARED_SEED_MIN_FREQ, vocab_size=VOCAB_SIZE):
    """Train ONE shared BPE over the weighted concatenation of all corpora.

    This is how the sibling HuggingFace tokenizer is built, and it reaches far lower
    absolute fertility than pre-training per language and stitching merge lists: the
    merges are ranked globally, so the largest corpus (English) naturally claims the
    slots it needs while `weights` keep the smaller scripts from being starved. No
    hill-climb — a single frequency-ranked pass fills the vocabulary.

    Returns (tokenizer, per_lang, agg) with the same faithful metrics as build_tokenizer.
    """
    langs = list(corpora.keys())
    weights = weights or {l: 1 for l in langs}
    combined = []
    for l in langs:
        combined += [corpora[l]] * weights.get(l, 1)

    # Seed from the UNWEIGHTED corpora so "seeded iff it occurs >= seed_min_freq times"
    # is literally true of the real corpus (and of the frequencies the widget shows).
    # Weighting is a merge-emphasis device only, applied when training merges below.
    seeded = seeded_clusters_from(list(corpora.values()), min_freq=seed_min_freq, style=style)
    base_chars = _base_atoms(corpora, seeded, style)
    reserved = len(base_chars) + len(SPECIAL_TOKENS) + len(BYTE_TOKENS)
    budget = max(0, vocab_size - reserved)
    merges = train_merges(combined, budget, seeded, style)

    tok = Tokenizer(pretok_style=style)
    tok.seeded_clusters = set(seeded)
    for c in sorted(base_chars):
        if c not in tok.vocab:
            tok.vocab[c] = len(tok.vocab)
    tok.merges = []
    for a, b in merges:
        if len(tok.vocab) >= vocab_size:
            break
        merged = a + b
        if merged not in tok.vocab:
            tok.vocab[merged] = len(tok.vocab)
        tok.merges.append((a, b))
    tok._rebuild_index()

    per_lang, agg = _faithful_metrics(tok, corpora, langs)
    return tok, per_lang, agg


def _sample_text(text, frac):
    """A representative strided sample of `text` by lines (deterministic).

    Used to speed up the budget search: fertility ratios are stable under sampling, so
    the hill-climb can score candidates on ~frac of the corpus while the final metrics
    are still computed on the full text.
    """
    lines = text.split("\n")
    if frac >= 1 or len(lines) < 40:
        return text
    step = max(2, round(1 / frac))
    return "\n".join(lines[::step])


def build_tokenizer(corpora, rounds=None, objective=None, style=GPT4, search_sample=None,
                    seed_min_freq=CLUSTER_MIN_FREQ):
    """Allocate the shared 10k vocab across languages to maximize the build score.

    Pre-trains each language once (BPE merges are prefix-stable, so a budget is
    just a slice), then runs an all-pairs hill-climb that keeps the total merge
    budget fixed at the full vocab and shifts merges between languages to cluster
    the four fertility ratios. Clustering minimizes X_max - X_min, which is what
    maximizes the build score, while full-vocab use keeps the ratios genuinely low.

    Merges are always learned from the full corpus. `search_sample` (a fraction in
    (0,1)) restricts only the hill-climb's *scoring* to a strided sample for speed on
    large corpora; the returned per-language metrics are recomputed on the full corpus.

    Two hard requirements hold on the eval: the objective's constraint (English X <= 1.2
    for the default word objective) and no <unk> on any corpus text (every corpus code
    point is seeded into the base vocabulary, so the encode fallback always resolves).
    """
    objective = objective or WORD_OBJECTIVE
    metrics_fn = objective["metrics"]
    score_key = objective["score_key"]
    constraint_lang = objective["constraint_lang"]
    langs = list(corpora.keys())

    seeded = seeded_clusters_from(list(corpora.values()), min_freq=seed_min_freq, style=style)
    base_chars = _base_atoms(corpora, seeded, style)
    reserved = len(base_chars) + len(SPECIAL_TOKENS) + len(BYTE_TOKENS)
    total_merges = max(0, VOCAB_SIZE - reserved)
    cap = min(MAX_PER_LANG, total_merges)

    # Pre-train each language once, up to the ceiling. Short corpora saturate early
    # (a language can never receive more merges than it actually produced).
    full_merges = {l: train_merges([corpora[l]], cap, seeded, style) for l in langs}
    max_useful = {l: len(full_merges[l]) for l in langs}

    # The hill-climb scores candidates on the (optionally sampled) search corpus.
    search_corpora = corpora
    if search_sample and 0 < search_sample < 1:
        search_corpora = {l: _sample_text(corpora[l], search_sample) for l in langs}

    def evaluate(budgets, on=None):
        tok = _build_from_budgets(full_merges, base_chars, budgets, langs, seeded, style)
        per_lang, agg = metrics_fn(tok, on or search_corpora, langs)
        return tok, per_lang, agg

    # Start from an even split (capped at what each language can actually use).
    even = total_merges // len(langs)
    budgets = {l: min(max_useful[l], max(MIN_PER_LANG, even)) for l in langs}

    # Seed feasibility: grow the constrained language's budget (borrowing from the
    # language with the most to spare) until its ratio meets the constraint, so the
    # hill-climb starts from a feasible allocation. Objectives without a hard
    # constraint (e.g. the faithful one) skip this and start from the even split.
    if constraint_lang:
        others = [l for l in langs if l != constraint_lang]
        while True:
            _, _, agg0 = evaluate(budgets)
            if agg0["constraint_x"] <= CONSTRAINT_MAX_X + 1e-9:
                break
            c_room = min(cap, max_useful[constraint_lang]) - budgets[constraint_lang]
            donor = max(others, key=lambda l: budgets[l])
            move = min(max(4, total_merges // 40), c_room, budgets[donor] - MIN_PER_LANG)
            if move <= 0:
                break  # cannot satisfy constraint even at full budget; best-effort
            budgets[constraint_lang] += move
            budgets[donor] -= move

    tok, per_lang, agg = evaluate(budgets)
    best = (score_key(agg, per_lang), dict(budgets), tok, dict(per_lang), dict(agg))

    # All-pairs hill-climb with a shrinking step. Each move shifts `step` merges
    # from one language to another (sum preserved => vocab stays full) and is kept
    # only if the score improves. The score enforces the English constraint, so a
    # move that would push English over 1.2 is never accepted. Trying every
    # donor/recipient pair avoids the local optima a min/max-only move gets stuck in.
    step = max(4, total_merges // 20)
    while step >= 2:
        improved = True
        while improved:
            improved = False
            _, budgets, _, per_lang, _ = best
            budgets = dict(budgets)
            order = sorted(langs, key=lambda l: per_lang[l]["X"], reverse=True)
            for recipient in order:                       # worst ratios first: they need merges
                if budgets[recipient] >= min(cap, max_useful[recipient]):
                    continue
                for donor in reversed(order):             # take from best ratios first
                    if donor == recipient or budgets[donor] - step < MIN_PER_LANG:
                        continue
                    trial = dict(budgets)
                    trial[recipient] = min(cap, max_useful[recipient], trial[recipient] + step)
                    trial[donor] -= step
                    t_tok, t_pl, t_agg = evaluate(trial)
                    if score_key(t_agg, t_pl) > best[0]:
                        best = (score_key(t_agg, t_pl), dict(trial), t_tok, dict(t_pl), dict(t_agg))
                        improved = True
                        break
                if improved:
                    break
        step //= 2

    _, budgets, tok, per_lang, agg = best
    # Report on the full corpus even when the search ran on a sample.
    if search_corpora is not corpora:
        per_lang, agg = metrics_fn(tok, corpora, langs)
    return tok, per_lang, agg


def _count_vocab_used(tok, text):
    return len(set(tok.encode(text)))


def verify_no_unk(tok, corpora):
    """Count <unk> ids emitted when encoding each corpus. Hard requirement: all 0.

    Holds by construction: every code point in the corpus is seeded into the base
    vocabulary, and encode decomposes any unknown cluster into code points before it
    could reach <unk>. This function checks that guarantee on the real data.
    """
    return {lang: sum(1 for i in tok.encode(text) if i == tok.unk_id)
            for lang, text in corpora.items()}


def _is_fragment(token):
    """A token that begins mid-syllable (starts with a combining mark or joiner)."""
    body = token.replace(BOUNDARY, "")
    if not body:
        return False
    c0 = body[0]
    return unicodedata.category(c0) in ("Mn", "Mc", "Me") or c0 in (ZWJ, ZWNJ)


def _fragment_count(tok):
    return sum(1 for t in tok.vocab if t not in SPECIAL_TOKENS and t != BOUNDARY and _is_fragment(t))


def _boundary_integrity(corpora, budget=1200, style=GPT4):
    """Compare fragment-token counts: naive code-point BPE vs. akshara-aware BPE.

    Demonstrates the correctness win — naive BPE spends many vocab entries on bare
    matras/anusvara/nukta that begin mid-syllable. Trains throwaway per-language
    tokenizers by temporarily disabling grapheme clustering.
    """
    result = {"naive": {}, "akshara": {}, "budget": budget}
    original = _bpe.grapheme_clusters
    for lang in ("hi", "te", "bn"):
        if lang not in corpora:
            continue
        akshara = Tokenizer(pretok_style=style)
        akshara.train([corpora[lang]], budget)
        result["akshara"][lang] = _fragment_count(akshara)
        try:
            _bpe.grapheme_clusters = lambda w: list(w)  # naive: one code point per unit
            naive = Tokenizer(pretok_style=style)
            naive.train([corpora[lang]], budget)
            result["naive"][lang] = _fragment_count(naive)
        finally:
            _bpe.grapheme_clusters = original
    return result


def _example_word(text, target_len=6):
    """Pick a representative word: the alphabetic word closest to target_len chars."""
    words = [w for w in re.findall(r"\S+", text) if any(ch.isalpha() for ch in w)]
    if not words:
        return ""
    return min(words, key=lambda w: (abs(len(w) - target_len), w))


def load_faithful_corpora():
    """Load the faithful Markdown corpus (corpus/<lang>.faithful.txt); build it first
    with `python3 fetch_faithful.py` if missing."""
    from fetch_faithful import PAGES, CORPUS_DIR
    corpora, missing = {}, []
    for lang in PAGES:
        path = os.path.join(CORPUS_DIR, f"{lang}.faithful.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                corpora[lang] = f.read()
        else:
            missing.append(path)
    if missing:
        raise SystemExit(
            "Missing faithful corpus files:\n  " + "\n  ".join(missing)
            + "\n\nBuild the corpus first:\n"
            "  pip install -r requirements-corpus.txt\n"
            "  python3 fetch_faithful.py")
    return corpora


def run(corpora, tokenizer_path="tokenizer.json", report_path="index.html", rounds=5,
        shared=True, style=WHITESPACE, weights=None):
    """Build the shipped tokenizer + widget.

    Defaults to the faithful-Markdown build: one **shared BPE** trained over the
    weighted concatenation of all corpora (metaspace/WHITESPACE pre-tokenization). This
    reaches the lowest absolute fertility — matching the sibling HuggingFace tokenizer
    — while staying akshara-aware and fully reversible. Pass `shared=False` to instead
    run the spread-optimizing per-language hill-climb (higher build score, higher
    fertility).
    """
    weights = weights or TRAIN_WEIGHTS
    if shared:
        tok, per_lang, agg = build_shared_tokenizer(corpora, style=style, weights=weights)
    else:
        tok, per_lang, agg = build_tokenizer(corpora, rounds=rounds,
                                             objective=FAITHFUL_OBJECTIVE, style=style,
                                             search_sample=0.25)
    for lang in per_lang:
        per_lang[lang]["vocab_used"] = _count_vocab_used(tok, corpora[lang])

    # Hard requirement: no <unk> on any corpus text. Verified here on the real data.
    unk_report = verify_no_unk(tok, corpora)
    total_unk = sum(unk_report.values())
    if total_unk:
        raise AssertionError(f"tokenizer emitted {total_unk} <unk> on corpus: {unk_report}")

    tokenizer_dict = tok.to_dict()
    # Virama code points present, so the in-browser tokenizer can reproduce the
    # exact akshara segmentation (it has no Unicode name database).
    tokenizer_dict["viramas"] = sorted(t for t in tok.vocab if len(t) == 1 and _is_virama(t))
    bs = agg["build_score"]
    adj = agg.get("hindi_adjusted_score", bs)
    tokenizer_dict["metrics"] = {
        **{l: per_lang[l] for l in per_lang},
        "build_score": None if bs == float("inf") else bs,
        "X_max": agg["X_max"],
        "X_min": agg["X_min"],
        "spread": agg["spread"],
        "sorted_X": agg["sorted_X"],
        "hindi_penalty": agg.get("hindi_penalty"),
        "hindi_adjusted_score": None if adj == float("inf") else adj,
        # Fertility is tokens per faithful unit, measured on the full faithful corpus.
        "evaluated_on": "faithful_markdown_corpus",
        "unit_policy": "one contiguous letter/mark/number run, or one visible symbol",
        "build": "shared_bpe" if shared else "hill_climb",
        "train_weights": weights if shared else None,
        "seed_min_freq": SHARED_SEED_MIN_FREQ if shared else CLUSTER_MIN_FREQ,
        "unk_on_corpus": total_unk,   # hard requirement: must be 0
    }

    lang_names = {l: LANGUAGES[l]["name"] if l in LANGUAGES else l for l in per_lang}

    # A representative example word per language (a mid-length word), with its tokens.
    examples = {}
    for l in per_lang:
        example = _example_word(corpora[l])
        toks = tok.tokenize(example) if example else []
        examples[l] = {"word": example, "tokens": toks, "n_tokens": len(toks)}

    # Visualization data: seeded-akshara frequencies and the boundary-integrity study.
    freq = cluster_frequencies(list(corpora.values()), style=style)
    viz = {
        "seeded_freq": {c: freq.get(c, 0) for c in tok.seeded_clusters},
        "boundary": _boundary_integrity(corpora, style=style),
    }
    # The actual implementation source, shown read-only in the widget.
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe.py"),
              encoding="utf-8") as f:
        code = f.read()

    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_dict, f, ensure_ascii=False, indent=2)
    html = render_report(per_lang, agg, tokenizer_dict, lang_names,
                         examples=examples, viz=viz, code=code)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return tok, per_lang, agg


def main():
    corpora = load_faithful_corpora()
    tok, per_lang, agg = run(corpora)
    adj = agg.get("hindi_adjusted_score", agg["build_score"])
    print("Vocab size:", len(tok.vocab))
    print("Build score (Hindi-adjusted):", "inf" if adj == float("inf") else round(adj, 2))
    print(f"Spread: {agg['spread']}  Hindi penalty: {round(agg.get('hindi_penalty', 1.0), 4)}")
    for l, m in per_lang.items():
        print(f"  {l}: fertility={m['X']} units={m['units']} tokens={m['tokens']} "
              f"vocab_used={m['vocab_used']}")
    unk = verify_no_unk(tok, corpora)
    print(f"<unk> on corpus: {sum(unk.values())} {unk}")


if __name__ == "__main__":
    main()
