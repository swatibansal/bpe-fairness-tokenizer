"""Unit tests (stdlib unittest, no external deps). Run: python -m unittest -v"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unicodedata

from bpe import (pre_tokenize, train_merges, Tokenizer, BOUNDARY, grapheme_clusters,
                 normalize, seeded_clusters_from, word_to_atoms, visible_chars,
                 GPT4, WHITESPACE)
from metrics import (language_metrics, aggregate_metrics,
                     faithful_units, faithful_language_metrics, faithful_aggregate)
from fetch_data import clean_text, take_words, LANGUAGES
from report_template import render_report
from train import build_tokenizer, run, VOCAB_SIZE


class TestPreTokenize(unittest.TestCase):
    def test_splits_words_with_boundary_marker(self):
        self.assertEqual(
            pre_tokenize("hola mundo"),
            [["▁", "h", "o", "l", "a"], ["▁", "m", "u", "n", "d", "o"]],
        )

    def test_segments_indic_into_aksharas(self):
        # भारत = भ+ा(matra), र, त  -> matra binds to its consonant, not a free token.
        self.assertEqual(pre_tokenize("भारत"), [["▁", "भा", "र", "त"]])

    def test_empty_string(self):
        self.assertEqual(pre_tokenize(""), [])


class TestHybridPreTokenize(unittest.TestCase):
    """Every word follows GPT-4-style splitting; grapheme clustering runs first, so
    matras/conjuncts stay inside their akshara while separate punctuation (the danda,
    commas, brackets) splits off — needed for faithful Markdown."""

    def test_latin_letters_only_word_is_one_chunk(self):
        self.assertEqual(pre_tokenize("hola"), [["▁", "h", "o", "l", "a"]])

    def test_trailing_punctuation_splits(self):
        self.assertEqual(pre_tokenize("India."), [["▁", "I", "n", "d", "i", "a"], ["."]])

    def test_one_leading_punctuation_attaches(self):
        self.assertEqual(
            pre_tokenize("(India)"),
            [["▁", "(", "I", "n", "d", "i", "a"], [")"]],
        )

    def test_contraction_splits_off(self):
        self.assertEqual(
            pre_tokenize("India's"),
            [["▁", "I", "n", "d", "i", "a"], ["'", "s"]],
        )

    def test_digits_group_in_threes(self):
        self.assertEqual(pre_tokenize("2024"), [["▁", "2", "0", "2"], ["4"]])
        self.assertEqual(pre_tokenize("1,234"), [["▁", "1"], [","], ["2", "3", "4"]])

    def test_brahmic_danda_splits_off_but_aksharas_stay(self):
        # देश stays as its aksharas; the danda "।" splits into its own chunk.
        self.assertEqual(pre_tokenize("देश।"), [["▁", "दे", "श"], ["।"]])

    def test_brahmic_trailing_comma_splits_off(self):
        self.assertEqual(pre_tokenize("गांधी,"), [["▁", "गां", "धी"], [","]])

    def test_roundtrip_mixed_scripts_and_punctuation(self):
        tok = Tokenizer()
        text = "India (2024): देश। \"quote,\" Gandhi's"
        tok.train([text], num_merges=30)
        self.assertEqual(tok.decode(tok.encode(text)), normalize(text))


class TestGraphemeClusters(unittest.TestCase):
    def test_matra_binds_to_consonant(self):
        # हि = ह + ि(i-matra) stays as one cluster.
        self.assertEqual(grapheme_clusters("हि"), ["हि"])

    def test_conjunct_stays_together(self):
        # क्ष = क + virama + ष is a single conjunct cluster.
        self.assertEqual(grapheme_clusters("क्ष"), ["क्ष"])
        # न्दी = न + virama + द + ी(ii-matra): one conjunct cluster.
        self.assertEqual(grapheme_clusters("हिन्दी"), ["हि", "न्दी"])

    def test_telugu_matra_binds(self):
        self.assertEqual(grapheme_clusters("భారత"), ["భా", "ర", "త"])

    def test_latin_is_one_cluster_per_letter(self):
        self.assertEqual(grapheme_clusters("hola"), ["h", "o", "l", "a"])

    def test_zwj_stays_within_cluster(self):
        # A ZWJ between virama and consonant must not become its own token.
        w = "क" + "्" + "‍" + "ष"   # ka, virama, ZWJ, ssa
        self.assertEqual(grapheme_clusters(w), [w])

    def test_normalize_makes_decomposed_equal_precomposed(self):
        precomposed = "क़"          # क़ (single code point)
        decomposed = "क़"     # क + nukta
        self.assertEqual(normalize(precomposed), normalize(decomposed))
        self.assertEqual(pre_tokenize(precomposed), pre_tokenize(decomposed))

    def test_indic_roundtrip_and_fewer_tokens_than_codepoints(self):
        tok = Tokenizer()
        text = "भारत एक विशाल देश है भारत"
        tok.train([text], num_merges=20)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), normalize(text))


class TestFrequencyThreshold(unittest.TestCase):
    def test_seeds_only_frequent_aksharas(self):
        # कि appears twice, कु once -> only कि earns a base slot at min_freq=2.
        seeded = seeded_clusters_from(["कि कि कु"], min_freq=2)
        self.assertIn("कि", seeded)
        self.assertNotIn("कु", seeded)

    def test_word_to_atoms_expands_unseeded_cluster(self):
        seeded = {"कि"}
        # Seeded cluster stays whole; unseeded कु expands to its code points.
        self.assertEqual(word_to_atoms(["कि"], seeded), ["कि"])
        self.assertEqual(word_to_atoms(["कु"], seeded), list("कु"))

    def test_serialize_preserves_seeded_and_roundtrips_indic(self):
        tok = Tokenizer()
        text = "भारत भारत भारत देश देश विशाल"
        tok.train([text], num_merges=15)
        tok2 = Tokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok2.seeded_clusters, tok.seeded_clusters)
        self.assertEqual(tok2.encode(text), tok.encode(text))


class TestTrainMerges(unittest.TestCase):
    def test_learns_most_frequent_pair(self):
        merges = train_merges(["ab", "ab", "ab", "ac"], num_merges=1)
        # First merge joins boundary+a (most frequent leading pair) or a+b.
        self.assertEqual(len(merges), 1)

    def test_respects_budget(self):
        self.assertLessEqual(len(train_merges(["banana", "banana"], num_merges=3)), 3)

    def test_zero_budget(self):
        self.assertEqual(train_merges(["abc"], num_merges=0), [])

    def test_incremental_matches_bruteforce(self):
        # The incremental pair-count optimization must pick exactly the same merges,
        # in the same order, as a full-rescan reference (same freq/lexicographic tie).
        from collections import Counter
        from bpe import pre_tokenize, word_to_atoms, _get_pairs, _merge_word

        def brute_force(corpus, num_merges, seeded=None):
            word_freqs = Counter()
            for text in corpus:
                for word in pre_tokenize(text):
                    word_freqs[tuple(word_to_atoms(word, seeded))] += 1
            words = {w: list(w) for w in word_freqs}
            merges = []
            for _ in range(num_merges):
                counts = Counter()
                for w, freq in word_freqs.items():
                    for p in _get_pairs(words[w]):
                        counts[p] += freq
                if not counts:
                    break
                best = max(counts, key=lambda p: (counts[p], p))
                for w in words:
                    words[w] = _merge_word(words[w], best)
                merges.append(best)
            return merges

        corpus = ["India भारत ভারত భారత దేశం is a great country. " * 12,
                  "banana bandana भारत का इतिहास देश। " * 12]
        for n in (5, 50, 200):
            self.assertEqual(train_merges(corpus, n), brute_force(corpus, n))


class TestTokenizer(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        tok = Tokenizer()
        tok.train(["banana banana bandana"], num_merges=10)
        self.assertEqual(tok.decode(tok.encode("banana")), "banana")

    def test_encode_ids_within_vocab(self):
        tok = Tokenizer()
        tok.train(["hola mundo"], num_merges=5)
        self.assertTrue(all(0 <= i < len(tok.vocab) for i in tok.encode("hola")))

    def test_serialize_roundtrip(self):
        tok = Tokenizer()
        tok.train(["hola mundo hola"], num_merges=5)
        tok2 = Tokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok2.encode("hola"), tok.encode("hola"))

    def test_unknown_char_round_trips_via_bytes(self):
        # An unseen character must NOT become <unk>; byte fallback keeps it faithful.
        tok = Tokenizer()
        tok.train(["abc"], num_merges=2)
        ids = tok.encode("z")
        self.assertNotIn(tok.unk_id, ids)
        self.assertEqual(tok.decode(ids), "z")

    def test_count_tokens_matches_encode_length(self):
        text = "India's 1,428,627,663 people. भारत देश। café | cell | http://x.org/India"
        for style in (GPT4, WHITESPACE):
            tok = Tokenizer(pretok_style=style)
            tok.train([text], num_merges=40)
            self.assertEqual(tok.count_tokens(text), len(tok.encode(text)))


class TestMetrics(unittest.TestCase):
    def test_language_metrics_ratio(self):
        class FakeTok:
            def encode(self, text):
                return [0, 1, 2] if text == "one two" else []
        m = language_metrics(FakeTok(), "one two")
        self.assertEqual((m["words"], m["tokens"], m["X"]), (2, 3, 1.5))
        self.assertFalse(m["within_1_2"])

    def test_aggregate_build_score(self):
        per_lang = {"en": {"X": 1.0}, "hi": {"X": 1.2}, "te": {"X": 1.1}, "es": {"X": 1.05}}
        agg = aggregate_metrics(per_lang)
        self.assertEqual(agg["X_max"], 1.2)
        self.assertEqual(agg["X_min"], 1.0)
        self.assertAlmostEqual(agg["build_score"], 1000 / 0.2)

    def test_aggregate_zero_spread(self):
        agg = aggregate_metrics({"en": {"X": 1.1}, "es": {"X": 1.1}})
        self.assertEqual(agg["build_score"], float("inf"))


class TestNoUnk(unittest.TestCase):
    MULTI = "India भारत ভারত భారత దేశం 2024 (a, b). देश। café"

    def test_trained_tokenizer_never_unks_its_corpus(self):
        # Every code point in the training text is seeded, so encode must not <unk>.
        tok = Tokenizer()
        tok.train([self.MULTI], num_merges=20)
        ids = tok.encode(self.MULTI)
        self.assertNotIn(tok.unk_id, ids)
        self.assertEqual(tok.decode(ids), normalize(self.MULTI))

    def test_verify_no_unk_reports_zero_for_corpus(self):
        from train import build_tokenizer, verify_no_unk
        corpora = {
            "en": "the quick brown fox " * 20,
            "hi": "भारत एक देश है " * 20,
            "te": "భారత దేశం గొప్ప " * 20,
            "bn": "ভারত একটি দেশ " * 20,
        }
        tok, _, _ = build_tokenizer(corpora, rounds=1)
        self.assertEqual(verify_no_unk(tok, corpora), {"en": 0, "hi": 0, "te": 0, "bn": 0})

    def test_unseen_codepoints_use_byte_fallback_not_unk(self):
        # Characters never present in the training corpus round-trip via UTF-8 byte
        # fallback (the faithful-tokenizer requirement), never <unk>.
        tok = Tokenizer()
        tok.train(["abc"], num_merges=2)
        for s in ["z", "@", "#cite_ref-1", "café", "中文", "😀", "https://x.org/a#b_c"]:
            ids = tok.encode(s)
            self.assertNotIn(tok.unk_id, ids)
            self.assertEqual(tok.decode(ids), normalize(s))


class TestFetchHelpers(unittest.TestCase):
    def test_clean_text(self):
        cleaned = clean_text("India[1] is a  country.\n\nSee also")
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("  ", cleaned)

    def test_take_words_limits(self):
        text = " ".join(["word"] * 10000)
        self.assertEqual(len(take_words(text, 5000).split()), 5000)

    def test_languages_config(self):
        self.assertEqual(set(LANGUAGES.keys()), {"en", "hi", "te", "bn"})

    def test_accumulate_skips_missing_and_caps_at_limit(self):
        from fetch_data import accumulate_extracts
        extracts = {"A": None, "B": "x " * 10, "C": "y " * 10}
        out = accumulate_extracts(["A", "B", "C"], lambda t: extracts[t], limit=15)
        words = out.split()
        self.assertEqual(len(words), 15)          # A skipped; B(10)+C(10) capped to 15
        self.assertTrue(set(words) <= {"x", "y"})


class TestTrain(unittest.TestCase):
    CORPORA = {
        "en": "the quick brown fox jumps over the lazy dog " * 20,
        "bn": "ভারত একটি দেশ ভারত মহান দেশ " * 20,
        "hi": "भारत एक देश है भारत महान है " * 20,
        "te": "భారత దేశం గొప్ప దేశం భారత " * 20,
    }

    def test_respects_total_vocab(self):
        tok, per_lang, agg = build_tokenizer(self.CORPORA, rounds=2)
        self.assertLessEqual(len(tok.vocab), VOCAB_SIZE)
        self.assertEqual(set(per_lang.keys()), set(self.CORPORA.keys()))
        self.assertIn("build_score", agg)

    def test_reports_ratios(self):
        _, per_lang, _ = build_tokenizer(self.CORPORA, rounds=1)
        for m in per_lang.values():
            self.assertGreater(m["X"], 0)

    def test_run_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            tj = os.path.join(d, "tokenizer.json")
            rep = os.path.join(d, "index.html")
            run(self.CORPORA, tokenizer_path=tj, report_path=rep, rounds=2)
            self.assertTrue(os.path.exists(tj) and os.path.exists(rep))
            with open(tj, encoding="utf-8") as fh:
                data = json.loads(fh.read())
            self.assertEqual(data["version"], 2)
            self.assertIn("metrics", data)


class TestReport(unittest.TestCase):
    def test_contains_build_score_and_table(self):
        per_lang = {"en": {"words": 100, "tokens": 110, "X": 1.1, "within_1_2": True, "vocab_used": 500}}
        agg = {"X_max": 1.1, "X_min": 1.1, "spread": 0.0, "sorted_X": [1.1], "build_score": float("inf")}
        html = render_report(per_lang, agg, {"version": 1, "vocab": {"a": 0}, "merges": []}, {"en": "English"})
        self.assertIn("build score", html.lower())
        self.assertIn("English", html)
        self.assertIn("Export tokenizer JSON", html)

    def test_embeds_valid_json(self):
        td = {"version": 1, "vocab": {"a": 0}, "merges": [["a", "b"]]}
        agg = {"build_score": 5.0, "sorted_X": [], "X_max": 0, "X_min": 0, "spread": 0}
        html = render_report({}, agg, td, {})
        start = html.index('id="tokenizer-data">') + len('id="tokenizer-data">')
        end = html.index("</script>", start)
        self.assertEqual(json.loads(html[start:end].strip())["version"], 1)


class TestFaithfulMetric(unittest.TestCase):
    """Faithful units: a run of letter/mark/number chars is one unit; each other
    visible char is its own unit (stdlib reproduction of the sibling repo's regex)."""

    def test_counts_letter_runs_and_symbols(self):
        self.assertEqual(faithful_units("India's"), 3)      # India | ' | s
        self.assertEqual(faithful_units("1,234"), 3)        # 1 | , | 234
        self.assertEqual(faithful_units("भारत देश"), 2)     # two letter runs
        self.assertEqual(faithful_units("[a](b)"), 6)       # [ a ] ( b )

    def test_whitespace_is_not_a_unit(self):
        self.assertEqual(faithful_units("  a\n\nb  "), 2)

    def test_faithful_language_metrics_ratio(self):
        class FakeTok:
            def encode(self, text):
                return [0, 1, 2, 3]
        m = faithful_language_metrics(FakeTok(), "a b")   # 2 units, 4 tokens
        self.assertEqual((m["units"], m["tokens"], m["X"]), (2, 4, 2.0))

    def test_aggregate_score_no_penalty_below_threshold(self):
        per_lang = {"en": {"X": 0.6}, "hi": {"X": 0.8}, "te": {"X": 0.7}, "bn": {"X": 0.75}}
        agg = faithful_aggregate(per_lang)
        self.assertAlmostEqual(agg["spread"], 0.2)
        self.assertAlmostEqual(agg["score"], 1000 / 0.2)
        self.assertEqual(agg["hindi_penalty"], 1.0)
        self.assertAlmostEqual(agg["hindi_adjusted_score"], 1000 / 0.2)

    def test_hindi_penalty_applies_above_threshold(self):
        import math
        per_lang = {"en": {"X": 1.0}, "hi": {"X": 1.8}, "te": {"X": 1.2}, "bn": {"X": 1.4}}
        agg = faithful_aggregate(per_lang)
        self.assertAlmostEqual(agg["hindi_penalty"], math.exp(1.8 / 1.2 - 1.0))
        self.assertLess(agg["hindi_adjusted_score"], agg["score"])


class TestFaithfulness(unittest.TestCase):
    """Full visible-character faithfulness on markdown-flavored, mixed-script text."""

    SAMPLE = ("## India\n\n[भारत](https://en.wikipedia.org/wiki/India) has "
              "1,428,627,663 people. देश। café's data | cell |")

    def test_visible_chars_preserved_round_trip(self):
        tok = Tokenizer()
        tok.train([self.SAMPLE], num_merges=60)
        decoded = tok.decode(tok.encode(self.SAMPLE))
        self.assertEqual(visible_chars(decoded), visible_chars(normalize(self.SAMPLE)))

    def test_no_unk_on_trained_markdown(self):
        tok = Tokenizer()
        tok.train([self.SAMPLE], num_merges=60)
        self.assertNotIn(tok.unk_id, tok.encode(self.SAMPLE))


class TestWhitespaceStyle(unittest.TestCase):
    """Metaspace-style pre-tokenization keeps each whitespace word whole, so BPE may
    merge across punctuation — still akshara-aware via grapheme clustering."""

    def test_latin_word_with_punctuation_kept_whole(self):
        self.assertEqual(pre_tokenize("India.", WHITESPACE),
                         [["▁", "I", "n", "d", "i", "a", "."]])

    def test_brahmic_word_with_danda_kept_whole(self):
        self.assertEqual(pre_tokenize("देश।", WHITESPACE), [["▁", "दे", "श", "।"]])

    def test_number_with_separators_kept_whole(self):
        # No digit grouping in whitespace style: the whole number stays one chunk.
        self.assertEqual(pre_tokenize("1,428", WHITESPACE), [["▁", "1", ",", "4", "2", "8"]])

    def test_default_style_is_gpt4(self):
        self.assertEqual(pre_tokenize("India."), pre_tokenize("India.", GPT4))

    def test_roundtrip_and_style_serialized(self):
        tok = Tokenizer(pretok_style=WHITESPACE)
        text = "India's 1,428,627,663 people. देश। café | cell |"
        tok.train([text], num_merges=40)
        self.assertEqual(visible_chars(tok.decode(tok.encode(text))),
                         visible_chars(normalize(text)))
        tok2 = Tokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok2.pretok_style, WHITESPACE)
        self.assertEqual(tok2.encode(text), tok.encode(text))

    def test_whitespace_gives_lower_fertility_than_gpt4(self):
        # On punctuation-heavy in-domain text, merging across punctuation compresses
        # more, so whitespace-style produces no more tokens than gpt4-style.
        text = "India's population is 1,428,627,663. " * 40
        g = Tokenizer(pretok_style=GPT4); g.train([text], num_merges=80)
        w = Tokenizer(pretok_style=WHITESPACE); w.train([text], num_merges=80)
        self.assertLess(len(w.encode(text)), len(g.encode(text)))


class TestFaithfulObjective(unittest.TestCase):
    CORPORA = {
        "en": "## India\n\nIndia is a country. See [India](x). " * 20,
        "hi": "## भारत\n\nभारत एक देश है। [भारत](x) " * 20,
        "te": "## భారత\n\nభారత దేశం గొప్ప దేశం. " * 20,
        "bn": "## ভারত\n\nভারত একটি দেশ, মহান দেশ। " * 20,
    }

    def test_build_with_faithful_objective(self):
        from train import FAITHFUL_OBJECTIVE
        tok, per_lang, agg = build_tokenizer(self.CORPORA, objective=FAITHFUL_OBJECTIVE)
        self.assertLessEqual(len(tok.vocab), VOCAB_SIZE)
        self.assertIn("hindi_adjusted_score", agg)
        for m in per_lang.values():
            self.assertIn("units", m)
            self.assertGreater(m["X"], 0)


class TestSharedBuilder(unittest.TestCase):
    """The shipped default: one shared BPE over the weighted concatenation."""

    CORPORA = TestFaithfulObjective.CORPORA

    def test_builds_faithful_shared_tokenizer(self):
        from train import build_shared_tokenizer
        tok, per_lang, agg = build_shared_tokenizer(
            self.CORPORA, weights={"en": 1, "hi": 1, "te": 1, "bn": 1}, seed_min_freq=2)
        self.assertLessEqual(len(tok.vocab), VOCAB_SIZE)
        self.assertEqual(tok.pretok_style, WHITESPACE)
        self.assertIn("hindi_adjusted_score", agg)
        # every merge's result is in the vocab (no dead ranks)
        for a, b in tok.merges:
            self.assertIn(a + b, tok.vocab)
        # faithful + no <unk> on each corpus
        for lang, text in self.CORPORA.items():
            self.assertIn("units", per_lang[lang])
            self.assertNotIn(tok.unk_id, tok.encode(text))
            self.assertEqual(visible_chars(tok.decode(tok.encode(text))),
                             visible_chars(normalize(text)))

    def test_weights_upweight_small_corpora(self):
        # A language given more weight gets more of the shared merges devoted to it,
        # i.e. its fertility is no worse than with equal weights.
        from train import build_shared_tokenizer
        eq, _, _ = build_shared_tokenizer(self.CORPORA, weights={l: 1 for l in self.CORPORA},
                                          seed_min_freq=2)
        self.assertGreater(len(eq.merges), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
