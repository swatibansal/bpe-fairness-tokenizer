"""Critical robustness tests — ML best-practice properties a tokenizer must satisfy.

Run: python -m unittest tests.test_critical -v  (or the whole suite via discover).

These are deliberately adversarial: determinism, empty/whitespace, Unicode
normalization, vocabulary integrity, byte-fallback coverage of arbitrary code points,
pathological inputs, round-trip idempotence, and serialization exactness.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import Tokenizer, normalize, visible_chars, BOUNDARY, SPECIAL_TOKENS, BYTE_TOKENS

# A small mixed-script tokenizer shared across tests (trained once, cheap).
TRAIN_TEXT = ("India भारत ভারত భారత దేశం is a great country. देश। café's data "
              "[link](https://en.wikipedia.org/wiki/India) 1,428,627,663 | table | ") * 8
TOK = Tokenizer(pretok_style="whitespace")
TOK.train([TRAIN_TEXT], num_merges=150)


def faithful(tok, text):
    """decode(encode(text)) preserves every visible (non-whitespace) char of NFC(text)."""
    return visible_chars(tok.decode(tok.encode(text))) == visible_chars(normalize(text))


class TestDeterminism(unittest.TestCase):
    def test_encode_is_deterministic(self):
        s = "India भारत 2024 café @x"
        self.assertEqual(TOK.encode(s), TOK.encode(s))

    def test_reload_is_deterministic(self):
        t2 = Tokenizer.from_dict(TOK.to_dict())
        self.assertEqual(t2.encode("भारत's #tag 中文"), TOK.encode("भारत's #tag 中文"))


class TestEmptyAndWhitespace(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(TOK.encode(""), [])
        self.assertEqual(TOK.decode([]), "")

    def test_whitespace_only(self):
        for s in [" ", "   ", "\n\t ", "\r\n   "]:
            self.assertTrue(faithful(TOK, s))   # no visible chars to lose

    def test_decode_empty_and_unknown_ids(self):
        self.assertEqual(TOK.decode([]), "")
        # ids outside the table decode to nothing, never crash
        self.assertIsInstance(TOK.decode([10 ** 9]), str)


class TestVocabIntegrity(unittest.TestCase):
    def test_ids_contiguous_and_unique(self):
        ids = sorted(TOK.vocab.values())
        self.assertEqual(ids, list(range(len(TOK.vocab))))

    def test_id_to_token_is_bijective(self):
        self.assertEqual(len(TOK.id_to_token), len(TOK.vocab))
        for t, i in TOK.vocab.items():
            self.assertEqual(TOK.id_to_token[i], t)

    def test_all_merge_results_in_vocab(self):
        for a, b in TOK.merges:
            self.assertIn(a + b, TOK.vocab)

    def test_specials_and_all_bytes_present(self):
        for t in SPECIAL_TOKENS + BYTE_TOKENS:
            self.assertIn(t, TOK.vocab)
        self.assertEqual(len(BYTE_TOKENS), 256)

    def test_encoded_ids_in_range(self):
        ids = TOK.encode(TRAIN_TEXT + " @#emoji😀 中文")
        self.assertTrue(all(0 <= i < len(TOK.vocab) for i in ids))


class TestUnicodeNormalization(unittest.TestCase):
    def test_nfc_nfd_equivalent(self):
        self.assertEqual(TOK.encode("café"), TOK.encode("café"))  # composed vs decomposed
        self.assertEqual(TOK.encode("क़"), TOK.encode("क़"))          # nukta composed vs not

    def test_normalization_idempotent(self):
        s = "café क़"
        self.assertEqual(normalize(normalize(s)), normalize(s))


class TestByteFallbackRobustness(unittest.TestCase):
    def test_no_unk_and_faithful_on_arbitrary_chars(self):
        samples = ["@", "email@example.com", "#cite_ref-1", "a_b-c",
                   "中文 日本語 한국어", "😀🇮🇳👍🏽", "👨‍👩‍👧‍👦",
                   "Ω≈ç√∫˜µ≤≥÷", "\x00\x01\x07 control", "\\ back \\ slash"]
        for s in samples:
            ids = TOK.encode(s)
            self.assertNotIn(TOK.unk_id, ids, f"<unk> on {s!r}")
            self.assertTrue(faithful(TOK, s), f"not faithful on {s!r}")

    def test_fuzz_across_unicode_blocks(self):
        # One char from many blocks (skip lone surrogates, which are not valid UTF-8).
        cps = list(range(0x20, 0x7F)) + [0xA9, 0xE9, 0x394, 0x4E2D, 0x0905, 0x0C05,
                                         0x09A4, 0x1F600, 0x2603, 0x20AC, 0x2581 + 1]
        for cp in cps:
            s = "x" + chr(cp) + "y"
            ids = TOK.encode(s)
            self.assertNotIn(TOK.unk_id, ids, f"<unk> on U+{cp:04X}")
            self.assertTrue(faithful(TOK, s), f"not faithful on U+{cp:04X}")


class TestPathologicalInputs(unittest.TestCase):
    def test_long_and_repeated(self):
        self.assertTrue(faithful(TOK, "a" * 50000))
        self.assertTrue(faithful(TOK, ("भ" * 3000) + " " + ("x" * 3000)))

    def test_punctuation_and_symbol_runs(self):
        self.assertTrue(faithful(TOK, "...,,,;;;!!!???@#$%^&*()_+-=[]{}|\\/<>~`"))

    def test_combining_marks_and_joiners(self):
        for s in ["िी́", "क्‍ष", "‍‌"]:
            self.assertTrue(faithful(TOK, s))

    def test_mixed_everything(self):
        s = "## भारत\n\n[India](https://x.org/भारत#a_b) 😀 costs $5 & ₹10; 中文 café's."
        self.assertNotIn(TOK.unk_id, TOK.encode(s))
        self.assertTrue(faithful(TOK, s))


class TestRoundTripIdempotence(unittest.TestCase):
    def test_decode_is_idempotent_on_its_own_output(self):
        # decode normalizes whitespace, so re-encoding its output must be stable.
        for s in ["India  is   a\tcountry.", "भारत   \n\n  देश।", "a,b, c  d"]:
            y = TOK.decode(TOK.encode(s))
            self.assertEqual(TOK.decode(TOK.encode(y)), y)

    def test_encode_decode_encode_stable(self):
        for s in ["India's data", "भारत 2024", "@x #y 中"]:
            y = TOK.decode(TOK.encode(s))
            self.assertEqual(TOK.encode(TOK.decode(TOK.encode(y))), TOK.encode(y))


class TestSerializationExact(unittest.TestCase):
    def test_roundtrip_preserves_everything(self):
        t2 = Tokenizer.from_dict(TOK.to_dict())
        self.assertEqual(t2.pretok_style, TOK.pretok_style)
        self.assertEqual(t2.seeded_clusters, TOK.seeded_clusters)
        self.assertEqual(t2.merges, TOK.merges)
        for s in ["@x 中 😀 भारत's", "https://x.org/a#b_c", ""]:
            self.assertEqual(t2.encode(s), TOK.encode(s))
            self.assertEqual(t2.decode(t2.encode(s)), TOK.decode(TOK.encode(s)))


class TestKnownLimitation(unittest.TestCase):
    """The metaspace marker U+2581 (▁) cannot appear literally in content and also serve
    as the word-boundary marker — the standard SentencePiece/Metaspace caveat. It does
    not occur in the faithful-Markdown evaluation domain. This test pins the behavior so
    it is tracked rather than a silent surprise."""

    def test_literal_boundary_marker_maps_to_space(self):
        s = "a" + BOUNDARY + "b"
        # The literal ▁ is rendered as a space on decode (documented limitation).
        self.assertEqual(TOK.decode(TOK.encode(s)), "a b")
        # Every OTHER visible character is still preserved.
        self.assertNotIn(TOK.unk_id, TOK.encode(s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
