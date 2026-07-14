"""From-scratch Byte-Pair-Encoding tokenizer (standard library only).

Grapheme-aware (Brahmic-capable): Indic scripts are segmented into akshara-level
grapheme clusters rather than raw code points, so combining vowel signs (matras),
viramas/conjuncts, and ZWJ/ZWNJ stay bound to their base consonant. See the design
doc for the reference (BrahmicTokenizer-131K, arXiv 2605.29379).

Two pre-tokenization styles are supported (see `pre_tokenize`): WHITESPACE (metaspace —
each whitespace word is one chunk, the shipped default) and GPT4 (splits punctuation and
groups digits). The style is stored in `to_dict`/`from_dict` so encode/decode stay
consistent.

Faithful on ANY input: the base vocabulary holds all 256 byte tokens (see BYTE_TOKENS),
so a code point not seen in the corpus is emitted as its UTF-8 bytes rather than <unk>.
Consequently `decode(encode(text))` preserves every visible (non-whitespace) character of
`normalize(text)` for arbitrary text, not just corpus text — see `visible_chars`.
"""

import re
import unicodedata
from collections import Counter

BOUNDARY = "▁"  # ▁ marks the start of a word so spacing is reconstructable
SPECIAL_TOKENS = ["<pad>", "<unk>"]

# Byte fallback: every one of the 256 byte values gets a permanent token, so ANY code
# point — even one never seen in the corpus — round-trips as its UTF-8 bytes instead of
# collapsing to <unk>. This is what makes decode(encode(text)) faithful on arbitrary
# input (the requirement for a faithful-Markdown tokenizer), not just on corpus text.
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
_TOK_TO_BYTE = {t: b for b, t in enumerate(BYTE_TOKENS)}

ZWJ = "‍"   # zero-width joiner
ZWNJ = "‌"  # zero-width non-joiner

# A multi-codepoint akshara earns a permanent base-vocabulary slot only if it
# occurs at least this many times; rarer aksharas fall back to their code points,
# so scarce vocabulary is not spent on single-use units.
CLUSTER_MIN_FREQ = 2

# Pre-tokenization styles (how a whitespace word is chunked before BPE):
#   GPT4       - split contractions/punctuation off and group digits (robust token
#                boundaries; fertility floors near 1 token per faithful unit).
#   WHITESPACE - keep each whitespace word whole (Metaspace-style); merges may fuse
#                punctuation and span faithful-unit boundaries, so fertility can drop
#                well below 1 on in-domain text. Grapheme clustering still applies, so
#                this stays akshara-aware unlike a plain byte/char Metaspace.
GPT4 = "gpt4"
WHITESPACE = "whitespace"


def normalize(text):
    """Canonical Unicode form so precomposed/decomposed and nukta variants match."""
    return unicodedata.normalize("NFC", text)


def _is_virama(ch):
    """True for Brahmic virama/halant marks (they bind the next consonant)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "VIRAMA" in name or "HALANT" in name or "AL-LAKUNA" in name


def _ends_with_virama(cluster):
    """Does the cluster end in a virama, ignoring any trailing ZWJ/ZWNJ?

    Handles the explicit conjunct request `consonant + virama + ZWJ + consonant`,
    where the joiner sits between the virama and the consonant it binds.
    """
    j = len(cluster) - 1
    while j >= 0 and cluster[j] in (ZWJ, ZWNJ):
        j -= 1
    return j >= 0 and _is_virama(cluster[j])


def grapheme_clusters(word):
    """Split a word into akshara-level grapheme clusters.

    A cluster is a base character plus any trailing combining marks (Mn/Mc/Me:
    matras, anusvara, nukta, visarga), ZWJ/ZWNJ, and virama-driven conjuncts
    (consonant + virama + consonant ...). Latin letters end up as one cluster each.
    """
    clusters = []
    i, n = 0, len(word)
    while i < n:
        cluster = word[i]
        i += 1
        while i < n:
            ch = word[i]
            if unicodedata.category(ch) in ("Mn", "Mc", "Me") or ch in (ZWJ, ZWNJ):
                cluster += ch          # combining mark / joiner attaches to the cluster
            elif _ends_with_virama(cluster):
                cluster += ch          # preceding virama -> bind this consonant (conjunct)
            else:
                break
            i += 1
        clusters.append(cluster)
    return clusters


CONTRACTIONS = frozenset({"s", "d", "m", "t", "ll", "ve", "re"})
DIGIT_GROUP = 3  # max digits per token (GPT-4/o200k-style number grouping)


def _atom_class(cluster):
    """Coarse class of a grapheme cluster: 'd' digit, 'l' letter/akshara, 'o' other."""
    ch = cluster[0]
    if ch.isdigit():
        return "d"
    if ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc", "Me") or ch in (ZWJ, ZWNJ):
        return "l"
    return "o"


def _contraction_len(clusters, i):
    """If clusters[i] starts an English contraction ('s, 'll, ...), return the number
    of letter clusters after the apostrophe (1 or 2); otherwise 0."""
    if clusters[i] != "'":
        return 0
    for size in (2, 1):
        tail = clusters[i + 1:i + 1 + size]
        if (len(tail) == size and all(_atom_class(c) == "l" for c in tail)
                and "".join(tail).lower() in CONTRACTIONS):
            return size
    return 0


def _split_word(clusters):
    """GPT-4-style split of a word into chunks (each a list of grapheme clusters).

    Contractions and trailing punctuation split off, one leading punctuation attaches
    to a following letter run, and digit runs are grouped into <= DIGIT_GROUP. Merges
    never cross a chunk boundary, so e.g. "India." never fuses the period onto the word.

    This is the chunker for the GPT4 pre-tokenization style (the WHITESPACE style keeps
    the word whole instead). Grapheme clustering has already run, so matras and virama
    conjuncts sit inside a single 'l' cluster and never split; it is applied uniformly to
    every script, so only separate punctuation clusters (the danda "।", brackets, pipes,
    URL characters) break off.
    """
    chunks, i, n = [], 0, len(clusters)
    while i < n:
        cls = _atom_class(clusters[i])
        m = _contraction_len(clusters, i)
        if m:                                                 # 's, 'll, 've ...
            chunks.append(clusters[i:i + 1 + m]); i += 1 + m
        elif cls == "o":
            if i + 1 < n and _atom_class(clusters[i + 1]) == "l":   # one leading punct + word
                chunk = [clusters[i]]; i += 1
                while i < n and _atom_class(clusters[i]) == "l":
                    chunk.append(clusters[i]); i += 1
                chunks.append(chunk)
            else:                                             # punctuation / symbol run
                chunk = []
                while i < n and _atom_class(clusters[i]) == "o" and not _contraction_len(clusters, i):
                    chunk.append(clusters[i]); i += 1
                chunks.append(chunk)
        elif cls == "l":                                      # letter / akshara run
            chunk = []
            while i < n and _atom_class(clusters[i]) == "l":
                chunk.append(clusters[i]); i += 1
            chunks.append(chunk)
        else:                                                 # digit run, grouped <= 3
            run = []
            while i < n and _atom_class(clusters[i]) == "d":
                run.append(clusters[i]); i += 1
            for j in range(0, len(run), DIGIT_GROUP):
                chunks.append(run[j:j + DIGIT_GROUP])
    return chunks


def pre_tokenize(text, style=GPT4):
    """Normalize, split on whitespace, then chunk each word for BPE.

    Every word is segmented into akshara-level grapheme clusters. Under the GPT4 style
    each word is further chunked by the script-agnostic splitter (see _split_word:
    contractions/punctuation split off, digit runs grouped). Under the WHITESPACE style
    the word is kept whole as one chunk, so BPE may merge across punctuation. Either
    way matras and conjuncts stay bound inside their cluster, and only the first chunk
    of each whitespace word carries BOUNDARY so spacing is reconstructable on decode.
    """
    text = normalize(text)
    out = []
    for word in re.findall(r"\S+", text):
        clusters = grapheme_clusters(word)
        segments = [clusters] if style == WHITESPACE else _split_word(clusters)
        for k, seg in enumerate(segments):
            out.append(([BOUNDARY] + seg) if k == 0 else seg)
    return out


def visible_chars(text):
    """Non-whitespace characters in order — the unit of the faithfulness guarantee.

    A tokenizer is faithful on `text` when decode(encode(text)) has the same
    visible_chars as normalize(text): every visible character survives the round trip,
    with whitespace allowed to be normalized.
    """
    return [c for c in text if not c.isspace()]


def cluster_frequencies(corpus, style=GPT4):
    """Count occurrences of each multi-codepoint akshara across the corpus."""
    freq = Counter()
    for text in corpus:
        for word in pre_tokenize(text, style):
            for c in word:
                if len(c) > 1:
                    freq[c] += 1
    return freq


def seeded_clusters_from(corpus, min_freq=CLUSTER_MIN_FREQ, style=GPT4):
    """Multi-codepoint aksharas frequent enough to earn a base-vocabulary slot."""
    return {c for c, f in cluster_frequencies(corpus, style).items() if f >= min_freq}


def word_to_atoms(clusters, seeded):
    """Turn a word's grapheme clusters into BPE atoms.

    A cluster stays whole if it is a single code point or a seeded akshara;
    otherwise it decomposes into its code points. `seeded=None` keeps every
    cluster whole (used when no frequency threshold applies).
    """
    if seeded is None:
        return list(clusters)
    atoms = []
    for c in clusters:
        if len(c) == 1 or c in seeded:
            atoms.append(c)
        else:
            atoms.extend(c)
    return atoms


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


def train_merges(corpus, num_merges, seeded=None, style=GPT4):
    """Learn up to num_merges BPE merges from a list of raw text strings.

    Words are reduced to BPE atoms (grapheme clusters, with rare aksharas decomposed
    per `seeded`) and counted by frequency. Pair counts are maintained incrementally:
    after each merge only the words that actually contained the merged pair are
    rescored, instead of rescanning the whole corpus every step. This yields the same
    merges (same frequency-then-lexicographic tie-break) far faster on large corpora.
    """
    word_freqs = Counter()
    for text in corpus:
        for word in pre_tokenize(text, style):
            word_freqs[tuple(word_to_atoms(word, seeded))] += 1

    words = [list(w) for w in word_freqs]        # mutable token list per unique word
    freqs = [word_freqs[w] for w in word_freqs]  # its corpus frequency

    # pair_counts[p] = total frequency of adjacent pair p; where[p] = word indices with p.
    pair_counts = Counter()
    where = {}
    for i, toks in enumerate(words):
        f = freqs[i]
        for p in _get_pairs(toks):
            pair_counts[p] += f
            where.setdefault(p, set()).add(i)

    def _remove(i):
        f = freqs[i]
        for p in _get_pairs(words[i]):
            pair_counts[p] -= f
            if pair_counts[p] <= 0:
                del pair_counts[p]
                where.pop(p, None)
            elif p in where:
                where[p].discard(i)

    def _add(i):
        f = freqs[i]
        for p in _get_pairs(words[i]):
            pair_counts[p] += f
            where.setdefault(p, set()).add(i)

    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        for i in list(where.get(best, ())):   # only words containing the merged pair
            _remove(i)
            words[i] = _merge_word(words[i], best)
            _add(i)
    return merges


class Tokenizer:
    def __init__(self, pretok_style=GPT4):
        self.merges = []          # list of (str, str)
        self.vocab = {}           # token -> id
        self.id_to_token = {}     # id -> token
        self.merge_ranks = {}     # (a, b) -> rank
        self.seeded_clusters = set()  # multi-codepoint aksharas kept whole
        self.pretok_style = pretok_style  # GPT4 or WHITESPACE (see pre_tokenize)
        self._build_specials()

    def _build_specials(self):
        for t in SPECIAL_TOKENS + BYTE_TOKENS:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        self.unk_id = self.vocab["<unk>"]

    def _emit_ids(self, tok, ids):
        """Append the id(s) for one merged token, using byte fallback for anything not
        in the vocabulary so no visible character is ever dropped or turned into <unk>."""
        if tok in self.vocab:
            ids.append(self.vocab[tok])
            return
        for cp in tok:                       # merged token not in vocab: go char by char
            if cp in self.vocab:
                ids.append(self.vocab[cp])
            else:                            # unseen code point: emit its UTF-8 bytes
                for b in cp.encode("utf-8"):
                    ids.append(self.vocab[BYTE_TOKENS[b]])

    def _rebuild_index(self):
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}

    def train(self, corpus, num_merges, seed_chars=None):
        # Seed the base vocabulary with frequent aksharas (kept whole) plus every
        # component code point. Code points give a graceful fallback for rare or
        # out-of-domain clusters (see encode) instead of emitting <unk>.
        self.seeded_clusters = seeded_clusters_from(corpus, style=self.pretok_style)
        atoms = set(seed_chars or [])
        for text in corpus:
            for word in pre_tokenize(text, self.pretok_style):
                for cluster in word:
                    if len(cluster) == 1 or cluster in self.seeded_clusters:
                        atoms.add(cluster)
                    atoms.update(cluster)  # individual code points
        for c in sorted(atoms):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        self.merges = train_merges(corpus, num_merges, self.seeded_clusters, self.pretok_style)
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
        for word in pre_tokenize(text, self.pretok_style):
            atoms = word_to_atoms(word, self.seeded_clusters)
            for tok in self._apply_merges(atoms):
                self._emit_ids(tok, ids)
        return ids

    def _token_len(self, tok):
        """How many ids `tok` contributes under byte fallback (mirrors _emit_ids)."""
        if tok in self.vocab:
            return 1
        return sum(1 if cp in self.vocab else len(cp.encode("utf-8")) for cp in tok)

    def count_tokens(self, text):
        """Number of ids encode(text) would produce, computed by tokenizing each unique
        word once and weighting by frequency. Equal to len(encode(text)) but far faster
        on repetitive corpora (the hot path for fertility during optimization)."""
        word_freqs = Counter()
        for word in pre_tokenize(text, self.pretok_style):
            word_freqs[tuple(word_to_atoms(word, self.seeded_clusters))] += 1
        per_word = {a: sum(self._token_len(t) for t in self._apply_merges(list(a)))
                    for a in word_freqs}
        return sum(per_word[a] * f for a, f in word_freqs.items())

    def tokenize(self, text):
        """Like encode, but return the token strings (for display/examples)."""
        toks = []
        for word in pre_tokenize(text, self.pretok_style):
            atoms = word_to_atoms(word, self.seeded_clusters)
            for tok in self._apply_merges(atoms):
                if tok in self.vocab:
                    toks.append(tok)
                else:
                    for cp in tok:
                        if cp in self.vocab:
                            toks.append(cp)
                        else:
                            toks.extend(BYTE_TOKENS[b] for b in cp.encode("utf-8"))
        return toks

    def decode(self, ids):
        """Inverse of encode: byte-fallback tokens are buffered and UTF-8 decoded so the
        original code points are reconstructed; BOUNDARY becomes a space."""
        out, buf = [], bytearray()
        for i in ids:
            tok = self.id_to_token.get(i, "")
            b = _TOK_TO_BYTE.get(tok)
            if b is not None:
                buf.append(b)
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            out.append(tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out).replace(BOUNDARY, " ").strip()

    def to_dict(self):
        return {
            "version": 2,
            "vocab_size": len(self.vocab),
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "seeded_clusters": sorted(self.seeded_clusters),
            "pretok_style": self.pretok_style,
            "byte_fallback": True,
        }

    @classmethod
    def from_dict(cls, d):
        tok = cls(pretok_style=d.get("pretok_style", GPT4))
        tok.vocab = dict(d["vocab"])
        tok.merges = [tuple(m) for m in d["merges"]]
        tok.seeded_clusters = set(d.get("seeded_clusters", []))
        tok.unk_id = tok.vocab["<unk>"]
        tok._rebuild_index()
        return tok
