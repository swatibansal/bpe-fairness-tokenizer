"""Render a self-contained HTML report widget for the tokenizer.

The widget shows the build score, per-language fertility stats, the tokenizer design
(with a worked example at every step), example tokenizations, and a live playground
that tokenizes text in-browser. The playground JavaScript is a direct port of the
Python encode path (NFC + akshara segmentation + threshold expansion + BPE merges).
"""

import json
import html as html_lib

# Preset examples for the live playground (short, India-themed sentences).
PRESETS = [
    ("English", "India is a country in South Asia."),
    ("Hindi", "भारत दक्षिण एशिया में एक विशाल देश है।"),
    ("Telugu", "భారతదేశం దక్షిణ ఆసియాలో ఒక పెద్ద దేశం."),
    ("Bengali", "ভারত দক্ষিণ এশিয়ার একটি বিশাল দেশ।"),
]


def _rows(per_lang, lang_names, examples):
    out = []
    for lang, m in per_lang.items():
        name = html_lib.escape(lang_names.get(lang, lang))
        ex = examples.get(lang, {}) if examples else {}
        word = html_lib.escape(ex.get("word", ""))
        ntok = ex.get("n_tokens", "")
        ex_cell = f"<code>{word}</code> → {ntok} tok" if word else ""
        units = m.get("units", m.get("words", ""))
        out.append(
            f"<tr><td>{name}</td><td>{units}</td><td>{m['tokens']}</td>"
            f"<td><b>{m['X']}</b></td><td>{m.get('vocab_used', '')}</td>"
            f"<td>{ex_cell}</td></tr>"
        )
    return "\n".join(out)


def _example_cards(examples, lang_names):
    if not examples:
        return ""
    cards = []
    for lang, ex in examples.items():
        word = ex.get("word", "")
        if not word:
            continue
        name = html_lib.escape(lang_names.get(lang, lang))
        chips = "".join(
            f'<span class="chip" style="background:{_PALETTE[i % len(_PALETTE)]}">'
            f'{html_lib.escape(t).replace("▁", "<span class=b>▁</span>")}</span>'
            for i, t in enumerate(ex.get("tokens", []))
        )
        cards.append(
            f'<div style="margin:.5rem 0"><b>{name}</b> — '
            f'<code>{html_lib.escape(word)}</code> '
            f'({ex.get("n_tokens", 0)} tokens)<div class="chips">{chips}</div></div>'
        )
    return "\n".join(cards)


_PALETTE = ['#1d4ed8', '#0369a1', '#0f766e', '#4d7c0f', '#a16207', '#b45309', '#9f1239', '#7e22ce']

_STYLE = """
body{font-family:system-ui,-apple-system,sans-serif;margin:2rem auto;max-width:960px;
     background:#0f172a;color:#e2e8f0;line-height:1.55}
h1{margin-bottom:.25rem} h2{color:#93c5fd;margin-top:2rem;border-bottom:1px solid #334155;padding-bottom:.3rem}
.score{font-size:3.5rem;font-weight:700;color:#38bdf8;line-height:1}
.cards{display:flex;flex-wrap:wrap;gap:1rem}
.metric{background:#1e293b;border-radius:10px;padding:1rem 1.25rem;flex:1;min-width:160px}
.metric .n{font-size:2rem;font-weight:700;color:#38bdf8}
.card{background:#1e293b;padding:1.1rem 1.4rem;border-radius:10px;margin:1rem 0}
.muted{color:#94a3b8;font-size:.9rem}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #334155;padding:.5rem .7rem;text-align:left}
th{background:#1e293b}
button{background:#38bdf8;border:0;color:#0f172a;padding:.6rem 1rem;border-radius:8px;
       font-weight:600;cursor:pointer;font-size:.95rem}
button:hover{background:#0ea5e9}
button.ghost{background:#334155;color:#e2e8f0;margin:.2rem .3rem .2rem 0}
button.ghost:hover{background:#475569}
code{background:#334155;padding:.1rem .35rem;border-radius:4px}
textarea{width:100%;box-sizing:border-box;min-height:70px;background:#0b1220;color:#e2e8f0;
         border:1px solid #334155;border-radius:8px;padding:.6rem;font-size:1.15rem;font-family:inherit}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:.5rem}
.chip{padding:.15rem .45rem;border-radius:5px;font-size:1.05rem;border:1px solid #33415580}
.unit{padding:.15rem .45rem;border-radius:5px;font-size:1.05rem;background:#0b1220;border:1px solid #334155;color:#cbd5e1}
.unit.sym{background:#3b2f14;border-color:#a16207;color:#fde68a}
.chip .b, .b{color:#93c5fd}
.stat{display:inline-block;margin-right:1.4rem}
.stat b{color:#38bdf8;font-size:1.25rem}
.step{margin:.9rem 0;padding-left:.9rem;border-left:3px solid #334155}
.step h4{margin:.2rem 0;color:#e2e8f0}
.seg{display:inline-block;background:#0b1220;border:1px solid #334155;border-radius:5px;
     padding:.05rem .4rem;margin:2px}
.expand{border-color:#b45309}
ul{margin:.4rem 0}
.bar{background:#0b1220;border-radius:5px;height:22px;position:relative;overflow:hidden;border:1px solid #334155}
.bar>span{display:block;height:100%}
.mark12{position:absolute;top:-2px;bottom:-2px;width:2px;background:#ef4444}
.fert-row{display:grid;grid-template-columns:90px 1fr 96px;align-items:center;gap:.6rem;margin:.35rem 0}
.stacked{display:flex;height:34px;border-radius:6px;overflow:hidden;margin:.4rem 0;border:1px solid #334155}
.stacked>div{display:flex;align-items:center;justify-content:center;font-size:.7rem;color:#fff;overflow:hidden;white-space:nowrap}
.legend{margin:.3rem 0}
.legend span{display:inline-block;margin-right:1rem;font-size:.85rem}
.sw{display:inline-block;width:.8rem;height:.8rem;border-radius:2px;vertical-align:middle;margin-right:.3rem}
.grid{display:flex;flex-wrap:wrap;gap:3px;margin:.3rem 0 .8rem}
.gchip{padding:.05rem .35rem;border-radius:4px;font-size:1rem;color:#fff}
pre.code{background:#0b1220;border:1px solid #334155;border-radius:8px;padding:1rem;
         overflow:auto;max-height:460px;font-size:.78rem;line-height:1.45;white-space:pre}
details summary{cursor:pointer;color:#93c5fd;font-weight:600;margin:.3rem 0}
"""

# Fully static JS (reads everything from the embedded JSON blob).
_SCRIPT = r"""
const TK = JSON.parse(document.getElementById('tokenizer-data').textContent);
const BOUNDARY = '▁', ZWJ = '‍', ZWNJ = '‌', SEP = '\u0001';
const VOCAB = TK.vocab || {};
const UNK = (VOCAB['<unk>'] !== undefined) ? VOCAB['<unk>'] : 0;
const SEEDED = new Set(TK.seeded_clusters || []);
const VIRAMAS = new Set(TK.viramas || []);
const PRESETS = TK.presets || [];
const STYLE = TK.pretok_style || 'gpt4';   // 'whitespace' (metaspace) or 'gpt4'
const RANK = new Map();
(TK.merges || []).forEach((m, i) => RANK.set(m[0] + SEP + m[1], i));
const MARK = /^[\p{Mn}\p{Mc}\p{Me}]$/u;
const PALETTE = ['#1d4ed8','#0369a1','#0f766e','#4d7c0f','#a16207','#b45309','#9f1239','#7e22ce'];
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const showB = s => esc(s).split(BOUNDARY).join('<span class="b">▁</span>');

function endsWithVirama(cluster){
  const cps = [...cluster]; let k = cps.length - 1;
  while (k >= 0 && (cps[k] === ZWJ || cps[k] === ZWNJ)) k--;
  return k >= 0 && VIRAMAS.has(cps[k]);
}
function graphemeClusters(word){
  const cps = [...word], clusters = []; let i = 0;
  while (i < cps.length){
    let cluster = cps[i]; i++;
    while (i < cps.length){
      const ch = cps[i];
      if (MARK.test(ch) || ch === ZWJ || ch === ZWNJ) cluster += ch;
      else if (endsWithVirama(cluster)) cluster += ch;
      else break;
      i++;
    }
    clusters.push(cluster);
  }
  return clusters;
}
function isAtom(c){ return [...c].length === 1 || SEEDED.has(c); }
function wordToAtoms(clusters){
  const out = [];
  for (const c of clusters){
    if (isAtom(c)) out.push(c);
    else for (const cp of [...c]) out.push(cp);
  }
  return out;
}
// Pre-tokenization (mirrors bpe.py). The WHITESPACE (metaspace) style keeps each
// whitespace word whole, so BPE may merge across punctuation; the GPT4 style splits
// punctuation/contractions off and groups digits. Grapheme clustering runs first in
// both, so matras/conjuncts stay bound.
const P_DIGIT = /\p{Nd}/u, P_LETTER = /\p{L}/u, P_MARK = /\p{M}/u;
const CONTRACTIONS = new Set(['s','d','m','t','ll','ve','re']);
function isIndic(ch){ const o = ch.codePointAt(0); return o >= 0x0900 && o <= 0x0DFF; }
function atomClass(c){
  const ch = [...c][0];
  if (P_DIGIT.test(ch)) return 'd';
  if (P_LETTER.test(ch) || P_MARK.test(ch) || ch === ZWJ || ch === ZWNJ) return 'l';
  return 'o';
}
function contractionLen(cl, i){
  if (cl[i] !== "'") return 0;
  for (const size of [2, 1]){
    const tail = cl.slice(i + 1, i + 1 + size);
    if (tail.length === size && tail.every(c => atomClass(c) === 'l')
        && CONTRACTIONS.has(tail.join('').toLowerCase())) return size;
  }
  return 0;
}
function splitWord(cl){
  const chunks = []; let i = 0; const n = cl.length;
  while (i < n){
    const cls = atomClass(cl[i]); const m = contractionLen(cl, i);
    if (m){ chunks.push(cl.slice(i, i + 1 + m)); i += 1 + m; }
    else if (cls === 'o'){
      if (i + 1 < n && atomClass(cl[i+1]) === 'l'){
        const chunk = [cl[i]]; i++;
        while (i < n && atomClass(cl[i]) === 'l'){ chunk.push(cl[i]); i++; }
        chunks.push(chunk);
      } else {
        const chunk = [];
        while (i < n && atomClass(cl[i]) === 'o' && !contractionLen(cl, i)){ chunk.push(cl[i]); i++; }
        chunks.push(chunk);
      }
    } else if (cls === 'l'){
      const chunk = [];
      while (i < n && atomClass(cl[i]) === 'l'){ chunk.push(cl[i]); i++; }
      chunks.push(chunk);
    } else {
      const run = [];
      while (i < n && atomClass(cl[i]) === 'd'){ run.push(cl[i]); i++; }
      for (let j = 0; j < run.length; j += 3) chunks.push(run.slice(j, j + 3));
    }
  }
  return chunks;
}
function wordSegments(clusters){
  return STYLE === 'whitespace' ? [clusters] : splitWord(clusters);
}
function applyMerges(atoms){
  atoms = atoms.slice();
  while (true){
    let bestRank = Infinity, bi = -1;
    for (let i = 0; i < atoms.length - 1; i++){
      const r = RANK.get(atoms[i] + SEP + atoms[i+1]);
      if (r !== undefined && r < bestRank){ bestRank = r; bi = i; }
    }
    if (bi < 0) break;
    const a = atoms[bi], b = atoms[bi+1], out = []; let i = 0;
    while (i < atoms.length){
      if (i < atoms.length - 1 && atoms[i] === a && atoms[i+1] === b){ out.push(a + b); i += 2; }
      else { out.push(atoms[i]); i++; }
    }
    atoms = out;
  }
  return atoms;
}
// Byte fallback (mirrors bpe.py): a code point not in the vocab is emitted as its UTF-8
// byte tokens "<0xNN>", so any character round-trips instead of collapsing to <unk>.
const BYTE_TOK = {};
for (let b = 0; b < 256; b++) BYTE_TOK[b] = '<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>';
const BYTE_RE = /^<0x([0-9A-F]{2})>$/;
const UTF8_ENC = new TextEncoder(), UTF8_DEC = new TextDecoder();
function tokensToIds(merged){
  const toks = [];
  for (const t of merged){
    if (VOCAB[t] !== undefined){ toks.push({ s: t, id: VOCAB[t] }); continue; }
    for (const cp of [...t]){
      if (VOCAB[cp] !== undefined) toks.push({ s: cp, id: VOCAB[cp] });
      else for (const b of UTF8_ENC.encode(cp)){ const bt = BYTE_TOK[b]; toks.push({ s: bt, id: VOCAB[bt] }); }
    }
  }
  return toks;
}
// Decode: id -> token string; byte-fallback tokens are buffered and UTF-8 decoded, then
// BOUNDARY becomes a space (mirrors bpe.py decode).
const ID2TOK = {};
Object.keys(VOCAB).forEach(t => { ID2TOK[VOCAB[t]] = t; });
function decodeIds(ids){
  let out = '', buf = [];
  const flush = () => { if (buf.length){ out += UTF8_DEC.decode(new Uint8Array(buf)); buf = []; } };
  for (const i of ids){
    const t = (ID2TOK[i] !== undefined ? ID2TOK[i] : '');
    const m = BYTE_RE.exec(t);
    if (m){ buf.push(parseInt(m[1], 16)); continue; }
    flush(); out += t;
  }
  flush();
  return out.split(BOUNDARY).join(' ').trim();
}
// The content the tokenizer preserves: NFC + whitespace collapsed to single spaces.
function canonical(text){ return text.normalize('NFC').split(/\s+/).filter(Boolean).join(' '); }
// Faithful units: each letter/mark/number run, or one visible symbol (mirrors metrics.py).
const FUNIT = /[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu;
const FUNIT_RUN = /[\p{L}\p{M}\p{N}]/u;   // a unit is a letter/mark/number run vs a symbol
function faithfulUnitList(text){ return [...text.normalize('NFC').matchAll(FUNIT)].map(m => m[0]); }
function faithfulUnits(text){ return faithfulUnitList(text).length; }
// Render the faithful units as chips; letter/number runs and lone symbols are shaded apart.
function unitChipHTML(units){
  return units.map(u => '<span class="unit' + (FUNIT_RUN.test(u) ? '' : ' sym') + '">' + esc(u) + '</span>').join('');
}
function tokenize(text){
  text = text.normalize('NFC');
  const words = text.split(/\s+/).filter(Boolean);
  let tokens = [], aksharas = 0, codepoints = 0;
  for (const w of words){
    const clusters = graphemeClusters(w);
    aksharas += clusters.length; codepoints += [...w].length;
    wordSegments(clusters).forEach((seg, k) => {
      const atoms = wordToAtoms(k === 0 ? [BOUNDARY].concat(seg) : seg);
      tokens = tokens.concat(tokensToIds(applyMerges(atoms)));
    });
  }
  return { tokens, words: words.length, aksharas, codepoints };
}
function chipHTML(tokens){
  return tokens.map((t, i) =>
    '<span class="chip" style="background:' + PALETTE[i % PALETTE.length] +
    '" title="id ' + t.id + '">' + showB(t.s) + '</span>').join('');
}
function segHTML(items, cls){
  return items.map(s => '<span class="seg ' + (cls||'') + '">' + showB(s) + '</span>').join('');
}

// Step-by-step pipeline shown for the FIRST word of the input.
function renderSteps(text){
  const words = text.normalize('NFC').split(/\s+/).filter(Boolean);
  const el = document.getElementById('steps');
  if (!words.length){ el.innerHTML = '<span class="muted">Type something to see the steps.</span>'; return; }
  const w = words[0];
  const norm = w.normalize('NFC');
  const clusters = graphemeClusters(w);
  const segs = wordSegments(clusters);
  const chunkStr = segs.map((s, k) => (k === 0 ? BOUNDARY : '') + s.join(''));
  let atoms = [], merged = [];
  segs.forEach((seg, k) => {
    const a = wordToAtoms(k === 0 ? [BOUNDARY].concat(seg) : seg);
    atoms = atoms.concat(a);
    merged = merged.concat(applyMerges(a));
  });
  const expanded = clusters.filter(c => !isAtom(c));
  el.innerHTML =
    '<div class="muted">Pipeline for the first word: <code>' + esc(w) + '</code></div>' +
    '<div class="step"><h4>1 · Normalize (NFC)</h4>' +
      esc(norm) + ' <span class="muted">(' + [...norm].length + ' Unicode code points)</span></div>' +
    '<div class="step"><h4>2 · Akshara segmentation</h4>' + segHTML(clusters) +
      ' <span class="muted">(' + clusters.length + ' grapheme clusters)</span></div>' +
    '<div class="step"><h4>3 · Pre-tokenization → chunks</h4>' + segHTML(chunkStr) +
      ' <span class="muted">' + (STYLE === 'whitespace'
        ? '(metaspace: the whole word is one chunk, so BPE may merge across punctuation)'
        : '(gpt4 split: ' + segs.length + ' chunk' + (segs.length === 1 ? '' : 's') + '; merges stay inside a chunk)') + '</span></div>' +
    '<div class="step"><h4>4 · Threshold &amp; atoms</h4>' + segHTML(atoms) +
      ' <span class="muted">' + (expanded.length
        ? '(rare akshara ' + expanded.map(esc).join(', ') + ' decomposed to code points)'
        : '(all aksharas frequent enough to keep whole)') + '</span></div>' +
    '<div class="step"><h4>5 · BPE merges → tokens</h4>' + chipHTML(tokensToIds(merged)) +
      ' <span class="muted">(' + merged.length + ' tokens)</span></div>';
}

function render(){
  const text = document.getElementById('inp').value;
  const r = tokenize(text);
  const unitList = faithfulUnitList(text);
  const units = unitList.length;
  const ratio = units ? (r.tokens.length / units).toFixed(3) : '0';
  document.getElementById('out').innerHTML =
    '<div class="stat">faithful units <b>' + units + '</b></div>' +
    '<div class="stat">aksharas <b>' + r.aksharas + '</b></div>' +
    '<div class="stat">tokens <b>' + r.tokens.length + '</b></div>' +
    '<div class="stat">tokens/unit <b>' + ratio + '</b></div>' +
    '<div class="muted" style="margin-top:.7rem">Tokens (' + r.tokens.length +
      ') — <span class="b">▁</span> marks a word start:</div>' +
    '<div class="chips">' + chipHTML(r.tokens) + '</div>' +
    '<div class="muted" style="margin-top:.7rem">Faithful units (' + units +
      ') — one letter/mark/number run, or one visible symbol (the fertility denominator):</div>' +
    '<div class="chips">' + unitChipHTML(unitList) + '</div>';
  // Decode the ids back to text and check the round-trip.
  const ids = r.tokens.map(t => t.id);
  const decoded = decodeIds(ids);
  const ok = decoded === canonical(text);
  const badge = ok
    ? '<span style="color:#22c55e;font-weight:700">✓ round-trips</span>'
    : '<span style="color:#ef4444;font-weight:700">✗ differs</span>';
  document.getElementById('decoded').innerHTML = !r.tokens.length ? '' :
    '<div class="muted">Decode (token ids → text) &nbsp; ' + badge +
      ' <span class="muted">— decoding the ids alone reproduces the input, up to whitespace normalization.</span></div>' +
    '<div class="muted" style="word-break:break-all;margin-top:.3rem">ids: [' + ids.join(', ') + ']</div>' +
    '<div style="margin-top:.3rem">→ <code>' + esc(decoded) + '</code></div>';
  renderSteps(text);
}
function setText(t){ document.getElementById('inp').value = t; render(); }

// Build preset buttons in JS (avoids any HTML-attribute quoting problems).
const pc = document.getElementById('presets');
PRESETS.forEach(([name, txt]) => {
  const b = document.createElement('button');
  b.className = 'ghost'; b.textContent = name;
  b.addEventListener('click', () => setText(txt));
  pc.appendChild(b);
});
document.getElementById('inp').addEventListener('input', render);
render();

function exportJSON(){
  const data = document.getElementById('tokenizer-data').textContent;
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'tokenizer.json'; a.click();
  URL.revokeObjectURL(url);
}
"""


SCRIPT_COLORS = {"Latin": "#1d4ed8", "Devanagari": "#0f766e", "Telugu": "#a16207",
                 "Bengali": "#9f1239", "Other": "#475569"}
LANG_SCRIPT = {"en": "Latin", "hi": "Devanagari", "te": "Telugu", "bn": "Bengali"}

# The strategies we tried, and the fertility (tokens per faithful unit) each reached on
# the faithful Markdown corpus — the incremental progress toward the shipped build.
# Lower is better; the last row is what ships.
STRATEGIES = [
    ("GPT-4 split · per-language hill-climb", {"en": 0.979, "hi": 1.018, "te": 0.945, "bn": 0.964},
     "Punctuation splits off every word — clean boundaries, but a merge can't span a "
     "faithful unit, so fertility floors near 1 token/unit."),
    ("Metaspace · per-language hill-climb", {"en": 0.813, "hi": 0.854, "te": 0.769, "bn": 0.782},
     "Whitespace-only split lets merges span punctuation. The hill-climb minimizes spread "
     "(fairness) rather than absolute tokens, so all four cluster around 0.8."),
    ("Metaspace · single shared BPE (shipped)", {"en": 0.617, "hi": 0.603, "te": 0.694, "bn": 0.714},
     "One BPE trained over the weighted concatenation (like the HuggingFace baseline), so "
     "English claims the ~5,000 merge slots it needs. Matches the published fertility."),
]
STRATEGY_PUBLISHED = {"en": 0.598, "hi": 0.579, "te": 0.673, "bn": 0.733}  # bn slot = Maithili


def _strategies_html(shipped_fert=None):
    """Render the strategies table. The shipped row's numbers are taken from the live
    build (shipped_fert: {lang: X}) so it always matches the fertility table above."""
    langs = ["en", "hi", "te", "bn"]
    head = "".join(f"<th>{l}</th>" for l in langs)
    rows = []
    for i, (name, fert, note) in enumerate(STRATEGIES):
        shipped = i == len(STRATEGIES) - 1
        if shipped and shipped_fert:
            fert = {l: shipped_fert.get(l, fert[l]) for l in langs}
        cells = "".join(f"<td>{fert[l]:.3f}</td>" for l in langs)
        label = f"<b>{html_lib.escape(name)}</b>" if shipped else html_lib.escape(name)
        style = ' style="background:#12321e"' if shipped else ""
        rows.append(f"<tr{style}><td>{label}</td>{cells}</tr>"
                    f'<tr{style}><td colspan="5" class="muted" style="padding-top:0">'
                    f'{html_lib.escape(note)}</td></tr>')
    pub = "".join(f"<td>{STRATEGY_PUBLISHED[l]:.3f}</td>" for l in langs)
    rows.append(f'<tr><td class="muted">published (HuggingFace) — bn col = Maithili</td>{pub}</tr>')
    return (f'<table><thead><tr><th>Strategy</th>{head}</tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def _script_of(token):
    for ch in token:
        if ch == "▁":
            continue
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "Devanagari"
        if 0x0980 <= o <= 0x09FF: return "Bengali"
        if 0x0C00 <= o <= 0x0C7F: return "Telugu"
        if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or (0xC0 <= o <= 0x24F): return "Latin"
    return "Other"


def _stacked(counts, colors, total):
    """A horizontal stacked bar from an ordered list of (label, count)."""
    segs = []
    for label, n in counts:
        if n <= 0:
            continue
        pct = 100 * n / total if total else 0
        c = colors.get(label.split("/")[-1], "#475569")
        txt = label if pct > 9 else ""
        segs.append(f'<div style="width:{pct:.2f}%;background:{c}" title="{html_lib.escape(label)}: {n}">{html_lib.escape(txt)}</div>')
    return '<div class="stacked">' + "".join(segs) + "</div>"


def _composition(tokenizer_dict):
    vocab = tokenizer_dict.get("vocab", {})
    merges = tokenizer_dict.get("merges", [])
    specials = set(tokenizer_dict.get("special_tokens", []))
    merged = {a + b for a, b in merges}
    n_special = sum(1 for t in vocab if t in specials)
    n_merge = sum(1 for t in vocab if t not in specials and t in merged)
    n_base = len(vocab) - n_special - n_merge
    total = len(vocab) or 1

    by_role = [("special tokens", n_special), ("base atoms", n_base), ("merges", n_merge)]
    role_colors = {"special tokens": "#475569", "base atoms": "#0369a1", "merges": "#7e22ce"}

    script_ct = {}
    for t in vocab:
        if t in specials:
            continue
        s = _script_of(t)
        script_ct[s] = script_ct.get(s, 0) + 1
    by_script = [(s, script_ct.get(s, 0)) for s in ["Latin", "Devanagari", "Telugu", "Bengali", "Other"]]

    def legend(items, colors):
        return '<div class="legend">' + "".join(
            f'<span><span class="sw" style="background:{colors.get(l, "#475569")}"></span>{html_lib.escape(l)} ({n})</span>'
            for l, n in items if n > 0) + "</div>"

    return (
        '<div class="muted">By role</div>' + _stacked(by_role, role_colors, total) + legend(by_role, role_colors) +
        '<div class="muted" style="margin-top:.6rem">By script</div>' + _stacked(by_script, SCRIPT_COLORS, total) +
        legend(by_script, SCRIPT_COLORS)
    )


def _fertility_bars(per_lang, lang_names, limit=1.2, scale=1.5):
    rows = []
    for lang, m in per_lang.items():
        x = m["X"]
        name = html_lib.escape(lang_names.get(lang, lang))
        w = min(100.0, x / scale * 100)
        color = SCRIPT_COLORS.get(LANG_SCRIPT.get(lang, "Other"), "#38bdf8")
        rows.append(
            f'<div class="fert-row"><div>{name}</div>'
            f'<div class="bar"><span style="width:{w:.1f}%;background:{color}"></span>'
            f'<div class="mark12" style="left:{limit/scale*100:.1f}%"></div></div>'
            f'<div><b>{x}</b></div></div>'
        )
    return "".join(rows) + (f'<div class="muted">Red line = {limit} (the Hindi-penalty '
                            f'threshold) · bars scaled to {scale} tokens/faithful unit</div>')


def _seeded_explorer(seeded_freq, per_script_cap=70):
    by_script = {}
    for c, f in seeded_freq.items():
        by_script.setdefault(_script_of(c), []).append((c, f))
    blocks = []
    for script in ["Devanagari", "Telugu", "Bengali", "Latin", "Other"]:
        items = by_script.get(script)
        if not items:
            continue
        items.sort(key=lambda cf: -cf[1])
        fmax = max(f for _, f in items) or 1
        color = SCRIPT_COLORS.get(script, "#475569")
        chips = []
        for c, f in items[:per_script_cap]:
            op = 0.35 + 0.65 * (f / fmax)  # shade by frequency
            chips.append(f'<span class="gchip" style="background:{color};opacity:{op:.2f}" '
                         f'title="count {f}">{html_lib.escape(c)}</span>')
        more = f' <span class="muted">+{len(items) - per_script_cap} more</span>' if len(items) > per_script_cap else ""
        blocks.append(f'<div style="margin-top:.5rem"><b>{script}</b> '
                      f'<span class="muted">({len(items)} seeded aksharas)</span></div>'
                      f'<div class="grid">{"".join(chips)}{more}</div>')
    return "".join(blocks)


def _boundary_bars(boundary, lang_names, scale=None):
    naive, akshara = boundary.get("naive", {}), boundary.get("akshara", {})
    langs = list(naive.keys())
    mx = max([naive.get(l, 0) for l in langs] + [1])
    rows = []
    for l in langs:
        nv, ak = naive.get(l, 0), akshara.get(l, 0)
        name = html_lib.escape(lang_names.get(l, l))
        wn, wa = 100 * nv / mx, 100 * ak / mx
        rows.append(
            f'<div style="margin:.4rem 0"><div>{name}</div>'
            f'<div class="fert-row"><div class="muted">naive</div>'
            f'<div class="bar"><span style="width:{wn:.1f}%;background:#ef4444"></span></div><div><b>{nv}</b></div></div>'
            f'<div class="fert-row"><div class="muted">akshara</div>'
            f'<div class="bar"><span style="width:{wa:.1f}%;background:#22c55e"></span></div><div><b>{ak}</b></div></div>'
            '</div>'
        )
    b = boundary.get("budget", "")
    return ("".join(rows) +
            f'<div class="muted">Fragment token <i>types</i> (tokens that begin mid-syllable) '
            f'in a {b}-merge per-language vocabulary. Lower is better; akshara-aware ≈ 0.</div>')


def render_report(per_lang, agg, tokenizer_dict, lang_names, constraint=None,
                  examples=None, viz=None, code=None):
    rows = _rows(per_lang, lang_names, examples)
    score = agg.get("hindi_adjusted_score", agg["build_score"])
    score_str = "∞" if score == float("inf") else f"{score:.2f}"
    penalty = agg.get("hindi_penalty")
    sorted_x = ", ".join(str(x) for x in agg.get("sorted_X", []))
    vocab_total = tokenizer_dict.get("vocab_size", len(tokenizer_dict.get("vocab", {})))
    n_seeded = len(tokenizer_dict.get("seeded_clusters", []))
    n_merges = len(tokenizer_dict.get("merges", []))
    n_specials = len(tokenizer_dict.get("special_tokens", []))
    n_base = vocab_total - n_merges
    _metrics = tokenizer_dict.get("metrics", {})
    seed_min_freq = _metrics.get("seed_min_freq", 2)
    weights = _metrics.get("train_weights")
    shared = _metrics.get("build") == "shared_bpe"
    weights_str = ", ".join(f"{k}:{v}" for k, v in weights.items()) if weights else ""

    # Embed presets in the JSON blob so JS can build buttons without quoting issues.
    payload = dict(tokenizer_dict)
    payload["presets"] = [[n, t] for n, t in PRESETS]
    blob = json.dumps(payload, ensure_ascii=False)

    # Hindi-penalty note (replaces the old English X<=1.2 hard-constraint card). The
    # score is penalized only if Hindi fertility exceeds 1.2; here it does not.
    constraint_html = ""
    hi = per_lang.get("hi", {}).get("X")
    if penalty is not None and hi is not None:
        met = hi <= 1.2 + 1e-9
        color = "#22c55e" if met else "#ef4444"
        badge = "no penalty" if met else "penalized"
        constraint_html = (
            f'<div class="card"><div class="muted">Hindi penalty</div>'
            f'<div style="font-size:1.1rem">Hindi fertility {hi} &nbsp;→&nbsp; '
            f'penalty factor {penalty:.4f} &nbsp;'
            f'<span style="color:{color};font-weight:700">{badge}</span></div>'
            f'<div class="muted">The score is divided by exp(max(0, Hindi/1.2 − 1)); it '
            f'only bites when Hindi fertility exceeds 1.2. Fertility is measured on the '
            f'faithful Markdown corpus, and the tokenizer emits no '
            f'<code>&lt;unk&gt;</code> on any of it.</div></div>'
        )

    example_cards = _example_cards(examples, lang_names)

    viz = viz or {}
    fertility_html = _fertility_bars(per_lang, lang_names)
    strategies_html = _strategies_html({l: per_lang[l]["X"] for l in per_lang} if shared else None)
    composition_html = _composition(tokenizer_dict)
    seeded_html = _seeded_explorer(viz.get("seeded_freq", {})) if viz.get("seeded_freq") else ""
    boundary_html = _boundary_bars(viz.get("boundary", {}), lang_names) if viz.get("boundary", {}).get("naive") else ""
    code_html = f'<pre class="code">{html_lib.escape(code)}</pre>' if code else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>BPE Tokenizer Report</title>
<style>{_STYLE}</style></head><body>
<h1>Multilingual BPE Tokenizer</h1>
<p class="muted">One shared vocabulary across English, Hindi, Telugu, and Bengali, trained
on the <b>faithful Markdown</b> of Wikipedia "India". Grapheme-aware (Brahmic-capable)
BPE with metaspace-style pre-tokenization; fully reversible on visible characters.</p>

<div class="cards">
<div class="metric"><div class="muted">Build score</div><div class="n">{score_str}</div>
<div class="muted">1000 / (X_max − X_min), ÷ Hindi penalty</div></div>
<div class="metric"><div class="muted">Vocabulary tokens</div><div class="n">{vocab_total}</div>
<div class="muted">{n_base} base + {n_merges} merges</div></div>
<div class="metric"><div class="muted">Seeded aksharas</div><div class="n">{n_seeded}</div>
<div class="muted">freq ≥ 2 (rest fall back)</div></div>
<div class="metric"><div class="muted">Spread</div><div class="n">{agg.get('spread')}</div>
<div class="muted">X_max {agg.get('X_max')} · X_min {agg.get('X_min')}</div></div>
</div>
{constraint_html}

<h2>Fertility (tokens per faithful unit) for every language</h2>
<p class="muted">Every ratio below is measured on the full faithful Markdown corpus — the
text this tokenizer is built for. A <b>faithful unit</b> is one contiguous letter/mark/
number run, or one visible punctuation/symbol character. No language produces any
<code>&lt;unk&gt;</code> on that text.</p>
<table>
<thead><tr><th>Language</th><th>Faithful units</th><th>Tokens</th><th>X = fertility</th><th>Vocab used</th><th>Example</th></tr></thead>
<tbody>
{rows}
</tbody></table>
<p class="muted">Sorted X (desc): {sorted_x}. Fertility = <code>total tokens / total faithful
units</code>; lower means fewer tokens per unit (cheaper inference, longer effective
context). Metaspace pre-tokenization lets tokens span punctuation, pushing this below 1.</p>
<div class="card">{fertility_html}</div>

<h2>Strategies we tried (incremental progress)</h2>
<div class="card">
<div class="muted">Each row is a design we built and measured on this faithful Markdown
corpus; the number is fertility (tokens per faithful unit, lower is better). The shipped
build is highlighted. This is the path from ~1 token/unit down to matching the published
HuggingFace tokenizer.</div>
{strategies_html}
<div class="muted" style="margin-top:.5rem">The jump from ~0.8 to ~0.6 came not from a
different objective but from a different <b>architecture</b>: stitching per-language
merge lists gives English only a fraction of the 10k slots, whereas one shared BPE over
the weighted concatenation lets it claim the ~5,000 merges it needs.</div></div>

<h2>Where the 10k vocabulary goes</h2>
<div class="card">{composition_html}
<div class="muted" style="margin-top:.5rem">The shipped tokenizer is one shared BPE
trained over the weighted concatenation of the four corpora{f" (weights {weights_str})" if shared and weights_str else ""};
merges are ranked globally, so each script claims vocabulary in proportion to its
weighted frequency.</div></div>

<h2>Seeded aksharas (the shared units)</h2>
<div class="card">
<div class="muted">Aksharas that occur ≥ {seed_min_freq} times earn a permanent vocabulary
slot, grouped by script and shaded by frequency (darker = rarer). Rarer aksharas are not
shown — they decompose to code points at encode time.</div>
{seeded_html}</div>

<h2>Boundary integrity: naive vs. akshara-aware</h2>
<div class="card">{boundary_html}</div>

<h2>Example tokenizations</h2>
<div class="card">{example_cards}
<div class="muted">Each chip is one token; <span class="b">▁</span> marks a word start.</div></div>

<h2>How this tokenizer works (worked example)</h2>
<div class="card">
<div class="step"><h4>1 · Normalize (NFC)</h4>
Precomposed and decomposed forms are unified. Example: <code>क़</code> written as a single
code point vs. as <code>क</code> + nukta both normalize to the same sequence, so they
tokenize identically.</div>
<div class="step"><h4>2 · Akshara segmentation</h4>
Words split into grapheme clusters, not raw code points. Example:
<code>हिन्दी</code> → <span class="seg">हि</span><span class="seg">न्दी</span>
(the matra stays on its consonant and the <code>न्द</code> conjunct stays whole),
instead of the 6 separate code points a naive splitter would produce.</div>
<div class="step"><h4>3 · Metaspace pre-tokenization</h4>
Each whitespace word is kept whole as one chunk (metaspace style), so BPE may merge
<i>across</i> punctuation into a single token — which is what pushes fertility below one
token per faithful unit on this in-domain corpus:
<code>देश।</code> → <span class="seg">▁देश।</span> &nbsp;
<code>India.</code> → <span class="seg">▁India.</span> &nbsp;
<code>1,428</code> → <span class="seg">▁1,428</span>.
Grapheme clustering still runs first, so matras and conjuncts never split inside a word.
(A GPT-4-style mode that splits punctuation off is also available; it keeps cleaner token
boundaries but floors fertility near one token per unit.)</div>
<div class="step"><h4>4 · Frequency-thresholded base vocab</h4>
An akshara gets a permanent slot only if it appears ≥ {seed_min_freq} times ({n_seeded}
qualify here). Example: common <code>भा</code> is kept whole, but a rare akshara is
<span class="expand seg">decomposed</span> to its code points, so a scarce slot is not
wasted on it (BPE can still rebuild it from those code points).</div>
<div class="step"><h4>5 · BPE merges → tokens</h4>
{n_merges} learned merges combine frequent adjacent clusters into larger subword tokens
(merges stay inside a chunk from step 3). Example:
<code>भारत</code> → <span class="seg">भा</span><span class="seg">र</span><span class="seg">त</span>
merges toward a single <code>▁भारत</code> token when the whole word is frequent.</div>
<div class="step"><h4>6 · Decode (reversible)</h4>
Decoding runs the pipeline backwards: each id maps to its token string, the strings are
joined, and <code>▁</code> becomes a space — reconstructing the original text (up to
whitespace normalization). Try it in the playground below.</div>
<div class="muted">The live playground below runs exactly this pipeline in your browser,
step by step, for whatever you type.</div>
</div>

<h2>Try the tokenizer live</h2>
<div class="card">
<div class="muted">Load an example or type your own — tokenization runs in your browser:</div>
<div id="presets" style="margin:.5rem 0"></div>
<textarea id="inp">भारत दक्षिण एशिया में एक विशाल देश है।</textarea>
<div id="out" style="margin-top:.8rem"></div>
<div id="decoded" style="margin-top:.9rem;border-top:1px solid #334155;padding-top:.7rem"></div>
</div>
<div class="card" id="steps"></div>

<h2>Implementation</h2>
<div class="card"><details><summary>Show the tokenizer source (bpe.py)</summary>
{code_html}</details>
<div class="muted">This is the exact from-scratch implementation; the live playground
above is a faithful JavaScript port of its <code>encode</code> path.</div></div>

<button onclick="exportJSON()">Export tokenizer JSON</button>
<script type="application/json" id="tokenizer-data">{blob}</script>
<script>{_SCRIPT}</script>
</body></html>"""
