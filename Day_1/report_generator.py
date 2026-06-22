def generate_report(raw_text, cleaned_text, tokens_no_stopwords,
                    stemmed, lemmatized, tokenization_results,
                    sample_sentence, pos_results, bow_results,
                    bow_sentences, real_stem_lemma_examples):

    # ── STEMMING vs LEMMATIZATION TABLE ───────────────────────────────
    stem_lemma_table = "\n".join([
        f"| {o:20} | {s:20} | {l:20} |"
        for o, s, l in zip(tokens_no_stopwords[:20], stemmed[:20], lemmatized[:20])
    ])

    # ── REAL DIFFERENCES TABLE ─────────────────────────────────────────
    if real_stem_lemma_examples:
        diff_table = "\n".join([
            f"| {o:20} | {s:20} | {l:20} |"
            for o, s, l in real_stem_lemma_examples[:15]
        ])
    else:
        diff_table = "| no differences found | - | - |"

    # ── TOKENIZATION SECTION ───────────────────────────────────────────
    tokenization_text = ""
    for method, tokens in tokenization_results.items():
        preview = tokens[:20]
        tokenization_text += f"**{method.upper()}** ({len(tokens)} tokens total):\n"
        tokenization_text += f"`{preview}`\n\n"

    token_count_table = "\n".join([
        f"| {method.upper():20} | {len(tokens):12} |"
        for method, tokens in tokenization_results.items()
    ])

    # ── POS TABLE ─────────────────────────────────────────────────────
    pos_table = "\n".join([
        f"| {token:20} | {pos:10} | {desc:25} |"
        for token, pos, desc in pos_results[:30]
    ])

    # ── BOW SECTION ───────────────────────────────────────────────────
    bow_text = ""
    for i, (sentence, bow) in enumerate(zip(bow_sentences, bow_results)):
        top_words = dict(sorted(bow.items(), key=lambda x: x[1], reverse=True)[:8])
        preview = sentence[:80] + "..." if len(sentence) > 80 else sentence
        bow_text += f"**Sentence {i+1}:** `{preview}`\n"
        bow_text += f"Top 8 words: `{top_words}`\n\n"

    # ── BUILD REPORT ───────────────────────────────────────────────────
    report = f"""# NLP Pipeline Report — Day 1
### Classic NLP Pipeline & Tokenization

---

## Pipeline Order Usedraw text

↓

regex cleaning

↓
stopword removal

↓
POS tagging (spaCy)       ← moved before lemmatization

↓
stemming (PorterStemmer)

lemmatization (WordNetLemmatizer + spaCy POS tags)

↓
tokenization comparison (char / word / BPE / WordPiece)

↓
Bag of Words


**Why this order matters:**  
POS tagging was moved before lemmatization because WordNetLemmatizer needs
to know whether a word is a noun, verb, adjective, or adverb to return the
correct base form. Without POS, it defaults to noun for every word which
gives wrong results for verbs and adjectives.

---

## 1. Text Cleaning

**Before (first 300 chars):**
{raw_text[:300]}

**After (first 300 chars):**
{cleaned_text[:300]}

| Regex Step | Pattern | Removes |
|------------|---------|---------|
| 1 | `\\[\\s*\\d+\\s*\\]` | Wikipedia citations like `[1]` |
| 2 | `\\(.*?\\)` | Stage directions like `(in Japanese)` |
| 3 | `•` | Bullet symbols |
| 4 | `\\d+` | All numbers |
| 5 | `[^a-zA-Z\\s]` | Special chars, accents, punctuation |
| 6 | `\\s+` | Extra whitespace |
| 7 | `.lower()` | Uppercase |

**Characters removed:** {len(raw_text) - len(cleaned_text)} out of {len(raw_text)} ({((len(raw_text) - len(cleaned_text)) / len(raw_text) * 100):.1f}%)

---

## 2. Stop Word Removal

**Tokens before:** {len(cleaned_text.split())}  
**Tokens after:** {len(tokens_no_stopwords)}  
**Removed:** {len(cleaned_text.split()) - len(tokens_no_stopwords)} tokens ({((len(cleaned_text.split()) - len(tokens_no_stopwords)) / len(cleaned_text.split()) * 100):.1f}%)

---

## 3. POS Tagging (spaCy) — Run Before Lemmatization

**Why spaCy was used for POS even though lemmatization uses NLTK:**  
WordNetLemmatizer requires a POS argument. Without it, it defaults to noun
and gives wrong results for verbs. spaCy was used only as a POS provider —
the actual lemmatization is still done by NLTK's WordNetLemmatizer so the
comparison between PorterStemmer and WordNetLemmatizer remains valid.

**First 30 tokens tagged:**

| Token | POS Tag | Description |
|-------|---------|-------------|
{pos_table}

**Confusion encountered here:**  
The original normalizer.py had `pos="v"` hardcoded for every token:
```python
# OLD — wrong approach
return [lemmatizer.lemmatize(t, pos="v") for t in tokens]
```
This was a shortcut that gave better results than defaulting to noun but was
still wrong — not every word is a verb. `better` forced as verb stays `better`
but `better` tagged as adjective by spaCy → `good` which is correct.

**Fix applied:**  
spaCy POS tags are now computed first, converted from spaCy format to WordNet
format using a mapping dictionary, then passed per-token to WordNetLemmatizer:
```python
SPACY_TO_WORDNET = {{"VERB": "v", "NOUN": "n", "ADJ": "a", "ADV": "r"}}
wordnet_pos = SPACY_TO_WORDNET.get(spacy_pos_tag, "n")
lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
```

---

## 4. Stemming vs Lemmatization

### Full Comparison Table (first 20 tokens)

| Original | Stemmed (Porter) | Lemmatized (WordNet + spaCy POS) |
|----------|-----------------|----------------------------------|
{stem_lemma_table}

### Tokens Where Results Differ

| Original | Stemmed | Lemmatized |
|----------|---------|------------|
{diff_table}

### Key Observations

**Observation 1 — Stemmer produces non-real words:**  
PorterStemmer chops suffixes using fixed rules without checking if the result
is a real English word. Words like `carrying` → `carri`, `accident` → `accid`,
`biological` → `biolog` are not real words. This is by design — stemming
doesn't care about linguistic correctness, only about grouping similar words.

**Observation 2 — Lemmatizer returns real dictionary words:**  
WordNetLemmatizer looks up the actual base form in WordNet's vocabulary.
`carrying` → `carry`, `studies` → `study`, `neurons` → `neuron`.
These are real words a human would recognize.

**Observation 3 — The `pos="v"` shortcut problem:**  
When lemmatizer was called with hardcoded `pos="v"` for every token,
`tokenizing` returned `tokenizing` (unchanged) while `carrying` returned
`carry` — inconsistent results because the verb lookup happened to work
for some words and not others. After fixing with actual POS tags, results
became consistent.

**Observation 4 — Same word, different POS, different lemma:**  
The word `train` appears in both news (tourist train = NOUN → `train`)
and could appear as a verb (to train = VERB → `train`). In this corpus
both happen to give the same lemma but `better` shows the difference clearly:
- `better` as ADJ → `good` (correct)
- `better` as VERB → `better` (unchanged, also correct for verbal use)

**When to use which:**

| | PorterStemmer | WordNetLemmatizer |
|-|--------------|-------------------|
| Speed | Fast | Slower (dict lookup) |
| Output | Non-words ok | Real words only |
| POS needed | No | Yes for accuracy |
| Use case | Search engines, IR | NLP pipelines, transformers |
| Urdu/Pashto | Fails completely | No support either — need Stanza |

---

## 5. Tokenization Comparison

**Sample sentence:**  
`"{sample_sentence}"`

### Raw Token Outputs (first 20 tokens each)

{tokenization_text}

### Token Count Summary

| Method | Token Count |
|--------|------------|
{token_count_table}

### Observations

**Observation 1 — Character level is impractical:**  
Every single character becomes a token including spaces. The same sentence
that has {len(tokenization_results.get('word', []))} word-level tokens becomes
{len(tokenization_results.get('character', []))} character tokens. For a
transformer with 512 token limit, a single paragraph would exceed the limit.
No meaningful information is captured at character level — `cat` and `car`
share `c` and `a` tokens but are completely unrelated words.

**Observation 2 — Word level misses subword structure:**  
Clean common words tokenize perfectly at word level. But morphologically
complex words that weren't in the training vocabulary would become `[UNK]`
(unknown token). For Urdu and Pashto where one root generates hundreds of
surface forms through affixation, word-level tokenization would mark the
majority of tokens as unknown.

**Observation 3 — BPE (GPT-2) uses `Ġ` to mark spaces:**  
Initially the `Ġ` character looked like corruption or a bug. It is actually
intentional — GPT-2's BPE tokenizer encodes the space character as `Ġ`
(Unicode U+0120) to mark "there was a space before this token." This lets
the tokenizer work on raw byte streams without pre-splitting on whitespace,
and allows perfect reconstruction of the original text. BERT's WordPiece
does not have this because it pre-splits on whitespace before tokenizing.

**Observation 4 — WordPiece `##` prefix vs BPE `Ġ` prefix:**  
Both mark subword boundaries but differently:
- BPE marks the START of a new word with `Ġ` on the first piece
- WordPiece marks CONTINUATION pieces with `##` on all pieces after the first
"synapses" in BPE:       ['Ġsyn', 'apses']        ← Ġ marks word start

"synapses" in WordPiece: ['syn', '##ap', '##ses']  ← ## marks continuation

**Observation 5 — Both subword methods struggle with non-English:**  
Spanish words from the news article (`ruta`, `tapa`, `ctel`) get fragmented
into meaningless pieces by both BPE and WordPiece because neither tokenizer
was trained sufficiently on Spanish data. This directly predicts what would
happen with Urdu and Pashto — the tokenizers would over-fragment these words
since they were trained predominantly on English text.

**Most important takeaway for company work:**  
For Urdu/Pashto/Punjabi products, a SentencePiece tokenizer must be trained
from scratch on target language data. It learns subword splits based on actual
frequency in your language — not English frequency patterns.

---

## 6. Bag of Words

{bow_text}

### Observations

**Observation 1 — BoW treats entire corpus as one document if not split:**  
Initially the corpus was being passed as one string to `build_bow()`, resulting
in one giant dictionary with all 600+ tokens. After fixing the sentence
splitting, each sentence gets its own frequency dictionary which makes
comparison meaningful.

**Observation 2 — High frequency words reveal domain:**  
Science sentences: `neural`, `network`, `neurons` dominate  
News sentences: `train`, `local`, `authorities` dominate  
Dialogue sentences: `man`, `elderly`, `japanese` dominate  
This shows BoW can distinguish domains through frequency alone.

**Observation 3 — BoW loses everything beyond frequency:**

| What BoW loses | Example from corpus |
|---------------|---------------------|
| Word order | `man bearded` = `bearded man` in BoW |
| Context | `train` (tourist) = `train` (ML) |
| Semantic similarity | `neurons` and `nerve cells` = 0 overlap |
| Negation | `cause unknown` vs `cause known` differ by one token |

**Why this matters:**  
These exact limitations motivated word2vec embeddings, then contextual
embeddings (BERT), then transformers. Every model your company builds on
(Whisper, mBERT, XLM-R) solves these BoW problems through learned
dense vector representations.

---

## 7. Changes Made During Development

| What changed | Why it changed | What it fixed |
|-------------|---------------|---------------|
| Pipeline order: POS moved before lemmatization | Lemmatizer needs POS to work correctly | `tokenizing` now correctly → `tokenize` not `tokenizing` |
| Removed hardcoded `pos="v"` in normalizer.py | Wrong for nouns and adjectives | `better` now → `good` not `better` |
| Added `SPACY_TO_WORDNET` mapping dict | spaCy and WordNet use different POS formats | Correct POS format passed to WordNetLemmatizer |
| Tokenization changed from full corpus to one sentence | Full corpus gave 668 BPE tokens — unreadable | Single sentence gives clean readable comparison |
| BoW changed from full corpus to split sentences | One giant dict hid per-sentence patterns | Each sentence now shows its own top words |
| Added `Ġ` explanation to report | Initially looked like encoding bug | Correct understanding of BPE space encoding |
| BERT truncation warning resolved | Full corpus exceeded 512 token BERT limit | Fixed by passing single sentence to tokenizer |

---

## 8. Key Takeaways

1. **POS before lemmatization** — always. Without POS, WordNetLemmatizer
   defaults to noun and gives wrong results for verbs and adjectives.

2. **Stemming and lemmatization serve different purposes** — stemming for
   fast search/retrieval, lemmatization for any pipeline where token
   quality matters.

3. **Subword tokenization is the industry standard** — BPE (GPT family)
   and WordPiece (BERT family) handle unknown words by splitting into
   known subword pieces. Character and word level are baselines to
   understand, not use in production.

4. **`Ġ` in BPE is not a bug** — it encodes space information so the
   original text can be perfectly reconstructed from tokens.

5. **All classical tools are English-centric** — every tool used today
   (NLTK, spaCy, GPT-2 BPE, BERT WordPiece) performs poorly on
   Urdu/Pashto/Punjabi. Production multilingual pipelines need
   SentencePiece + Stanza + multilingual pretrained models.

6. **BoW limitations directly motivate modern NLP** — every problem
   BoW has (no order, no context, no semantic similarity) is solved
   by the embedding and attention mechanisms in the models your
   company builds on.

---
*Report generated by main.py — Day 1: Classic NLP Pipeline & Tokenization*
"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("report.md written successfully.")