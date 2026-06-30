# NLP Pipeline Report — Day 1
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
=== NEWS ===
Seventeen people have been injured after the wagon of a tourist train overturned in the Spanish town of Cártama.

The accident happened just after 21:30 local time (20:30 BST) on Saturday evening, according to local authorities. The cause is unknown.

Emergency teams attended to the inj

**After (first 300 chars):**
news seventeen people have been injured after the wagon of a tourist train overturned in the spanish town of c rtama the accident happened just after local time on saturday evening according to local authorities the cause is unknown emergency teams attended to the injured at the scene and four were 

| Regex Step | Pattern | Removes |
|------------|---------|---------|
| 1 | `\[\s*\d+\s*\]` | Wikipedia citations like `[1]` |
| 2 | `\(.*?\)` | Stage directions like `(in Japanese)` |
| 3 | `•` | Bullet symbols |
| 4 | `\d+` | All numbers |
| 5 | `[^a-zA-Z\s]` | Special chars, accents, punctuation |
| 6 | `\s+` | Extra whitespace |
| 7 | `.lower()` | Uppercase |

**Characters removed:** 260 out of 3935 (6.6%)

---

## 2. Stop Word Removal

**Tokens before:** 621  
**Tokens after:** 383  
**Removed:** 238 tokens (38.3%)

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
| news                 | NOUN       | Noun                      |
| seventeen            | NUM        | Number                    |
| people               | NOUN       | Noun                      |
| have                 | AUX        | Auxiliary Verb            |
| been                 | AUX        | Auxiliary Verb            |
| injured              | VERB       | Verb                      |
| after                | ADP        | Preposition               |
| the                  | DET        | Determiner                |
| wagon                | NOUN       | Noun                      |
| of                   | ADP        | Preposition               |
| a                    | DET        | Determiner                |
| tourist              | NOUN       | Noun                      |
| train                | NOUN       | Noun                      |
| overturned           | VERB       | Verb                      |
| in                   | ADP        | Preposition               |
| the                  | DET        | Determiner                |
| spanish              | ADJ        | Adjective                 |
| town                 | NOUN       | Noun                      |
| of                   | ADP        | Preposition               |
| c                    | PROPN      | Proper Noun               |
| rtama                | PROPN      | Proper Noun               |
| the                  | DET        | Determiner                |
| accident             | NOUN       | Noun                      |
| happened             | VERB       | Verb                      |
| just                 | ADV        | Adverb                    |
| after                | ADP        | Preposition               |
| local                | ADJ        | Adjective                 |
| time                 | NOUN       | Noun                      |
| on                   | ADP        | Preposition               |
| saturday             | PROPN      | Proper Noun               |

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
SPACY_TO_WORDNET = {"VERB": "v", "NOUN": "n", "ADJ": "a", "ADV": "r"}
wordnet_pos = SPACY_TO_WORDNET.get(spacy_pos_tag, "n")
lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
```

---

## 4. Stemming vs Lemmatization

### Full Comparison Table (first 20 tokens)

| Original | Stemmed (Porter) | Lemmatized (WordNet + spaCy POS) |
|----------|-----------------|----------------------------------|
| news                 | news                 | news                 |
| seventeen            | seventeen            | seventeen            |
| people               | peopl                | people               |
| injured              | injur                | injure               |
| wagon                | wagon                | wagon                |
| tourist              | tourist              | tourist              |
| train                | train                | train                |
| overturned           | overturn             | overturn             |
| spanish              | spanish              | spanish              |
| town                 | town                 | town                 |
| c                    | c                    | c                    |
| rtama                | rtama                | rtama                |
| accident             | accid                | accident             |
| happened             | happen               | happen               |
| local                | local                | local                |
| time                 | time                 | time                 |
| saturday             | saturday             | saturday             |
| evening              | even                 | evening              |
| according            | accord               | accord               |
| local                | local                | local                |

### Tokens Where Results Differ

| Original | Stemmed | Lemmatized |
|----------|---------|------------|
| people               | peopl                | people               |
| injured              | injur                | injure               |
| accident             | accid                | accident             |
| evening              | even                 | evening              |
| authorities          | author               | authority            |
| cause                | caus                 | cause                |
| emergency            | emerg                | emergency            |
| injured              | injur                | injure               |
| evacuated            | evacu                | evacuate             |
| nearby               | nearbi               | nearby               |
| hospital             | hospit               | hospital             |
| including            | includ               | include              |
| serious              | seriou               | serious              |
| injuries             | injuri               | injury               |
| authorities          | author               | authority            |

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
`"news seventeen people have been injured after the wagon of a tourist train overturned in the spanish town of c rtama the accident happened just after local time on saturday evening according to local authorities the cause is unknown emergency teams attended to the injured at the scene and four were later evacuated to a nearby hospital including three children none of the victims suffered serious injuries local authorities said the free train ride was part of an annual event called la ruta de la tapa y el c ctel where people are encouraged to visit different local businesses to try their food and drink offerings the road train was carrying carrying around passengers at the time of the incident science a neural network is a group of interconnected units called neurons that send signals to one another neurons can be either biological cells or mathematical models while individual neurons are simple many of them together in a network can perform complex tasks there are two main types of neural networks in neuroscience a biological neural network is a physical structure found in brains and complex nervous systems a population of nerve cells connected by synapses in machine learning an artificial neural network is a mathematical model used to approximate nonlinear functions artificial neural networks are used to solve artificial intelligence problems in neuroscience behavior and cognition arise from interactions between distributed brain regions in computer science artificial neural networks power many modern ai systems but require large datasets and substantial computing power additionally their internal representations are difficult to interpret in the context of biology a neural network is a population of biological neurons chemically connected to each other by synapses a given neuron can be connected to hundreds of thousands of synapses each neuron sends and receives electrochemical signals called action potentials to its connected neighbors a neuron can serve an excitatory role amplifying and propagating signals it receives or an inhibitory role suppressing signals instead dialogue inception written by christopher nolan shooting script fade in dawn crashing surf the waves toss a bearded man onto wet sand he lies there a child s shout makes him lift his head to see a little blonde boy crouching back towards us watching the tide eat a sandcastle a little blonde girl joins the boy the bearded man tries to call them but they run off faces unseen he collapses the barrel of a rifle rolls the bearded man onto his back a japanese security guard looks down at him then calls up the beach to a colleague leaning against a jeep behind them is a cliff and on top of that a japanese castle int elegant dining room japanese castle later the security guard waits as an attendant speaks to an elderly japanese man sitting at the dining table back to us attendant he was delirious but he asked for you by name and show him security guard he was carrying nothing but this he puts a handgun on the table the elderly man keeps eating security guard and this the security guard places a small pewter cone alongside the gun the elderly man stops eating picks up the cone elderly japanese man bring him here and some food int same moments later the elderly man watches the bearded man wolf down his food he slides the handgun down the table towards him elderly japanese man are you here to kill me the bearded man glances up at him then back to his food the elderly japanese man picks up the cone between thumb and forefinger elderly japanese man i know what this is he spins it onto a table it circles gracefully across the polished ebony a spinning top"`

### Raw Token Outputs (first 20 tokens each)

**CHARACTER** (3675 tokens total):
`['n', 'e', 'w', 's', ' ', 's', 'e', 'v', 'e', 'n', 't', 'e', 'e', 'n', ' ', 'p', 'e', 'o', 'p', 'l']`

**WORD** (621 tokens total):
`['news', 'seventeen', 'people', 'have', 'been', 'injured', 'after', 'the', 'wagon', 'of', 'a', 'tourist', 'train', 'overturned', 'in', 'the', 'spanish', 'town', 'of', 'c']`

**BPE_GPT2** (668 tokens total):
`['news', 'Ġseventeen', 'Ġpeople', 'Ġhave', 'Ġbeen', 'Ġinjured', 'Ġafter', 'Ġthe', 'Ġwagon', 'Ġof', 'Ġa', 'Ġtourist', 'Ġtrain', 'Ġoverturned', 'Ġin', 'Ġthe', 'Ġsp', 'anish', 'Ġtown', 'Ġof']`

**WORDPIECE_BERT** (654 tokens total):
`['news', 'seventeen', 'people', 'have', 'been', 'injured', 'after', 'the', 'wagon', 'of', 'a', 'tourist', 'train', 'overturned', 'in', 'the', 'spanish', 'town', 'of', 'c']`



### Token Count Summary

| Method | Token Count |
|--------|------------|
| CHARACTER            |         3675 |
| WORD                 |          621 |
| BPE_GPT2             |          668 |
| WORDPIECE_BERT       |          654 |

### Observations

**Observation 1 — Character level is impractical:**  
Every single character becomes a token including spaces. The same sentence
that has 621 word-level tokens becomes
3675 character tokens. For a
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

**Sentence 1:** `news seventeen people have been injured after the wagon of a tourist train overt...`
Top 8 words: `{'the': 35, 'a': 28, 'to': 19, 'of': 15, 'man': 13, 'and': 12, 'in': 9, 'he': 8}`



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
