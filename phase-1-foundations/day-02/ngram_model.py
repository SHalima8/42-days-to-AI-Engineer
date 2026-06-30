from collections import defaultdict, Counter
import random
import re


# ── HELPERS ────────────────────────────────────────────────────────────────────

def tokenize(sentence):
    """Lowercase and split into word tokens, stripping punctuation."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    return sentence.split()


def build_ngrams(sentences, n):
    """
    Given a list of sentences and n, return a dict mapping
    each (n-1)-word context tuple → Counter of following words.

    Example (bigram, n=2):
        context ("earthquake",) → Counter({"struck": 4, "occurred": 2, ...})
    """
    model = defaultdict(Counter)
    for sentence in sentences:
        tokens = tokenize(sentence)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i: i + n - 1])   # (n-1) words before
            next_word = tokens[i + n - 1]             # the word we're predicting
            model[context][next_word] += 1
    return model


def laplace_probability(model, context, word, vocab_size):
    """
    P(word | context) with Laplace (add-1) smoothing.

    Without smoothing: unseen pairs get probability 0, which breaks generation.
    With smoothing:    we pretend every word appeared at least once.

    Formula:  (count(context, word) + 1) / (total_after_context + vocab_size)
    """
    context_counts = model[context]
    total = sum(context_counts.values())
    word_count = context_counts.get(word, 0)
    return (word_count + 1) / (total + vocab_size)


def get_next_word(model, context, vocab, vocab_size):
    """
    Given a context tuple, sample the next word weighted by Laplace probability.
    Falls back to a random vocab word if context was never seen.
    """
    if context not in model:
        return random.choice(vocab)

    words = list(model[context].keys())
    probs = [laplace_probability(model, context, w, vocab_size) for w in words]
    total = sum(probs)
    probs = [p / total for p in probs]             # normalize to sum to 1
    return random.choices(words, weights=probs, k=1)[0]


# ── MAIN PUBLIC FUNCTIONS ──────────────────────────────────────────────────────

def train_ngram(sentences, n=2):
    """Train an n-gram model. Returns (model dict, vocab list, vocab_size)."""
    model = build_ngrams(sentences, n)
    # build vocabulary from all tokens across all sentences
    all_tokens = []
    for sentence in sentences:
        all_tokens.extend(tokenize(sentence))
    vocab = list(set(all_tokens))
    return model, vocab, len(vocab)


def generate_text(model, vocab, vocab_size, seed_words, n=2, num_words=20):
    """
    Generate text by predicting one word at a time.

    seed_words: list of starting words (must be length n-1)
    The model only ever looks back (n-1) words — that is the fixed window.
    """
    generated = list(seed_words)
    for _ in range(num_words):
        context = tuple(generated[-(n - 1):])      # only look back (n-1) words
        next_word = get_next_word(model, context, vocab, vocab_size)
        generated.append(next_word)
    return " ".join(generated)


def show_top_predictions(model, vocab_size, context_tuple, topn=5):
    """
    Print the top-n most probable next words for a given context,
    with their Laplace-smoothed probabilities.
    """
    if context_tuple not in model:
        print(f"  Context {context_tuple} not found in training data.")
        return

    words = list(model[context_tuple].keys())
    probs = {w: laplace_probability(model, context_tuple, w, vocab_size)
             for w in words}
    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:topn]

    print(f"\n  Next-word probabilities after {context_tuple}:")
    for word, prob in top:
        print(f"    '{word}': {prob:.4f}")


def demonstrate_fixed_window_limit(bigram_model, trigram_model, vocab,
                                   vocab_size_bi, vocab_size_tri, sentences):
    """
    Show concretely why a fixed window misses long-range context.

    We pick a sentence from the corpus that has a subject far from its verb,
    then show the bigram can only see 1 word back while the trigram sees 2.
    Neither can connect information across a full sentence.
    """
    print("\n  ── Fixed-Window Limitation Demo ──")
    print("  Bigram window  : 1 word back")
    print("  Trigram window : 2 words back")
    print("  Transformer    : entire sequence (that's Day 3, task 3)\n")

    # Find the longest sentence to show the gap
    tokenized = [tokenize(s) for s in sentences]
    longest = max(tokenized, key=len)
    length = len(longest)

    print(f"  Longest sentence in corpus: {length} tokens")
    print(f"  A bigram predicting token #{length} only sees token #{length - 1}")
    print(f"  Everything before that is invisible to the model.")
    print(f"\n  Example: '...{' '.join(longest[-4:-1])} [PREDICT]'")
    print(f"  Bigram context : ('{longest[-2]}')")
    print(f"  Trigram context: ('{longest[-3]}', '{longest[-2]}')")
    print(f"  Token #{1} ('{longest[0]}') has zero influence on this prediction.")


# ── PRINT HELPERS ──────────────────────────────────────────────────────────────

def print_ngram_results(bigram_model, trigram_model,
                        bi_vocab, bi_vocab_size,
                        tri_vocab, tri_vocab_size,
                        sentences):
    """Master print function — called from main.py."""

    print("\n── Bigram Model ──")
    print(f"  Unique bigram contexts : {len(bigram_model)}")
    print(f"  Vocabulary size        : {bi_vocab_size}")

    print("\n── Trigram Model ──")
    print(f"  Unique trigram contexts: {len(trigram_model)}")
    print(f"  Vocabulary size        : {tri_vocab_size}")

    # Show next-word probability distributions
    show_top_predictions(bigram_model, bi_vocab_size,
                         context_tuple=("earthquake",))
    show_top_predictions(trigram_model, tri_vocab_size,
                         context_tuple=("the", "earthquake"))

    # Generate sample text
    print("\n── Generated Text (Bigram, seed='the earthquake') ──")
    random.seed(42)
    bi_text = generate_text(bigram_model, bi_vocab, bi_vocab_size,
                            seed_words=["the", "earthquake"], n=2, num_words=20)
    print(f"  {bi_text}")

    print("\n── Generated Text (Trigram, seed='the earthquake') ──")
    random.seed(42)
    tri_text = generate_text(trigram_model, tri_vocab, tri_vocab_size,
                             seed_words=["the", "earthquake"], n=3, num_words=20)
    print(f"  {tri_text}")

    # Fixed window limitation
    demonstrate_fixed_window_limit(bigram_model, trigram_model,
                                   bi_vocab, bi_vocab_size, tri_vocab_size,
                                   sentences)