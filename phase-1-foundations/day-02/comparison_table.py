import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ── HELPERS ────────────────────────────────────────────────────────────────────

def cosine_between(vec1, vec2):
    """Cosine similarity between two 1D numpy arrays."""
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    return float(sklearn_cosine(v1, v2)[0][0])


# ── TF-IDF SIMILARITY ──────────────────────────────────────────────────────────

def tfidf_word_similarity(word1, word2, vectorizer, matrix):
    """
    Find all sentences containing word1 or word2.
    Average their TF-IDF vectors and compute cosine similarity.
    This is the same approach you used in Day 2.
    """
    feature_names = vectorizer.get_feature_names_out()
    vocab = list(feature_names)

    if word1 not in vocab or word2 not in vocab:
        return None

    idx1 = vocab.index(word1)
    idx2 = vocab.index(word2)

    vec1 = np.array(matrix[:, idx1].todense()).flatten()
    vec2 = np.array(matrix[:, idx2].todense()).flatten()

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ── WORD2VEC SIMILARITY ────────────────────────────────────────────────────────

def w2v_word_similarity(word1, word2, w2v_model):
    """
    Direct cosine similarity between two Word2Vec vectors.
    Returns None if either word is not in vocabulary.
    """
    try:
        return float(w2v_model.wv.similarity(word1, word2))
    except KeyError:
        return None


# ── N-GRAM SIMILARITY ──────────────────────────────────────────────────────────

def ngram_context_overlap(word1, word2, bigram_model, vocab_size):
    """
    N-gram models don't produce word vectors — they produce probability
    distributions over the next word.

    We measure similarity by comparing the next-word distributions of
    word1 and word2 using cosine similarity between their probability vectors.

    If 'neural' and 'network' both tend to be followed by similar words,
    their distributions will be similar.
    """
    from collections import Counter

    def get_prob_vector(word, model, vocab_size, all_words):
        context = (word,)
        counts  = model.get(context, Counter())
        total   = sum(counts.values())
        if total == 0:
            return np.zeros(len(all_words))
        # Laplace smoothed probability for each word in vocabulary
        vec = np.array([(counts.get(w, 0) + 1) / (total + vocab_size)
                        for w in all_words])
        return vec

    all_words = list(set(
        w for context in bigram_model for w in bigram_model[context]
    ))

    if not all_words:
        return None

    vec1 = get_prob_vector(word1, bigram_model, vocab_size, all_words)
    vec2 = get_prob_vector(word2, bigram_model, vocab_size, all_words)

    if vec1.sum() == 0 or vec2.sum() == 0:
        return None

    return cosine_between(vec1, vec2)


# ── BERT SIMILARITY ────────────────────────────────────────────────────────────

def bert_word_similarity(word1, word2, tokenizer, bert_model):
    """
    Get BERT vectors for word1 and word2 each in a neutral sentence,
    then compute cosine similarity.

    We use a simple carrier sentence so BERT has some context to work with.
    """
    from bert_embeddings import get_word_embedding

    # neutral carrier sentences
    sent1 = f"The word {word1} appears in this sentence."
    sent2 = f"The word {word2} appears in this sentence."

    try:
        vec1, _, _ = get_word_embedding(sent1, word1, tokenizer, bert_model)
        vec2, _, _ = get_word_embedding(sent2, word2, tokenizer, bert_model)
        return cosine_between(vec1, vec2)
    except ValueError:
        return None


# ── MASTER COMPARISON TABLE ────────────────────────────────────────────────────

def build_comparison_table(synonym_pairs,
                            vectorizer, tfidf_matrix,
                            w2v_model,
                            bigram_model, bigram_vocab_size,
                            tokenizer, bert_model):
    """
    For each word pair, compute similarity using all four methods.
    Returns a list of dicts — one per word pair.
    """
    results = []

    for word1, word2 in synonym_pairs:
        row = {
            "pair"   : (word1, word2),
            "tfidf"  : tfidf_word_similarity(word1, word2, vectorizer, tfidf_matrix),
            "w2v"    : w2v_word_similarity(word1, word2, w2v_model),
            "ngram"  : ngram_context_overlap(word1, word2, bigram_model, bigram_vocab_size),
            "bert"   : bert_word_similarity(word1, word2, tokenizer, bert_model),
        }
        results.append(row)
        print(f"  ✓ {word1:15} vs {word2:15} — done")

    return results


# ── PRINT HELPERS ──────────────────────────────────────────────────────────────

def print_comparison_table(results):
    """Print a clean formatted comparison table."""

    print(f"\n{'Pair':<30} {'TF-IDF':>10} {'Word2Vec':>10} {'N-Gram':>10} {'BERT':>10}")
    print("─" * 72)

    for row in results:
        w1, w2  = row["pair"]
        pair_str = f"{w1} vs {w2}"

        def fmt(val):
            return f"{val:.4f}" if val is not None else "  N/A  "

        print(f"  {pair_str:<28} {fmt(row['tfidf']):>10} "
              f"{fmt(row['w2v']):>10} {fmt(row['ngram']):>10} "
              f"{fmt(row['bert']):>10}")

    print("─" * 72)

    print(f"""
── How to Read This Table ──
  Each number is a cosine similarity score (0 = unrelated, 1 = identical).

  TF-IDF  : Measures if two words appear in similar documents.
             High score = both words used in same context sentences.
             Does NOT understand meaning — just document co-occurrence.

  Word2Vec: Measures if two words appear near similar words in training.
             Captures semantic similarity but gives one fixed vector per word.

  N-Gram  : Measures if two words tend to be followed by similar next words.
             Not a true semantic measure — purely distributional pattern.

  BERT    : Measures similarity of contextual representations.
             Most semantically aware — but in neutral carrier sentences,
             scores here reflect general word sense rather than deep context.

── Key Takeaway ──
  No single method is best at everything.
  TF-IDF and N-Gram capture surface patterns.
  Word2Vec captures semantic proximity.
  BERT captures contextual meaning — most powerful for polysemy.
""")