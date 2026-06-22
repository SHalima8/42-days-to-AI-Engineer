from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def train_word2vec(sentences):
    """
    sentences: list of strings
    Word2Vec expects list of lists of tokens
    """
    tokenized = [s.lower().split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=100,  # each word gets a 100-dim vector
        window=5,         # context window — look 5 words left and right
        min_count=2,      # ignore words appearing less than 2 times
        workers=4,
        epochs=50         # train for 50 passes over corpus
    )
    return model


def word_similarity_w2v(model, word1, word2):
    """Cosine similarity between two words using Word2Vec vectors."""
    try:
        sim = model.wv.similarity(word1, word2)
        return round(float(sim), 4)
    except KeyError as e:
        return f"N/A — {e} not in vocabulary"


def nearest_neighbors(model, word, topn=5):
    """Find most similar words to a given word."""
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        return f"'{word}' not in vocabulary"


def word_analogy(model, positive, negative, topn=3):
    """
    Word analogy: king - man + woman = ?
    positive = ['king', 'woman']
    negative = ['man']
    """
    try:
        return model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    except KeyError as e:
        return f"N/A — {e} not in vocabulary"


def get_word_vector(model, word):
    """Get the raw vector for a word."""
    try:
        return model.wv[word]
    except KeyError:
        return None


def compare_synonym_pairs_w2v(model, pairs):
    """Compare same synonym pairs using Word2Vec instead of TF-IDF."""
    results = []
    for word1, word2 in pairs:
        sim = word_similarity_w2v(model, word1, word2)
        results.append((word1, word2, sim))
    return results


def show_polysemy_failure(model, word):
    """
    Show that Word2Vec gives ONE vector for a word
    regardless of which meaning is being used.
    """
    vec = get_word_vector(model, word)
    if vec is None:
        return f"'{word}' not in vocabulary"
    neighbors = nearest_neighbors(model, word, topn=5)
    return {
        "word": word,
        "vector_snippet": vec[:5].tolist(),  # first 5 values of the vector
        "nearest_neighbors": neighbors,
        "explanation": f"Word2Vec gives '{word}' ONE fixed vector regardless of context. "
               f"In this corpus 'network' appears both as a computational model "
               f"(artificial neural network) and as a biological structure "
               f"(biological neural network in the brain). "
               f"Word2Vec cannot distinguish these two senses — same vector both times."
    }


def print_w2v_results(synonym_pairs_tfidf, synonym_pairs_w2v):
    print("\n========== TF-IDF vs WORD2VEC SYNONYM COMPARISON ==========")
    print(f"  {'Word 1':<15} {'Word 2':<15} {'TF-IDF Sim':<15} {'W2V Sim'}")
    print(f"  {'-'*60}")
    for (w1, w2, tfidf_sim), (_, _, w2v_sim) in zip(synonym_pairs_tfidf, synonym_pairs_w2v):
        print(f"  {w1:<15} {w2:<15} {str(tfidf_sim):<15} {w2v_sim}")