import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_tf(sentence):
    tokens = sentence.lower().split()
    count = Counter(tokens)
    total = len(tokens)
    return {word: freq / total for word, freq in count.items()}


def compute_idf(sentences):
    total_docs = len(sentences)
    word_doc_count = Counter()
    for sentence in sentences:
        unique_words = set(sentence.lower().split())
        for word in unique_words:
            word_doc_count[word] += 1
    idf = {}
    for word, doc_count in word_doc_count.items():
        idf[word] = math.log(total_docs / doc_count)
    return idf


def compute_tfidf_manual(sentences):
    idf = compute_idf(sentences)
    tfidf_results = []
    for sentence in sentences:
        tf = compute_tf(sentence)
        tfidf = {word: round(tf_score * idf.get(word, 0), 4)
                 for word, tf_score in tf.items()}
        tfidf_sorted = dict(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))
        tfidf_results.append(tfidf_sorted)
    return tfidf_results


def compute_tfidf_sklearn(sentences):
    # stop_words='english' removes the/is/of etc automatically
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    sklearn_results = []
    for i, row in enumerate(matrix):
        scores = zip(feature_names, row.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_words = {word: round(score, 4) for word, score in sorted_scores if score > 0}
        sklearn_results.append(top_words)

    return sklearn_results, vectorizer, matrix


def retrieve_similar_sentences(query, sentences, vectorizer, matrix):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(
        zip(sentences, similarities),
        key=lambda x: x[1], reverse=True
    )
    return [(sent[:80], round(float(score), 4)) for sent, score in ranked]


def compare_synonym_pairs_tfidf(pairs, vectorizer, matrix):
    vocab = vectorizer.vocabulary_
    results = []
    for word1, word2 in pairs:
        if word1 not in vocab or word2 not in vocab:
            results.append((word1, word2, "N/A — not in vocabulary"))
            continue
        idx1 = vocab[word1]
        idx2 = vocab[word2]
        vec1 = matrix[:, idx1].toarray()
        vec2 = matrix[:, idx2].toarray()
        sim = cosine_similarity(vec1.T, vec2.T)[0][0]
        results.append((word1, word2, round(float(sim), 4)))
    return results


def print_tfidf_comparison(sentences, manual_results, sklearn_results):
    print("\n========== MANUAL TF-IDF vs SKLEARN ==========")
    for i, (manual, sklearn) in enumerate(zip(manual_results[:5], sklearn_results[:5])):
        print(f"\nSentence {i+1}: {sentences[i][:70]}...")
        print(f"  Top 3 Manual : {list(manual.items())[:3]}")
        print(f"  Top 3 Sklearn: {list(sklearn.items())[:3]}")


def print_retrieval_results(query, ranked):
    print(f"\n========== RETRIEVAL: query = '{query}' ==========")
    for i, (sent, score) in enumerate(ranked[:5]):
        print(f"  Rank {i+1} (score={score}): {sent}...")


def print_synonym_tfidf(synonym_results):
    print("\n========== SYNONYM PAIR SIMILARITY (TF-IDF) ==========")
    print(f"  {'Word 1':<15} {'Word 2':<15} {'Cosine Similarity'}")
    print(f"  {'-'*50}")
    for word1, word2, sim in synonym_results:
        print(f"  {word1:<15} {word2:<15} {sim}")