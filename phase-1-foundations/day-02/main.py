from scrapper import scrape_all
from cleaner import clean_text, split_into_sentences
from normalizer import remove_stopwords, stem_text, lemmatize_text_with_pos
from pos_tagger import pos_tag_text
from tokenizer_compare import compare_tokenizers
from bow import build_bow
from ngram_model import (train_ngram, print_ngram_results)
from tfidf import (compute_tfidf_manual, compute_tfidf_sklearn,
                   retrieve_similar_sentences, compare_synonym_pairs_tfidf,
                   print_tfidf_comparison, print_retrieval_results,
                   print_synonym_tfidf)
from word2vec_model import (train_word2vec, compare_synonym_pairs_w2v,
                             nearest_neighbors, word_analogy,
                             show_polysemy_failure, print_w2v_results)
from lstm import run_lstm_and_get_gradients, print_lstm_results
from bert_embeddings import (load_bert, run_bert_polysemy,
                              run_bert_extra_contexts, print_bert_results)

from comparison_table import build_comparison_table, print_comparison_table
from tsne_plot import run_tsne_comparison

def main():

    # ── STEP 1: SCRAPE ─────────────────────────────────────────────────
    print("\n========== SCRAPING ==========")
    raw_text = scrape_all()

    # ── STEP 2: CLEAN ──────────────────────────────────────────────────
    print("\n========== CLEANING ==========")
    cleaned_text = clean_text(raw_text)
    sentences = split_into_sentences(cleaned_text)
    print(f"Total sentences after splitting: {len(sentences)}")
    print(f"Sample sentence: {sentences[0]}")

    # ── STEP 3: POS TAGGING ────────────────────────────────────────────
    print("\n========== POS TAGGING ==========")
    pos_results = pos_tag_text(cleaned_text)
    for token, pos, desc in pos_results[:15]:
        print(f"  {token:20} | {pos:10} | {desc}")

    # ── STEP 4: STOPWORDS + STEM + LEMMATIZE ───────────────────────────
    print("\n========== NORMALIZATION ==========")
    tokens = cleaned_text.split()
    tokens_no_stopwords = remove_stopwords(tokens)
    stemmed = stem_text(tokens_no_stopwords)
    lemmatized = lemmatize_text_with_pos(pos_results, tokens_no_stopwords)
    print(f"Tokens before stopword removal: {len(tokens)}")
    print(f"Tokens after stopword removal:  {len(tokens_no_stopwords)}")
    print("\nSample stemming vs lemmatization:")
    for o, s, l in zip(tokens_no_stopwords[:10], stemmed[:10], lemmatized[:10]):
        print(f"  {o:20} | stem: {s:20} | lemma: {l}")

    # ── STEP 5: TOKENIZATION COMPARISON ───────────────────────────────
    print("\n========== TOKENIZATION COMPARISON ==========")
    sample_sentence = sentences[5]
    print(f"Sample: {sample_sentence}")
    tokenization_results = compare_tokenizers(sample_sentence)

    # ── STEP 6: BAG OF WORDS ───────────────────────────────────────────
    print("\n========== BAG OF WORDS ==========")
    bow_results = build_bow(sentences[:5])
    for i, bow in enumerate(bow_results):
        top = dict(sorted(bow.items(), key=lambda x: x[1], reverse=True)[:5])
        print(f"  Sentence {i+1}: {top}")

    # ── STEP 7: TF-IDF ─────────────────────────────────────────────────
    print("\n========== TF-IDF: MANUAL COMPUTATION ==========")
    manual_results = compute_tfidf_manual(sentences)

    print("\n========== TF-IDF: SKLEARN COMPUTATION ==========")
    sklearn_results, vectorizer, matrix = compute_tfidf_sklearn(sentences)

    print_tfidf_comparison(sentences, manual_results, sklearn_results)

    # ── STEP 8: RETRIEVAL ──────────────────────────────────────────────
    query = "neural network brain signals learning"
    ranked = retrieve_similar_sentences(query, sentences, vectorizer, matrix)
    print_retrieval_results(query, ranked)

    # ── STEP 9: SYNONYM PAIRS — pick from actual corpus vocabulary ─────
    # these are words that actually exist in our corpus
    synonym_pairs = [
    ("earthquake", "fault"),       # both in news
    ("neural", "biological"),      # both in science
    ("learning", "training"),      # both in science
    ("network", "model"),          # both in science
    ("dialogue", "conversation"),  # both in dialogue
]

    print("\n========== TF-IDF: SYNONYM COMPARISON ==========")
    tfidf_synonym_results = compare_synonym_pairs_tfidf(synonym_pairs, vectorizer, matrix)
    print_synonym_tfidf(tfidf_synonym_results)

    # ── STEP 10: WORD2VEC ──────────────────────────────────────────────
    print("\n========== TRAINING WORD2VEC ==========")
    w2v_model = train_word2vec(sentences)
    print(f"Vocabulary size: {len(w2v_model.wv)}")

    # nearest neighbors
    print("\n--- Nearest Neighbors ---")
    for word in ["earthquake", "neural", "dialogue", "learning"]:
        neighbors = nearest_neighbors(w2v_model, word, topn=3)
        print(f"  {word}: {neighbors}")

    # word analogy
    print("\n--- Word Analogy ---")
    result = word_analogy(w2v_model,
                          positive=["neural", "learning"],
                          negative=["earthquake"])
    print(f"  neural - earthquake + learning = {result}")

    # ── STEP 11: W2V SYNONYM COMPARISON ───────────────────────────────
    w2v_synonym_results = compare_synonym_pairs_w2v(w2v_model, synonym_pairs)
    print_w2v_results(tfidf_synonym_results, w2v_synonym_results)

    # ── STEP 12: POLYSEMY FAILURE ──────────────────────────────────────
    print("\n========== POLYSEMY FAILURE ==========")
    # 'network' appears as both computer network and biological network
    # in our corpus — same word, two meanings, one vector
    polysemy_result = show_polysemy_failure(w2v_model, "network")
    print(f"  Word: {polysemy_result['word']}")
    print(f"  Vector (first 5 values): {polysemy_result['vector_snippet']}")
    print(f"  Nearest neighbors: {polysemy_result['nearest_neighbors']}")
    print(f"  Explanation: {polysemy_result['explanation']}")


    # ── STEP 13: N-GRAM LANGUAGE MODEL ────────────────────────────────
    print("\n========== N-GRAM LANGUAGE MODEL ==========")
    bigram_model,  bi_vocab,  bi_vocab_size  = train_ngram(sentences, n=2)
    trigram_model, tri_vocab, tri_vocab_size = train_ngram(sentences, n=3)
    print_ngram_results(bigram_model, trigram_model,
                        bi_vocab,  bi_vocab_size,
                        tri_vocab, tri_vocab_size,
                        sentences)
    
    # ── STEP 14: LSTM FORWARD/BACKWARD PASS ───────────────────────────
    print("\n========== LSTM: SEQUENTIAL MODEL & GRADIENT DECAY ==========")
    lstm_results = run_lstm_and_get_gradients(sentences)
    print_lstm_results(lstm_results)

    # ── STEP 15: BERT CONTEXTUAL EMBEDDINGS ───────────────────────────
    print("\n========== BERT: CONTEXTUAL EMBEDDINGS ==========")
    tokenizer, bert_model = load_bert()
    bert_results = run_bert_polysemy(tokenizer, bert_model)
    print_bert_results(bert_results)

    # also get extra context vectors — needed for Task 5 t-SNE
    bert_sentences, bert_labels, bert_vectors = run_bert_extra_contexts(
        tokenizer, bert_model, target_word="network"
    )
    
    # ── STEP 16: COMPARISON TABLE ──────────────────────────────────────
    print("\n========== COMPARISON TABLE: ALL METHODS ==========")
    print("  Computing similarities for all word pairs...")
    comparison_results = build_comparison_table(
        synonym_pairs,
        vectorizer, matrix,
        w2v_model,
        bigram_model, bi_vocab_size,
        tokenizer, bert_model
    )
    print_comparison_table(comparison_results)

    # ── STEP 17: t-SNE COMPARISON PLOT ────────────────────────────────
    print("\n========== t-SNE: STATIC vs CONTEXTUAL CLUSTERING ==========")
    run_tsne_comparison(w2v_model, tokenizer, bert_model, target_word="network")

if __name__ == "__main__":
    main()