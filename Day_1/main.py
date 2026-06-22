from scraper import scrape_all
from cleaner import clean_text
from normalizer import remove_stopwords, stem_text, lemmatize_text_with_pos
from tokenizer_compare import compare_tokenizers
from pos_tagger import pos_tag_text
from bow import build_bow
from report_generator import generate_report


def main():
    # Step 1: scrape and read corpus
    print("\n========== SCRAPING CORPUS ==========")
    raw_text = scrape_all()
    print(raw_text[:300])

    # Step 2: clean
    print("\n========== CLEANING ==========")
    cleaned_text = clean_text(raw_text)
    print(cleaned_text[:300])

    # Step 3: stopword removal
    print("\n========== STOPWORD REMOVAL ==========")
    tokens = cleaned_text.split()
    tokens_no_stopwords = remove_stopwords(tokens)
    print(tokens_no_stopwords[:20])

    # Step 4: POS tagging FIRST on cleaned text
    print("\n========== POS TAGGING ==========")
    pos_results = pos_tag_text(cleaned_text)
    for token, pos, description in pos_results[:20]:
        print(f"{token:20} | {pos:10} | {description}")

    # Step 5: stemming and lemmatization
    # stemming uses tokens directly — no POS needed
    # lemmatization uses POS results from step 4
    print("\n========== STEMMING vs LEMMATIZATION ==========")
    stemmed = stem_text(tokens_no_stopwords)
    lemmatized = lemmatize_text_with_pos(pos_results, tokens_no_stopwords)
    for original, stem, lemma in zip(tokens_no_stopwords[:10], stemmed[:10], lemmatized[:10]):
        print(f"{original:20} | stem: {stem:20} | lemma: {lemma}")

    # Step 6: tokenization comparison — one sentence
    print("\n========== TOKENIZATION COMPARISON ==========")
    sentences_for_comparison = [s.strip() for s in cleaned_text.split(".") if len(s.strip()) > 30]
    sample_sentence = sentences_for_comparison[0]
    print(f"Sample sentence used: {sample_sentence}")
    tokenization_results = compare_tokenizers(sample_sentence)

    # Step 7: Bag of Words
    print("\n========== BAG OF WORDS ==========")
    bow_sentences = [s.strip() for s in cleaned_text.split(".") if len(s.strip()) > 20]
    bow_results = build_bow(bow_sentences[:5])
    for i, bow in enumerate(bow_results):
        print(f"Sentence {i+1}: {bow}")

    # Step 8: find real differences for report
    real_stem_lemma_examples = [
        (o, s, l) for o, s, l in zip(tokens_no_stopwords[:50], stemmed[:50], lemmatized[:50])
        if s != l and s != o  # only show where stemmer actually changed something
    ]

    # Step 9: generate report
    print("\n========== GENERATING REPORT ==========")
    generate_report(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        tokens_no_stopwords=tokens_no_stopwords,
        stemmed=stemmed,
        lemmatized=lemmatized,
        tokenization_results=tokenization_results,
        sample_sentence=sample_sentence,
        pos_results=pos_results,
        bow_results=bow_results,
        bow_sentences=bow_sentences,
        real_stem_lemma_examples=real_stem_lemma_examples
    )
    print("Report saved to report.md")

if __name__ == "__main__":
    main()