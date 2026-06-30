import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

nlp = spacy.load("en_core_web_sm")

SPACY_TO_WORDNET = {
    "VERB": "v",
    "NOUN": "n",
    "ADJ":  "a",
    "ADV":  "r"
}

def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop_words]

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def lemmatize_text_with_pos(pos_results, tokens_no_stopwords):
    lemmatizer = WordNetLemmatizer()
    pos_lookup = {}
    for token, pos_tag, _ in pos_results:
        wordnet_pos = SPACY_TO_WORDNET.get(pos_tag, "n")
        pos_lookup[token] = wordnet_pos

    lemmatized = []
    for token in tokens_no_stopwords:
        wordnet_pos = pos_lookup.get(token, "n")
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized.append(lemma)
    return lemmatized