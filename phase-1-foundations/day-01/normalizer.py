import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# spaCy POS → WordNet POS format
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
    """
    Uses POS tags from spaCy 
    to correctly lemmatize using WordNetLemmatizer.
    
    pos_results: list of (token, pos_tag, description) from pos_tagger.py
    tokens_no_stopwords: the filtered token list we want to lemmatize
    """
    lemmatizer = WordNetLemmatizer()
    
    # build a lookup dict from pos_results: token → wordnet POS
    # pos_results contains ALL tokens including stopwords so we look up each token_no_stopword in this dict
    pos_lookup = {}
    for token, pos_tag, _ in pos_results:
        wordnet_pos = SPACY_TO_WORDNET.get(pos_tag, "n")  # default noun
        pos_lookup[token] = wordnet_pos
    
    # lemmatize each token using its POS from spaCy
    lemmatized = []
    for token in tokens_no_stopwords:
        wordnet_pos = pos_lookup.get(token, "n")  # default noun if not found
        lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized.append(lemma)
    
    return lemmatized