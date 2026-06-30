import spacy

nlp = spacy.load("en_core_web_sm")

POS_DESCRIPTIONS = {
    "NOUN": "Noun",
    "VERB": "Verb",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PRON": "Pronoun",
    "DET": "Determiner",
    "ADP": "Preposition",
    "CONJ": "Conjunction",
    "PUNCT": "Punctuation",
    "PROPN": "Proper Noun",
    "NUM": "Number",
    "AUX": "Auxiliary Verb",
    "CCONJ": "Coordinating Conjunction",
    "SCONJ": "Subordinating Conjunction",
}

def pos_tag_text(text):
    """Return list of (token, POS tag, description) tuples for entire text."""
    
    # nlp.pipe is more memory efficient for large text than nlp(text) directly
    # max_length increase handles large corpus without memory error
    nlp.max_length = 2000000
    
    doc = nlp(text)
    results = []
    for token in doc:
        # skip punctuation and whitespace tokens for cleaner output
        if token.pos_ in ("PUNCT", "SPACE"):
            continue
        description = POS_DESCRIPTIONS.get(token.pos_, token.pos_)
        results.append((token.text, token.pos_, description))
    return results