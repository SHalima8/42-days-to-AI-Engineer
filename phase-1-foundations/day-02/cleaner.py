import re

def clean_text(text):
    # remove citation brackets like [1], [2], [citation needed]
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    # remove round bracket content like (M w), (DST)
    text = re.sub(r'\(.*?\)', ' ', text)
    # remove === domain labels
    text = re.sub(r'===.*?===', ' ', text)
    # remove bullet symbols
    text = re.sub(r'•', ' ', text)
    # remove numbers
    text = re.sub(r'\d+', ' ', text)
    # keep letters, spaces, and periods only
    text = re.sub(r'[^a-zA-Z\s\.]', ' ', text)
    # collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # lowercase
    text = text.lower()
    return text


def split_into_sentences(text):
    """Split cleaned text into individual sentences."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 5]
    return sentences