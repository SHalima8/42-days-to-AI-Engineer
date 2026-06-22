from collections import Counter

def build_bow(sentences):
    bow_list = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        bow = dict(Counter(tokens))
        bow_list.append(bow)
    return bow_list