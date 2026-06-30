import torch
import numpy as np
from transformers import BertTokenizer, BertModel


# ── SETUP ──────────────────────────────────────────────────────────────────────

def load_bert():
    """
    Load pretrained BERT tokenizer and model from HuggingFace.
    bert-base-uncased: 12 layers, 768-dimensional output vectors.
    First run downloads ~440MB — cached locally after that.
    """
    print("  Loading BERT (bert-base-uncased) from HuggingFace...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased")
    model.eval()   # inference mode — no dropout, no gradient tracking
    print("  BERT loaded successfully.")
    return tokenizer, model


# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

def get_word_embedding(sentence, target_word, tokenizer, model):
    """
    Extract the contextual embedding for target_word inside sentence.

    Steps:
    1. Tokenize the sentence with BERT's own tokenizer
    2. Find which token index corresponds to target_word
    3. Run the full sentence through BERT
    4. Return the last hidden state vector at that token's position

    Returns:
        vector     : numpy array of shape (768,)
        token_index: position of the target word in the token list
        all_tokens : full list of tokens BERT sees (includes [CLS], [SEP])
    """
    # BERT tokenization — adds [CLS] at start and [SEP] at end
    inputs     = tokenizer(sentence, return_tensors="pt")
    token_ids  = inputs["input_ids"][0]
    all_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # find the position of target_word in the token list
    target_lower = target_word.lower()
    token_index  = None
    for i, tok in enumerate(all_tokens):
        if tok == target_lower:
            token_index = i
            break

    # fallback: try partial match (handles subword tokenization)
    if token_index is None:
        for i, tok in enumerate(all_tokens):
            if target_lower in tok:
                token_index = i
                break

    if token_index is None:
        raise ValueError(
            f"Word '{target_word}' not found in BERT tokens.\n"
            f"Tokens: {all_tokens}\n"
            f"Try checking the sentence contains the exact word."
        )

    # forward pass — no gradient needed, we just want the vectors
    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state shape: (1, seq_len, 768)
    # we pull out the vector at our target token's position
    last_hidden = outputs.last_hidden_state[0]        # (seq_len, 768)
    vector      = last_hidden[token_index].numpy()    # (768,)

    return vector, token_index, all_tokens


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two numpy vectors. Range: -1 to 1."""
    dot     = np.dot(vec1, vec2)
    norm1   = np.linalg.norm(vec1)
    norm2   = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ── MAIN ANALYSIS ──────────────────────────────────────────────────────────────

def run_bert_polysemy(tokenizer, model):
    """
    Demonstrate that BERT gives the same word different vectors
    depending on sentence context — directly resolving Day 2's
    polysemy failure with Word2Vec.

    Uses 'network' which appeared in Day 2 as the polysemy example:
      Sense A: neural/computational network
      Sense B: social/infrastructure network
    """

    target_word = "network"

    # Sense A — from your actual corpus (neural network domain)
    sentence_a = "A neural network consists of connected units called artificial neurons which model the brain."

    # Sense B — infrastructure/social meaning (not in corpus — crafted for contrast)
    sentence_b = "The earthquake damaged the communication network across the affected region."

    print(f"\n  Target word     : '{target_word}'")
    print(f"\n  Sentence A (neural/computational sense):")
    print(f"    {sentence_a}")
    print(f"\n  Sentence B (infrastructure/communication sense):")
    print(f"    {sentence_b}")

    # get BERT vectors
    vec_a, idx_a, tokens_a = get_word_embedding(sentence_a, target_word, tokenizer, model)
    vec_b, idx_b, tokens_b = get_word_embedding(sentence_b, target_word, tokenizer, model)

    # cosine similarity between the two vectors
    similarity = cosine_similarity(vec_a, vec_b)

    return {
        "target_word" : target_word,
        "sentence_a"  : sentence_a,
        "sentence_b"  : sentence_b,
        "tokens_a"    : tokens_a,
        "tokens_b"    : tokens_b,
        "token_idx_a" : idx_a,
        "token_idx_b" : idx_b,
        "vec_a"       : vec_a,
        "vec_b"       : vec_b,
        "similarity"  : similarity,
    }


def run_bert_extra_contexts(tokenizer, model, target_word="network"):
    """
    Get BERT vectors for 6 sentences using 'network' in varied contexts.
    These vectors feed into the t-SNE plot in Task 5.
    """
    sentences = [
        # neural network sense (3 sentences)
        "A neural network learns patterns from training data.",
        "The deep network achieved state of the art performance on image classification.",
        "Recurrent neural networks process sequential data like speech.",
        # infrastructure / social sense (3 sentences)
        "The phone network was overloaded after the disaster.",
        "She built a strong professional network over her career.",
        "The road network in the city needs urgent repair.",
    ]

    labels  = [
        "neural-1", "neural-2", "neural-3",
        "infra-1",  "infra-2",  "infra-3",
    ]

    vectors = []
    for sent in sentences:
        try:
            vec, _, _ = get_word_embedding(sent, target_word, tokenizer, model)
            vectors.append(vec)
        except ValueError as e:
            print(f"  Warning: {e}")
            vectors.append(np.zeros(768))

    return sentences, labels, vectors


# ── PRINT HELPERS ──────────────────────────────────────────────────────────────

def print_bert_results(results):
    """Master print function — called from main.py."""

    print(f"\n── BERT Tokenization ──")
    print(f"  Sentence A tokens : {results['tokens_a']}")
    print(f"  '{results['target_word']}' found at position : {results['token_idx_a']}")
    print(f"\n  Sentence B tokens : {results['tokens_b']}")
    print(f"  '{results['target_word']}' found at position : {results['token_idx_b']}")

    print(f"\n── Contextual Vectors (first 8 of 768 dimensions) ──")
    print(f"  Sentence A vector : {results['vec_a'][:8].round(4)}")
    print(f"  Sentence B vector : {results['vec_b'][:8].round(4)}")

    print(f"\n── Cosine Similarity Between the Two Vectors ──")
    sim = results['similarity']
    print(f"  Similarity score : {sim:.4f}")

    if sim >= 0.99:
        verdict = "Vectors are nearly identical — no contextual difference detected."
    elif sim >= 0.90:
        verdict = "Vectors are similar but not identical — mild contextual difference."
    elif sim >= 0.70:
        verdict = "Clear contextual difference — BERT is distinguishing the two senses."
    else:
        verdict = "Strong contextual difference — BERT clearly separates the two meanings."

    print(f"  Interpretation   : {verdict}")

    print(f"\n── What This Resolves ──")
    print(f"  Day 2 (Word2Vec) : 'network' had ONE fixed vector regardless of context.")
    print(f"  Day 3 (BERT)     : 'network' now has TWO different vectors — one per meaning.")
    print(f"  Similarity of {sim:.4f} proves the vectors are not identical.")
    print(f"  The lower this number, the more BERT has separated the two senses.")
    print(f"\n  This is the core advantage of contextual embeddings over static ones.")