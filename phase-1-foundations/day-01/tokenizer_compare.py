from transformers import GPT2Tokenizer, BertTokenizer

def compare_tokenizers(sentence):
    """Compare 4 tokenization methods on the same sentence."""
    results = {}

    # 1. character level
    results["character"] = list(sentence)

    # 2. word level
    results["word"] = sentence.split()

    # 3. BPE via GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    results["bpe_gpt2"] = gpt2_tokenizer.tokenize(sentence)

    # 4. WordPiece via BERT
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    results["wordpiece_bert"] = bert_tokenizer.tokenize(sentence)

    # print comparison
    for method, tokens in results.items():
        print(f"\n{method.upper()} ({len(tokens)} tokens):")
        print(tokens)

    return results