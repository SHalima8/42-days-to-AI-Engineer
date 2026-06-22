from transformers import GPT2Tokenizer, BertTokenizer

def compare_tokenizers(sentence):
    results = {}

    # 1. character level
    results["character"] = list(sentence)

    # 2. word level
    results["word"] = sentence.split()

    # 3. BPE via GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    results["bpe_gpt2"] = gpt2_tokenizer.tokenize(sentence)

    # 4. WordPiece via BERT — truncate if over 512
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = bert_tokenizer.tokenize(sentence)
    if len(tokens) > 512:
        tokens = tokens[:512]
    results["wordpiece_bert"] = tokens

    for method, tokens in results.items():
        print(f"\n{method.upper()} ({len(tokens)} tokens):")
        print(tokens[:30], "..." if len(tokens) > 30 else "")

    return results