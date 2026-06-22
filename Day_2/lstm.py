import torch
import torch.nn as nn
import numpy as np
import re


# ── HELPERS ────────────────────────────────────────────────────────────────────

def tokenize(sentence):
    """Lowercase and split into word tokens, stripping punctuation."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    return sentence.split()


def build_vocab(sentences):
    """Build a word → index mapping from all sentences."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    """Convert a sentence string into a list of token indices."""
    tokens = tokenize(sentence)
    return [vocab.get(w, vocab["<UNK>"]) for w in tokens]


def get_long_sentence(sentences, min_tokens=50):
    """
    Find the first sentence with at least min_tokens tokens.
    Falls back to the longest sentence if none meet the threshold.
    """
    tokenized = [(s, tokenize(s)) for s in sentences]
    long = [(s, t) for s, t in tokenized if len(t) >= min_tokens]
    if long:
        return long[0]
    # fallback: just use the longest one
    return max(tokenized, key=lambda x: len(x[1]))


# ── LSTM MODEL ─────────────────────────────────────────────────────────────────

class SimpleLSTM(nn.Module):
    """
    A minimal single-layer LSTM.

    embedding_dim : size of each word vector fed into the LSTM
    hidden_dim    : size of the hidden state (the 'sticky note')
    vocab_size    : total number of unique tokens
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super(SimpleLSTM, self).__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x shape: (1, sequence_length) — one sentence, multiple tokens.

        Returns:
            logits      : prediction scores for each timestep
            hidden_states: the hidden state vector at every timestep
        """
        embedded = self.embedding(x)                    # (1, seq_len, embed_dim)
        outputs, _ = self.lstm(embedded)                # (1, seq_len, hidden_dim)
        logits = self.output_layer(outputs)             # (1, seq_len, vocab_size)
        return logits, outputs


# ── GRADIENT ANALYSIS ──────────────────────────────────────────────────────────

def run_lstm_and_get_gradients(sentences):
    """
    1. Pick a 50+ token sentence from the corpus
    2. Build vocabulary and encode the sentence
    3. Run a forward pass through an untrained LSTM
    4. Compute a dummy loss and run backward pass
    5. Extract gradient magnitudes at each timestep

    Returns a dict with everything needed for printing and analysis.
    """
    # ── pick a long sentence ───────────────────────────────────────────
    sentence_str, tokens = get_long_sentence(sentences, min_tokens=50)
    seq_len = len(tokens)

    # ── build vocab and encode ────────────────────────────────────────
    vocab = build_vocab(sentences)
    vocab_size = len(vocab)
    encoded = encode_sentence(sentence_str, vocab)

    # truncate to first 60 tokens if very long, for cleaner output
    encoded = encoded[:60]
    tokens  = tokens[:60]
    seq_len = len(encoded)

    # convert to tensor: shape (1, seq_len)
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    # ── build model ───────────────────────────────────────────────────
    model = SimpleLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64)
    model.train()

    # ── forward pass ──────────────────────────────────────────────────
    # retain_grad() lets us read gradients on non-leaf tensors (hidden states)
    embedded = model.embedding(input_tensor)
    embedded.retain_grad()

    outputs, _ = model.lstm(embedded)          # (1, seq_len, hidden_dim)
    outputs.retain_grad()

    logits = model.output_layer(outputs)       # (1, seq_len, vocab_size)

    # ── dummy loss: predict next token at each step ───────────────────
    # shift: input[0..n-1] predicts target[1..n]
    if seq_len > 1:
        predictions = logits[0, :-1, :]        # (seq_len-1, vocab_size)
        targets     = input_tensor[0, 1:]      # (seq_len-1,)
        loss_fn     = nn.CrossEntropyLoss()
        loss        = loss_fn(predictions, targets)
    else:
        loss = logits.sum()                    # fallback

    # ── backward pass ─────────────────────────────────────────────────
    loss.backward()

    # ── extract gradient magnitudes per timestep ──────────────────────
    # outputs.grad shape: (1, seq_len, hidden_dim)
    # we take the L2 norm across hidden_dim to get one number per timestep
    grad_magnitudes = []
    if outputs.grad is not None:
        grads = outputs.grad[0]                # (seq_len, hidden_dim)
        for t in range(seq_len):
            mag = grads[t].norm().item()       # scalar magnitude at timestep t
            grad_magnitudes.append(mag)
    else:
        grad_magnitudes = [0.0] * seq_len

    # ── hidden state magnitudes (shows what the LSTM is "remembering") ─
    with torch.no_grad():
        embedded_eval = model.embedding(input_tensor)
        hidden_states, _ = model.lstm(embedded_eval)
        hidden_mags = [hidden_states[0, t].norm().item() for t in range(seq_len)]

    return {
        "sentence"       : sentence_str,
        "tokens"         : tokens,
        "seq_len"        : seq_len,
        "vocab_size"     : vocab_size,
        "loss"           : loss.item(),
        "grad_magnitudes": grad_magnitudes,
        "hidden_mags"    : hidden_mags,
    }


# ── PRINT HELPERS ──────────────────────────────────────────────────────────────

def print_lstm_results(results):
    """Master print function — called from main.py."""
    import matplotlib.pyplot as plt

    print(f"\n── Sentence Used ({results['seq_len']} tokens) ──")
    print(f"  {results['sentence'][:120]}...")

    print(f"\n── Model Info ──")
    print(f"  Vocabulary size : {results['vocab_size']}")
    print(f"  Sequence length : {results['seq_len']} tokens")
    print(f"  Loss (untrained): {results['loss']:.4f}  "
          f"(high is expected — model is randomly initialized)")

    grads   = results["grad_magnitudes"]
    seq_len = results["seq_len"]
    tokens  = results["tokens"]

    # ── Print all 60 gradient values ──────────────────────────────────
    print(f"\n── Gradient Magnitudes — All {seq_len} Tokens ──")
    print(f"  (How much each token influences the model's learning)")
    print(f"  Higher = more influence | Lower = being forgotten\n")

    max_grad = max(grads) if max(grads) > 0 else 1.0
    for t in range(seq_len):
        bar_len = int(grads[t] / max_grad * 30)
        bar = "█" * bar_len
        print(f"  Token {t+1:3d} ('{tokens[t]:15}') | grad: {grads[t]:.6f} | {bar}")

    # ── Key observation ───────────────────────────────────────────────
    print(f"\n── Key Observation ──")
    first_grad = grads[0]
    last_grad  = grads[-1]
    non_zero   = [g for g in grads if g > 0]
    avg_grad   = sum(non_zero) / len(non_zero) if non_zero else 0

    print(f"  First token gradient : {first_grad:.6f}")
    print(f"  Last  token gradient : {last_grad:.6f}")
    print(f"  Average (non-zero)   : {avg_grad:.6f}")
    print(f"\n  NOTE: This model is randomly initialized and untrained.")
    print(f"  Pronounced gradient decay typically appears after training,")
    print(f"  when the model has learned to compress information over time.")
    print(f"  What this experiment still demonstrates:")
    print(f"    → Gradients DO flow backward through all timesteps")
    print(f"    → Token {seq_len} (last) gets zero gradient — nothing to predict after it")
    print(f"    → The backward pass is strictly sequential, mirroring the forward pass")

    if last_grad == 0.0 and first_grad > 0:
        print(f"    → Clear asymmetry: token 1 has gradient {first_grad:.6f}, "
              f"token {seq_len} has 0.0")

    # ── Matplotlib line chart ─────────────────────────────────────────
    print(f"\n── Plotting Gradient Magnitudes (saved as lstm_gradients.png) ──")

    timesteps = list(range(1, seq_len + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps, grads, color="steelblue", linewidth=1.5,
            marker="o", markersize=3, label="Gradient magnitude")

    # highlight first and last token
    ax.scatter([1], [grads[0]], color="green", zorder=5, s=60,
               label=f"Token 1 ('{tokens[0]}'): {grads[0]:.6f}")
    ax.scatter([seq_len], [grads[-1]], color="red", zorder=5, s=60,
               label=f"Token {seq_len} ('{tokens[-1]}'): {grads[-1]:.6f}")

    ax.set_xlabel("Timestep (Token Position)", fontsize=12)
    ax.set_ylabel("Gradient Magnitude", fontsize=12)
    ax.set_title(
        "LSTM Gradient Magnitudes Across Timesteps\n"
        "(Untrained model — decay more pronounced after training)",
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lstm_gradients.png", dpi=150)
    plt.show()
    print(f"  Chart saved as lstm_gradients.png")

    # ── No parallelism ────────────────────────────────────────────────
    print(f"\n── Why No Parallelism Is Possible ──")
    print(f"  Token 1  must be processed before Token 2")
    print(f"  Token 2  must be processed before Token 3")
    print(f"  ...and so on for all {seq_len} tokens.")
    print(f"  Each step depends on the hidden state from the previous step.")
    print(f"  This is a hard sequential dependency — cannot be parallelized.")
    print(f"  Transformers remove this by processing all tokens simultaneously.")