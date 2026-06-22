import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from bert_embeddings import get_word_embedding


# ── WORD2VEC VECTORS FOR MULTIPLE CONTEXTS ────────────────────────────────────

def get_w2v_vectors(w2v_model, target_word, n_copies):
    """
    Word2Vec gives the SAME vector every time regardless of sentence.
    We return n_copies of that one vector to match the number of sentences.
    If the word isn't in vocabulary, return zero vectors.
    """
    try:
        vec = w2v_model.wv[target_word]          # shape: (vector_size,)
        return [vec.copy() for _ in range(n_copies)]
    except KeyError:
        print(f"  Warning: '{target_word}' not in Word2Vec vocabulary.")
        dim = w2v_model.vector_size
        return [np.zeros(dim) for _ in range(n_copies)]


# ── BERT VECTORS FOR MULTIPLE CONTEXTS ───────────────────────────────────────

def get_bert_vectors(sentences, target_word, tokenizer, bert_model):
    """
    BERT gives a DIFFERENT vector for target_word in each sentence.
    Returns one 768-dim vector per sentence.
    """
    vectors = []
    for sent in sentences:
        try:
            vec, _, _ = get_word_embedding(sent, target_word, tokenizer, bert_model)
            vectors.append(vec)
        except ValueError as e:
            print(f"  Warning: {e}")
            vectors.append(np.zeros(768))
    return vectors


# ── t-SNE REDUCTION ───────────────────────────────────────────────────────────

def reduce_tsne(vectors, n_components=2, perplexity=3, random_state=42):
    """
    Reduce high-dimensional vectors to 2D using t-SNE.

    perplexity: loosely means 'how many neighbors to consider'.
                Must be less than the number of points.
                We use 3 since we have 6 points — standard rule is perplexity < n_samples.

    Returns numpy array of shape (n_samples, 2).
    """
    arr = np.array(vectors)

    # if all vectors are identical (Word2Vec case), add tiny noise
    # so t-SNE doesn't collapse — this is expected and honest behavior
    if np.allclose(arr, arr[0]):
        arr = arr + np.random.RandomState(random_state).normal(
            0, 1e-6, arr.shape
        )

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000)
    return tsne.fit_transform(arr)


# ── MAIN PLOT FUNCTION ────────────────────────────────────────────────────────

def run_tsne_comparison(w2v_model, tokenizer, bert_model, target_word="network"):
    """
    Build and save the side-by-side t-SNE comparison plot:
    Left  : Word2Vec — same word, all points clustered together
    Right : BERT     — same word, points separated by meaning
    """

    # ── sentences: 3 neural sense, 3 infrastructure sense ─────────────
    sentences = [
        "A neural network learns patterns from training data.",
        "The deep network achieved state of the art performance on image classification.",
        "Recurrent neural networks process sequential data like speech.",
        "The phone network was overloaded after the disaster.",
        "She built a strong professional network over her career.",
        "The road network in the city needs urgent repair.",
    ]

    labels = [
        "neural-1", "neural-2", "neural-3",
        "infra-1",  "infra-2",  "infra-3",
    ]

    # sense group for coloring: 0 = neural, 1 = infrastructure
    groups = [0, 0, 0, 1, 1, 1]
    colors = ["steelblue", "tomato"]
    group_names = ["Neural/Computational", "Infrastructure/Social"]

    print(f"\n  Target word : '{target_word}'")
    print(f"  Sentences   : {len(sentences)} "
          f"(3 neural sense + 3 infrastructure sense)")

    # ── get vectors ────────────────────────────────────────────────────
    print("\n  Getting Word2Vec vectors...")
    w2v_vecs  = get_w2v_vectors(w2v_model, target_word, len(sentences))

    print("  Getting BERT vectors...")
    bert_vecs = get_bert_vectors(sentences, target_word, tokenizer, bert_model)

    # ── t-SNE reduction ────────────────────────────────────────────────
    print("\n  Running t-SNE on Word2Vec vectors...")
    w2v_2d  = reduce_tsne(w2v_vecs,  perplexity=3)

    print("  Running t-SNE on BERT vectors...")
    bert_2d = reduce_tsne(bert_vecs, perplexity=3)

    # ── plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"t-SNE: Static (Word2Vec) vs Contextual (BERT)\n"
        f"Same word '{target_word}' across 6 different-context sentences",
        fontsize=14, fontweight="bold"
    )

    plot_data = [
        (axes[0], w2v_2d,  "Word2Vec (Static)\nAll points = same vector"),
        (axes[1], bert_2d, "BERT (Contextual)\nPoints shaped by sentence meaning"),
    ]

    for ax, coords, title in plot_data:
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y,
                       color=colors[groups[i]],
                       s=120, zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(labels[i],
                        (x, y),
                        textcoords="offset points",
                        xytext=(8, 4),
                        fontsize=9)

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("t-SNE dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE dimension 2", fontsize=10)
        ax.grid(True, alpha=0.3)

        # legend
        patches = [mpatches.Patch(color=colors[i], label=group_names[i])
                   for i in range(2)]
        ax.legend(handles=patches, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Plot saved as tsne_comparison.png")

    # ── print cosine similarities between sense groups ─────────────────
    print("\n── BERT Cosine Similarities Between Sense Groups ──")
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    neural_vecs = [bert_vecs[i] for i in range(3)]
    infra_vecs  = [bert_vecs[i] for i in range(3, 6)]

    # within neural group
    within_neural = []
    for i in range(len(neural_vecs)):
        for j in range(i + 1, len(neural_vecs)):
            s = float(sk_cos([neural_vecs[i]], [neural_vecs[j]])[0][0])
            within_neural.append(s)

    # within infra group
    within_infra = []
    for i in range(len(infra_vecs)):
        for j in range(i + 1, len(infra_vecs)):
            s = float(sk_cos([infra_vecs[i]], [infra_vecs[j]])[0][0])
            within_infra.append(s)

    # across groups
    across = []
    for v1 in neural_vecs:
        for v2 in infra_vecs:
            s = float(sk_cos([v1], [v2])[0][0])
            across.append(s)

    print(f"  Within neural sense  (should be HIGH) : "
          f"{np.mean(within_neural):.4f} avg")
    print(f"  Within infra sense   (should be HIGH) : "
          f"{np.mean(within_infra):.4f} avg")
    print(f"  Across sense groups  (should be LOW)  : "
          f"{np.mean(across):.4f} avg")
    print(f"\n  → If within > across, BERT is successfully separating meanings.")
    print(f"  → Word2Vec would show identical scores for all three — no separation.")