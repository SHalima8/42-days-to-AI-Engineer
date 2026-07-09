# day-17/embeddings/embedder_factory.py

from sentence_transformers import SentenceTransformer

# BGE models expect a special instruction prefix on QUERIES only (not on documents).
# Skipping this silently makes BGE look worse than it actually is.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class LocalEmbedder:
    """Wraps a sentence-transformers model (MiniLM, mpnet, BGE)."""
    def __init__(self, model_name, hf_path, needs_query_prefix=False):
        self.model_name = model_name
        self.model = SentenceTransformer(hf_path)
        self.needs_query_prefix = needs_query_prefix

    def embed_docs(self, texts):
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)

    def embed_query(self, text):
        if self.needs_query_prefix:
            text = BGE_QUERY_PREFIX + text
        return self.model.encode([text])[0]


def get_embedder(model_name):
    """
    Single entry point used by every other script.
    model_name options: 'minilm', 'mpnet', 'bge'
    """
    registry = {
        "minilm": lambda: LocalEmbedder("minilm", "all-MiniLM-L6-v2"),
        "mpnet": lambda: LocalEmbedder("mpnet", "all-mpnet-base-v2"),
        "bge": lambda: LocalEmbedder("bge", "BAAI/bge-large-en-v1.5", needs_query_prefix=True),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(registry.keys())}")
    return registry[model_name]()