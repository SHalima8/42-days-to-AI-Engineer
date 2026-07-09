from embeddings.embedder_factory import get_embedder
from vector_stores.chroma_store import ChromaStore

embedder = get_embedder("minilm")
store = FaissStore(model_name="minilm")

texts = ["The refund policy allows returns within 30 days.", "Our office is located in Rawalpindi."]
embeddings = embedder.embed_docs(texts)
ids = ["chunk_1", "chunk_2"]
metadatas = [{"source": "test"}, {"source": "test"}]

store.add_chunks(ids, embeddings, texts, metadatas)
print("Count:", store.count())  # expect 2

query_vec = embedder.embed_query("what is the refund window?")
results = store.query(query_vec, top_k=2)
print(results)