# run this from inside day-17/, e.g. in a scratch.py file or python shell
from embeddings.embedder_factory import get_embedder

e = get_embedder("minilm")
vec = e.embed_query("what is the refund policy?")
print(len(vec))  # expect 384