from embeddings.embedder_factory import get_embedder

e_mpnet = get_embedder("mpnet")
vec_mpnet = e_mpnet.embed_query("what is the refund policy?")
print("mpnet:", len(vec_mpnet))   # expect 768

e_bge = get_embedder("bge")
vec_bge = e_bge.embed_query("what is the refund policy?")
print("bge:", len(vec_bge))       # expect 1024