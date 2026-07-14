"""
list_collections.py
Run this from inside day-18-RAG-PIPELINE/ to confirm the Chroma path is
correct and see what collections exist + how many items each has.
"""

import chromadb

client = chromadb.PersistentClient(path="../day-17/outputs/chroma_db")

collections = client.list_collections()

if not collections:
    print("No collections found. Check that the path above actually points "
          "to where Day 17 built the Chroma DB.")
else:
    for c in collections:
        print("Collection name:", c.name)
        print("  Number of items:", c.count())