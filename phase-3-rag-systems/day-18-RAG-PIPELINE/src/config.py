"""
config.py
Points at your EXISTING Day 16/17 assets — nothing here gets rebuilt,
Day 18 only reads from these paths.
"""

# --- Day 16 outputs (table registry) ---
TABLES_LOOKUP_PATH = "../day-16/data/extracted/tables_lookup.json"
TABLES_SUMMARY_PATH = "../day-16/data/extracted/tables_summary.md"

# --- Day 17 outputs (vector store) ---
CHROMA_DB_PATH = "../day-17/outputs/chroma_db"
COLLECTION_NAME = "chunks_minilm"   # ChromaStore names collections as f"chunks_{model_name}"

# --- Embedding (must match what Day 17 indexed with) ---
EMBEDDING_MODEL_NAME = "minilm"

# --- Retrieval ---
TOP_K = 4

# --- Generation (Groq) ---
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.3