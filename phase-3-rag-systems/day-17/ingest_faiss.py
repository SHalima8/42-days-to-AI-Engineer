(venv) D:\Planet Beyond\phase-3-rag-systems\day-17>python ingest_chroma.py
Loaded 45 chunks from ../day-16/outputs/recursive_docx_text.json

--- Ingesting into ChromaDB using model: minilm ---
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3990.07it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.26it/s]
Done. Collection now has 45 chunks.

--- Ingesting into ChromaDB using model: mpnet ---
Loading weights: 100%|███████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3044.81it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████| 2/2 [00:11<00:00,  5.65s/it]
Done. Collection now has 45 chunks.

--- Ingesting into ChromaDB using model: bge ---
Loading weights: 100%|███████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 2624.90it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████| 2/2 [00:32<00:00, 16.02s/it]
Done. Collection now has 45 chunks.