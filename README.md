# Hybrid Search Demo with Pinecone

This repository is a demonstration of hybrid search using dense embeddings and sparse (BM25-style) text features stored in Pinecone. The notebooks show how to create a Pinecone index, build dense embeddings with a Hugging Face sentence-transformer model, compute sparse BM25-style encodings using the Pinecone Text BM25 encoder, and run hybrid retrieval with LangChain community utilities.

Contents
-------
- `hybrid_search.ipynb` — A clean, runnable demo notebook that creates an index (if needed), trains/loads a BM25 encoder, adds example documents, and runs a hybrid query.
- `bm25_values.json` — Saved BM25 encoder values used by the demo (example/seed values).
- `requirements.txt` — Python dependencies used for the demo.

Quick overview
--------------
Hybrid search combines two retrieval signals:

- Dense (vector) similarity: semantic embeddings from a transformer model (e.g., sentence-transformers). Captures semantic similarity beyond exact word overlap.
- Sparse (BM25-like) relevance: token-level term-frequency/inverse-document-frequency style features that preserve exact matching and term importance.

Combining both signals (often via a weighted merge) improves retrieval quality: dense vectors find semantic matches, while BM25 protects against missing keyword matches and favors documents containing query terms.

Why use Pinecone and LangChain
------------------------------
- Pinecone provides a managed vector database with support for hybrid search (dense vectors + sparse features) and fast approximate nearest neighbor search.
- LangChain community provides convenient retriever wrappers (e.g., `PineconeHybridSearchRetriever`) that wire embeddings, sparse encoders, and Pinecone index operations together.

Setup instructions (Windows / PowerShell)
---------------------------------------
1. Clone or open this repository in VS Code.
2. Create and activate a Python virtual environment (recommended):

   # PowerShell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1

3. Install dependencies (preferably inside the venv):

   pip install -r requirements.txt

4. Create a `.env` file in the repository root (it is already listed in `.gitignore` to avoid accidental commits). Add the following environment variables:

   HF_TOKEN=<your-huggingface-token>
   PINECONE_API_KEY=<your-pinecone-api-key>

   Notes:
   - The repository contains an example `.env` content used during development (do not commit or share real keys).
   - If you prefer, you can set `PINECONE_API_KEY` directly in the environment instead of the `.env` file.

5. Open `hybrid_search.ipynb` in Jupyter or VS Code and run the cells in order. The notebook will:
   - Initialize the Pinecone client and create an index named `pinecone-hybrid-search` if it does not exist.
   - Load a Hugging Face sentence-transformer (`all-MiniLM-L6-v2`) for dense embeddings.
   - Create and fit a BM25 encoder, saving values to `bm25_values.json`.
   - Construct a `PineconeHybridSearchRetriever`, add sample texts, and run a sample query.

How the notebooks are organized
-----------------------------
- `hybrid_search.ipynb` is the recommended starting point — it's structured, annotated, and includes short waits where appropriate (e.g., after index creation).


Key code snippets
-----------------
The core pieces used in the notebooks are:

- Initialize Pinecone:

  from pinecone import Pinecone, ServerlessSpec
  pc = Pinecone(api_key=api_key)

- Create index (example):

  if index_name not in [idx.name for idx in pc.list_indexes()]:
      pc.create_index(
          name=index_name,
          dimension=384,
          metric='dotproduct',
          spec=ServerlessSpec(cloud='aws', region='us-east-1')
      )

- Embeddings (sentence-transformers via LangChain wrapper):

  from langchain_huggingface import HuggingFaceEmbeddings
  embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

- BM25 sparse encoder (Pinecone Text):

  from pinecone_text.sparse import BM25Encoder
  bm25_encoder = BM25Encoder().default()
  bm25_encoder.fit(sentences)
  bm25_encoder.dump('bm25_values.json')

- Hybrid retriever (LangChain community):

  from langchain_community.retrievers import PineconeHybridSearchRetriever
  retriever = PineconeHybridSearchRetriever(
      embeddings=embeddings,
      sparse_encoder=bm25_encoder,
      index=index,
  )

Security notes
--------------
- Never commit API keys or tokens. This repository's `.gitignore` contains `.env` for that reason.
- Rotate credentials if you accidentally commit keys to public repositories.

Troubleshooting
---------------
- If index creation fails: check your Pinecone API key, project or usage limits in the Pinecone dashboard, and the chosen cloud/region.
- If embedding downloads fail: ensure `HF_TOKEN` is valid and your environment can access Hugging Face models.
- If the BM25 encoder throws errors: verify the `pinecone-text` / `pinecone_text` package is installed and compatible. The notebook uses `pinecone_text.sparse.BM25Encoder`.

Notes on reproducibility and costs
--------------------------------
- Running embeddings and creating Pinecone indexes can incur network usage and costs depending on your Pinecone plan and cloud provider. Use small example corpora during experimentation.
- The notebooks use `all-MiniLM-L6-v2` (384-dim) to keep the index dimension small and cost-effective.

Further improvements / next steps
-------------------------------
- Add an automated script (Python module) to load documents, compute embeddings+BM25 encodings, and batch-upsert to Pinecone.
- Add unit tests for the retrieval logic and a small local fallback (FAISS or Chroma) for offline testing without Pinecone.
- Add more advanced reranking (e.g., a cross-encoder) to refine hybrid results.

License and attribution
-----------------------
This demo uses open-source models and libraries. Check each project's license (Hugging Face model license, Pinecone terms). The code in this repository is provided as-is for demonstration and education.

Contact / Questions
-------------------
If you want help adapting the demo to your corpus or adding a batch-upsert pipeline, open an issue or ask for changes.
