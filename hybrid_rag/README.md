# Hybrid RAG

Demonstrates a **Hybrid Retrieval-Augmented Generation** pipeline combining dense vector search (semantic) and sparse keyword search (lexical), enhanced with Multi-Query Retrieval and Reranking.

## Notebooks

| Notebook | Purpose |
|---|---|
| `knowledge_base_creation.ipynb` | Loads *Friends* episode subtitle files, chunks them, and stores embeddings in a ChromaDB collection with both sparse and dense vector indexes. |
| `hybrid_rag.ipynb` | Implements the full RAG pipeline: hybrid retrieval (RRF fusion), multi-query expansion via Azure OpenAI, reranking, and answer generation. |

## Data

`data/subtitles/` — SRT subtitle files for *Friends* Season 2, Episodes 1–10.

## Key Concepts

- **Hybrid Retrieval** — combines dense (semantic) and sparse (BM25 keyword) search with Reciprocal Rank Fusion (RRF), weighted 70/30.
- **Multi-Query Retrieval** — generates multiple query variants using an LLM to improve recall.
- **Reranking** — re-scores retrieved chunks for higher answer relevance.

## Prerequisites

```bash
pip install langchain langchain-community langchain-openai langchain-chroma snowballstemmer sentence-transformers
```

Place your Azure OpenAI API keys in the `keys/` directory:
- `.azure_openai_api_key.txt`
- `.azure_openai_embedding_key.txt`

## Usage

1. Run `knowledge_base_creation.ipynb` once to build the ChromaDB knowledge base.
2. Run `hybrid_rag.ipynb` to query the knowledge base and generate answers.
