# Quick Reference Guide

## 🚀 Getting Started (30 seconds)

```bash
# 1. Install dependencies
uv sync

# 2. Setup database
createdb rag_db
psql rag_db < schema.sql

# 3. Load data and run
python rag.py --load-data
python rag.py --chat
```

## 📂 File Structure

```
rag.py              ← Run this!
README.md           ← Start here for docs
src/                ← All code here
docs/               ← Detailed docs here
```

## 💻 Common Commands

```bash
# Interactive chat
python rag.py --chat

# Quick test
python rag.py

# Benchmarks
python rag.py --benchmark

# Load data
python rag.py --load-data
```

## 🐍 Python API

### Simple Search
```python
from src.retrieval import search

results = search(
    query="How does CRISPR work?",
    method="hybrid",  # or "vector" or "fulltext"
    k=5,
    use_rewrite=True,
    reranker="cohere"  # or "bge", "claude", None
)
```

### Full RAG Pipeline
```python
from src.pipeline import rag_pipeline, print_rag_answer

result = rag_pipeline(
    query="What causes COVID-19?",
    k=3,
    use_rewrite=True,
    reranker="cohere",
    use_compression=True,
    verbose=True
)

print_rag_answer(result)
```

### Individual Components

```python
# Classification
from src.processing import classify_query
needs_retrieval = classify_query("What is 2+2?")

# Query rewriting
from src.processing import rewrite_query
rewritten = rewrite_query("CRISPR gene editing")

# Compression
from src.processing import compress_documents
compressed = compress_documents(query, documents)

# Answer generation
from src.generation import generate_answer
answer = generate_answer(query, documents)
```

## 🔧 Configuration

Edit `src/config.py`:

```python
# AWS
AWS_PROFILE = "your-profile"
AWS_REGION = "us-east-1"

# RAG
RAG_TOP_K = 5
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.9

# Database
DB_CONFIG = {
    "host": "localhost",
    "database": "rag_db",
    "user": "your_user",
    "password": "your_password",
}
```

## 📊 Module Overview

| Import From | What You Get |
|-------------|--------------|
| `src.retrieval` | `search`, `vector_search`, `hybrid_search` |
| `src.processing` | `classify_query`, `rewrite_query`, `compress_documents` |
| `src.generation` | `generate_answer` |
| `src.pipeline` | `rag_pipeline`, `print_rag_answer` |
| `src.core` | `DatabaseManager`, `get_embeddings` |

## 🎯 Common Tasks

### Add a New Reranker
1. Add function to `src/retrieval/reranking.py`
2. Export it in `src/retrieval/__init__.py`
3. Use it: `search(..., reranker="my_new_method")`

### Change Database Connection
Edit `src/config.py` → `DB_CONFIG`

### Adjust AWS Models
Edit `src/core/clients.py` → model IDs

### Modify Search Behavior
Edit `src/retrieval/search.py`

## 🐛 Troubleshooting

**Import errors?**
```bash
# Make sure you're in project root
cd /path/to/advanced-rag
python rag.py
```

**Database errors?**
```bash
# Check database is running
psql -l

# Recreate database
dropdb rag_db
createdb rag_db
psql rag_db < schema.sql
python rag.py --load-data
```

**AWS errors?**
Check `src/config.py` → `AWS_PROFILE` and credentials

## 📚 More Info

- **Architecture**: `docs/ARCHITECTURE.md`
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`
- **Full README**: `README.md`
- **Overview**: `OVERVIEW.md`

## ✨ Key Features

✅ Query classification (save 40% cost)
✅ Query rewriting (optimize for search type)
✅ Hybrid search (vector + fulltext)
✅ 3 reranking strategies (Cohere, BGE, Claude)
✅ Contextual compression (30-50% token reduction)
✅ Answer generation with citations
✅ Performance benchmarks
✅ Interactive chat mode

---

**Need help?** Check the docs in `docs/` folder!

