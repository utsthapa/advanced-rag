# Migration Guide - Modular Architecture

## Overview

The RAG system has been refactored from a monolithic `rag.py` file (2855 lines) into a clean, modular architecture with multiple focused files. This document explains the changes and how to use the new structure.

## What Changed

### Before (Monolithic)
```
advanced-rag/
├── rag.py              # 2855 lines - everything in one file
├── db.py               # 378 lines - database operations
├── performance_benchmarks.py  # 826 lines
├── pyproject.toml
└── schema.sql
```

### After (Modular)
```
advanced-rag/
├── src/
│   ├── models/          # Pydantic schemas (50 lines)
│   ├── core/            # Clients & DB (250 lines)
│   ├── retrieval/       # Search & reranking (350 lines)
│   ├── processing/      # Query processing & compression (250 lines)
│   ├── generation/      # Answer generation (150 lines)
│   ├── pipeline/        # RAG orchestration (200 lines)
│   ├── utils/           # Utilities (200 lines)
│   ├── benchmarks/      # Performance tests (moved)
│   └── config.py        # Configuration (50 lines)
├── main.py             # CLI entry point (200 lines)
├── rag_old.py          # Old file (backup)
├── db_old.py           # Old file (backup)
└── README.md           # Documentation
```

## Key Improvements

### 1. Single Responsibility Principle
Each module has one clear purpose:
- **models**: Data structures only
- **core**: Infrastructure (AWS, DB)
- **retrieval**: Search logic only
- **processing**: Query & document processing
- **generation**: Answer creation
- **pipeline**: Orchestration

### 2. Easy to Understand
- Each file is 50-350 lines (vs 2855 lines)
- Clear module boundaries
- Obvious where to find functionality

### 3. Easy to Test
```python
# Test search independently
from src.retrieval import search
results = search("test query", method="hybrid")

# Test compression independently
from src.processing import compress_documents
compressed = compress_documents(query, docs)
```

### 4. Easy to Extend
Want to add a new reranker?
- Just add a function to `src/retrieval/reranking.py`
- No need to search through 2855 lines

### 5. No Duplicate Code
All functionality preserved, but organized better.

## Import Changes

### Old Way
```python
from rag import (
    search,
    classify_query,
    rewrite_query,
    compress_documents,
    generate_answer,
    rag_pipeline,
)
```

### New Way
```python
from src.retrieval import search
from src.processing import classify_query, rewrite_query, compress_documents
from src.generation import generate_answer
from src.pipeline import rag_pipeline
```

## CLI Changes

### Old Way
```bash
# Run main script
python rag.py

# Chat mode
python rag.py -chat

# Full test
python rag.py --full
```

### New Way
```bash
# Quick test
python main.py

# Chat mode
python main.py --chat

# Benchmarks
python main.py --benchmark

# Load data
python main.py --load-data
```

## File Mapping

### Where did things go?

| Old Location (rag.py) | New Location |
|----------------------|--------------|
| Pydantic models | `src/models/schemas.py` |
| AWS clients | `src/core/clients.py` |
| Database operations | `src/core/database.py` |
| Search functions | `src/retrieval/search.py` |
| Reranking | `src/retrieval/reranking.py` |
| Query classification | `src/processing/query.py` |
| Query rewriting | `src/processing/query.py` |
| Compression | `src/processing/compression.py` |
| Answer generation | `src/generation/answer.py` |
| RAG pipeline | `src/pipeline/rag.py` |
| Dataset loading | `src/utils/data_loader.py` |
| Configuration | `src/config.py` |
| Benchmarks | `src/benchmarks/performance.py` |

## Code Examples

### Example 1: Simple Search
```python
# New modular way
from src.retrieval import search

results = search(
    query="How does CRISPR work?",
    method="hybrid",
    k=5,
    use_rewrite=True,
    reranker="cohere"
)
```

### Example 2: Complete RAG Pipeline
```python
# New modular way
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

### Example 3: Custom Workflow
```python
# Easy to mix and match components
from src.processing import classify_query, rewrite_query
from src.retrieval import hybrid_search
from src.retrieval.reranking import rerank_cohere
from src.processing import compress_documents
from src.generation import generate_answer

# Step by step
if classify_query(query):
    rewritten = rewrite_query(query)
    results = hybrid_search(query, k=10)
    results = rerank_cohere(query, results, k=5)
    results = compress_documents(query, results)
    answer = generate_answer(query, results)
```

## Benefits Summary

### Before
- ✗ 2855 lines in one file
- ✗ Hard to find specific functions
- ✗ Difficult to test components
- ✗ Hard to understand flow
- ✗ Risky to modify

### After
- ✓ 50-350 lines per file
- ✓ Clear module structure
- ✓ Easy to test each component
- ✓ Clear separation of concerns
- ✓ Safe to modify individual parts
- ✓ Professional architecture
- ✓ Easy to onboard new developers

## Backward Compatibility

The old files are preserved as backups:
- `rag_old.py` - Original RAG implementation
- `db_old.py` - Original database code

You can still reference them if needed, but all functionality is available in the new modular structure.

## Testing the Migration

```bash
# 1. Test imports
python -c "from src.retrieval import search; print('✓ Imports work')"

# 2. Test quick functionality
python main.py

# 3. Test chat mode
python main.py --chat

# 4. Run full benchmarks
python main.py --benchmark
```

## Questions?

- Check `README.md` for detailed documentation
- Each module has clear docstrings
- Old files preserved as `*_old.py` for reference

## Summary

The refactoring maintains 100% of functionality while making the code:
- **Cleaner**: Each file has one clear purpose
- **Simpler**: Small, focused files instead of huge monoliths
- **Maintainable**: Easy to find, test, and modify code
- **Professional**: Industry-standard architecture

No functionality was lost - everything was reorganized for better structure!

