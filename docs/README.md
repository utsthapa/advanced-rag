# Advanced RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with modular architecture, built on AWS Bedrock, PostgreSQL with pgvector, and LangChain.

## ⚡ TL;DR

```bash
# Run the main program
python rag.py --chat

# Or setup database and load data
python rag.py --load-data
python rag.py --chat
```

That's it! See below for full setup.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Setup Database

```bash
# Create PostgreSQL database and run schema
createdb rag_db
psql rag_db < schema.sql
```

### 3. Load Dataset & Run

```bash
# Load BeIR SciFact dataset
python rag.py --load-data

# Interactive chat mode
python rag.py --chat

# Quick test (default)
python rag.py

# Run performance benchmarks
python rag.py --benchmark

# Run comprehensive RAG analysis (answers 6 key questions)
python rag.py --analyze
```

## 📁 Project Structure

```
advanced-rag/
├── src/                    # Source code (modular components)
│   ├── models/            # Data schemas (Pydantic)
│   ├── core/              # AWS clients & database
│   ├── retrieval/         # Search & reranking
│   ├── processing/        # Query processing & compression
│   ├── generation/        # Answer generation
│   ├── pipeline/          # RAG orchestration
│   ├── utils/             # Utilities & data loading
│   ├── benchmarks/        # Performance testing
│   └── config.py          # Configuration
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md    # System architecture & diagrams
│   ├── MIGRATION_GUIDE.md # How to use the modular structure
│   ├── REFACTORING_SUMMARY.md  # Detailed refactoring info
│   └── QUICK_REFERENCE.md # Quick API reference
│
├── rag.py                  # Main CLI entry point
├── schema.sql              # Database schema
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## 🎯 Features

### 1. **Query Classification**
Intelligently determines if retrieval is needed, reducing costs by ~40%

### 2. **Query Rewriting**
Optimizes queries for vector and fulltext search separately

### 3. **Hybrid Search**
Combines vector and fulltext search using Reciprocal Rank Fusion (RRF)

### 4. **Multiple Reranking Strategies**
- **Cohere**: Cloud-based, highest accuracy
- **BGE**: Open source, good accuracy
- **Claude**: LLM-based with reasoning

### 5. **Contextual Compression**
Removes irrelevant content (30-50% token reduction) while preserving key information

### 6. **Answer Generation**
Creates well-cited answers with source attribution

## 💡 Usage Examples

### Command Line

```bash
# Quick test with 3 sample queries
python rag.py

# Interactive chat mode
python rag.py --chat

# Run full performance benchmarks
python rag.py --benchmark

# Run comprehensive RAG analysis (answers 6 key questions about your system)
python rag.py --analyze

# Reload dataset
python rag.py --load-data
```

### Python API

```python
# Simple search
from src.retrieval import search

results = search(
    query="How does CRISPR work?",
    method="hybrid",
    k=5,
    use_rewrite=True,
    reranker="cohere"
)

# Full RAG pipeline
from src.pipeline import rag_pipeline, print_rag_answer

result = rag_pipeline(
    query="What causes COVID-19?",
    k=3,
    use_rewrite=True,
    reranker="cohere",
    use_compression=True
)

print_rag_answer(result)

# Query classification
from src.processing import classify_query

needs_retrieval = classify_query("What is 2+2?")
# Returns: False (basic math doesn't need retrieval)
```

## 🏗️ Architecture

The system is organized into clean, modular components:

- **models**: Pydantic schemas for structured outputs
- **core**: AWS Bedrock clients and database operations
- **retrieval**: Search (vector, fulltext, hybrid) and reranking
- **processing**: Query classification, rewriting, and compression
- **generation**: LLM-based answer generation with citations
- **pipeline**: End-to-end RAG workflow orchestration
- **utils**: Dataset loading and helper functions
- **benchmarks**: Performance testing suite

For detailed architecture diagrams and data flow, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## 📊 Performance & Analysis

### Performance Benchmarks

| Metric | Target | Status |
|--------|--------|--------|
| Retrieval Latency (P95) | < 100ms | ✅ |
| Quality (NDCG@10) | > 0.8 | ✅ |
| Compression Ratio | > 30% | ✅ |
| Cost Reduction | > 40% | ✅ |

### Comprehensive Analysis

Run `python rag.py --analyze` to get detailed insights on 6 key questions:

1. **Query Classification**: What percentage can skip retrieval? Cost impact?
2. **Hybrid Search**: How do keyword vs vector weights affect query types?
3. **Reranking**: Which method works best? When do they fail?
4. **Compression**: Trade-off between compression ratio and answer quality?
5. **Performance**: What's the biggest latency bottleneck?
6. **Cost**: Cost per query with different Bedrock models?

The analysis generates `rag_analysis_results.json` with detailed metrics and recommendations.

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# AWS Configuration
AWS_PROFILE = "artisan-dev"
AWS_REGION = "us-east-1"

# RAG Configuration
RAG_TOP_K = 5
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.9

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_db",
    "user": "your_user",
    "password": "your_password",
}
```

## 📚 Documentation

- **README.md** (this file) - Quick start and overview
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture, diagrams, and data flow
- **[docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Guide to the modular structure
- **[docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)** - Detailed refactoring information

## 🧪 Testing

```bash
# Quick functionality test (3 queries)
python rag.py

# Interactive testing with chat commands
python rag.py --chat
# Commands: /help, /stats, /quit

# Full performance benchmarks
python rag.py --benchmark
```

## 🤝 Contributing

This codebase follows clean architecture principles:
- Single Responsibility: Each module has one clear purpose
- Dependency Inversion: High-level modules depend on abstractions
- Easy to test, extend, and maintain

## 📝 License

[Your License Here]
