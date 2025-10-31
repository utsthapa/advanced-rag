# Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                            │
│                    • Load data                                   │
│                    • Quick test                                  │
│                    • Chat mode                                   │
│                    • Benchmarks                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    src/pipeline/rag.py                           │
│                    RAG Pipeline Orchestrator                     │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────┐    │
│   │1.Classify│2.Retrieve│3.Rerank  │4.Compress│5.Generate│    │
│   └──────────┴──────────┴──────────┴──────────┴──────────┘    │
└───┬────────────┬────────────┬────────────┬────────────┬────────┘
    │            │            │            │            │
    ▼            ▼            ▼            ▼            ▼
┌───────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Process│  │Retrieval │  │Retrieval │  │Process   │  │Generation│
│       │  │          │  │          │  │          │  │          │
│Query  │  │Search    │  │Reranking │  │Compress  │  │Answer    │
│       │  │          │  │          │  │          │  │          │
└───┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
    │           │             │             │             │
    │           │             │             │             │
    ▼           ▼             ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────┐
│                      Core Infrastructure                       │
│  ┌──────────────────┐           ┌──────────────────┐         │
│  │  core/clients.py │           │ core/database.py │         │
│  │                  │           │                  │         │
│  │  AWS Bedrock     │           │  PostgreSQL      │         │
│  │  • Embeddings    │           │  • Documents     │         │
│  │  • LLMs          │           │  • Chunks        │         │
│  │  • Classifiers   │           │  • pgvector      │         │
│  └──────────────────┘           └──────────────────┘         │
└───────────────────────────────────────────────────────────────┘
    │                                       │
    ▼                                       ▼
┌──────────────┐                  ┌──────────────┐
│ AWS Bedrock  │                  │ PostgreSQL   │
│ • Titan      │                  │ • Full-text  │
│ • Claude     │                  │ • Vector     │
│ • Nova       │                  │ • RRF        │
│ • Cohere     │                  │              │
└──────────────┘                  └──────────────┘
```

## Component Interaction Flow

### 1. Query Processing Flow
```
User Query
    │
    ▼
┌─────────────────────┐
│ classify_query()    │ ◄─── models/QueryClassification
│ (processing/query)  │
└──────────┬──────────┘
           │
           ├─► Skip retrieval (basic question)
           │
           └─► Need retrieval ──┐
                                │
                                ▼
                        ┌───────────────┐
                        │ rewrite_query │ ◄─── models/QueryRewrite
                        │               │
                        └───────┬───────┘
                                │
                                ├─► Vector-optimized query
                                └─► Fulltext-optimized query
```

### 2. Retrieval Flow
```
Optimized Queries
    │
    ▼
┌─────────────────────┐
│  hybrid_search()    │
│ (retrieval/search)  │
└──────────┬──────────┘
           │
           ├─► Vector search ──────────┐
           │   (semantic similarity)   │
           │                           │
           └─► Fulltext search ────────┤
               (keyword matching)      │
                                       │
                                       ▼
                               ┌───────────────┐
                               │  RRF Fusion   │
                               │  (database)   │
                               └───────┬───────┘
                                       │
                                       ▼
                               ┌───────────────┐
                               │   Results     │
                               │ (k documents) │
                               └───────────────┘
```

### 3. Reranking Flow
```
Retrieved Documents
    │
    ▼
┌─────────────────────┐
│  rerank_*()         │ ◄─── models/RelevanceScore
│ (retrieval/rerank)  │
└──────────┬──────────┘
           │
           ├─► Cohere Rerank v3.5 ───┐
           │   (cloud, highest accuracy)
           │                          │
           ├─► BGE Reranker ──────────┤
           │   (local, good accuracy)  │
           │                          │
           └─► Claude Reranking ──────┤
               (LLM, with reasoning)  │
                                      │
                                      ▼
                              ┌───────────────┐
                              │   Top K       │
                              │  Documents    │
                              └───────────────┘
```

### 4. Compression Flow
```
Reranked Documents
    │
    ▼
┌─────────────────────┐
│ compress_documents()│ ◄─── models/CompressedDocument
│ (processing/compress)│
└──────────┬──────────┘
           │
           │ For each document:
           │
           ├─► Extract relevant content
           ├─► Remove irrelevant parts
           ├─► Preserve key facts
           └─► Generate key points
                      │
                      ▼
              ┌───────────────┐
              │  Compressed   │
              │  Documents    │
              │ (30-50% less) │
              └───────────────┘
```

### 5. Answer Generation Flow
```
Compressed Documents
    │
    ▼
┌─────────────────────┐
│ generate_answer()   │
│ (generation/answer) │
└──────────┬──────────┘
           │
           ├─► Build context from docs
           ├─► Add document numbers [1],[2]...
           └─► Invoke Claude Haiku
                      │
                      ▼
              ┌───────────────┐
              │  Final Answer │
              │  with sources │
              └───────────────┘
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    pipeline/rag.py                           │
│              (Orchestrates all components)                   │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Process │ │Retrieve│ │Rerank  │ │Compress│ │Generate│
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
       │          │          │          │          │
       └──────────┴──────────┴──────────┴──────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   core/         │
              │   • clients.py  │
              │   • database.py │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   models/       │
              │   • schemas.py  │
              └─────────────────┘
```

## Data Flow Example

Let's trace a complete query through the system:

```
1. User Input
   "How does CRISPR gene editing work?"

2. Classification (processing/query.py)
   ✓ Needs retrieval: TRUE
   ✓ Confidence: 0.95
   ✓ Type: scientific

3. Query Rewriting (processing/query.py)
   ✓ Vector query: "CRISPR Cas9 genetic modification mechanisms molecular biology gene editing techniques"
   ✓ Fulltext query: "CRISPR gene editing genetic modification DNA cutting Cas9 enzyme"

4. Hybrid Search (retrieval/search.py + core/database.py)
   ✓ Vector search: Top 100 results by similarity
   ✓ Fulltext search: Top 100 results by keyword match
   ✓ RRF fusion: Combine with k=60
   ✓ Retrieved: 10 documents

5. Reranking (retrieval/reranking.py)
   ✓ Method: Cohere Rerank v3.5
   ✓ Reranked: Top 5 documents by relevance

6. Compression (processing/compression.py)
   ✓ Original: 5 docs × ~500 tokens = 2,500 tokens
   ✓ Compressed: 5 docs × ~300 tokens = 1,500 tokens
   ✓ Savings: 40% reduction

7. Answer Generation (generation/answer.py)
   ✓ Context: 1,500 tokens from 5 documents
   ✓ Model: Claude 3 Haiku
   ✓ Answer: Generated with citations [1], [2], [3]

8. Response to User
   ✓ Answer with 3-5 cited sources
   ✓ Relevance scores displayed
   ✓ Key points highlighted
   ✓ Total time: ~2-3 seconds
```

## File Size Comparison

### Before Refactoring
```
rag.py                       ████████████████████████████████████████ 2,855 lines
db.py                        ████████ 378 lines
performance_benchmarks.py    █████████████ 826 lines
```

### After Refactoring
```
config.py                    █ 50 lines
models/schemas.py            █ 70 lines
core/clients.py              ███ 100 lines
core/database.py             ████ 150 lines
retrieval/search.py          ████ 150 lines
retrieval/reranking.py       █████ 200 lines
processing/query.py          ███ 120 lines
processing/compression.py    ███ 130 lines
generation/answer.py         ████ 150 lines
pipeline/rag.py              █████ 200 lines
utils/data_loader.py         █████ 200 lines
benchmarks/performance.py    ████████████ 826 lines (same)
main.py                      █████ 200 lines
```

**Maximum file size**: 826 lines (benchmarks, unchanged)
**Average file size**: ~150 lines
**Largest new file**: 200 lines (vs 2,855 before!)

## Key Architectural Principles

### 1. **Separation of Concerns**
Each module handles ONE thing:
- Models: Data structures
- Core: Infrastructure
- Retrieval: Search operations
- Processing: Query/document manipulation
- Generation: Answer creation
- Pipeline: Orchestration

### 2. **Dependency Inversion**
High-level modules (pipeline) depend on abstractions (interfaces), not concrete implementations.

### 3. **Single Responsibility**
Each file has one clear job. Easy to understand and modify.

### 4. **DRY (Don't Repeat Yourself)**
Common functionality extracted into shared modules (core, utils).

### 5. **Open/Closed Principle**
Easy to extend (add new rerankers) without modifying existing code.

## Summary

The new architecture provides:
- ✅ Clear separation of concerns
- ✅ Easy to understand flow
- ✅ Simple to test components
- ✅ Safe to modify individual parts
- ✅ Professional structure
- ✅ Scalable design

All achieved while maintaining 100% of original functionality! 🎉

