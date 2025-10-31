# Contributing Guide

## Project Structure

This project follows a modular architecture where each component has a single, clear responsibility.

### Module Organization

```
src/
├── models/         # Data structures (Pydantic schemas)
├── core/           # Infrastructure (AWS clients, database)
├── retrieval/      # Search operations
├── processing/     # Query and document processing
├── generation/     # Answer generation
├── pipeline/       # Workflow orchestration
├── utils/          # Helper functions
└── benchmarks/     # Performance testing
```

## Adding New Features

### Example: Adding a New Reranker

1. **Add function to `src/retrieval/reranking.py`**:
```python
def rerank_my_new_method(query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank documents using my new method"""
    # Your implementation here
    return reranked_docs
```

2. **Update exports in `src/retrieval/__init__.py`**:
```python
from .reranking import rerank_my_new_method
__all__ = [..., "rerank_my_new_method"]
```

3. **Use it in search**:
```python
from src.retrieval import search
results = search(query, method="hybrid", reranker="my_new_method")
```

### Example: Adding a New Search Method

1. **Add function to `src/retrieval/search.py`**:
```python
def my_custom_search(query: str, k: int = 5) -> List[Dict]:
    """My custom search implementation"""
    # Your implementation
    return results
```

2. **Update the main `search()` function**:
```python
def search(query: str, method: str = "vector", ...):
    if method == "my_custom":
        return my_custom_search(query, k)
    # ...
```

## Code Style

- **Keep files small**: 50-350 lines per file
- **Single responsibility**: Each function/class does ONE thing
- **Clear names**: Use descriptive function and variable names
- **Type hints**: Always use type annotations
- **Docstrings**: Document all public functions

### Example
```python
def compress_document(query: str, document: dict) -> dict:
    """
    Compress a single document using Claude.

    Args:
        query: The search query
        document: Document dict with 'content' field

    Returns:
        Document with compressed content and metadata
    """
    # Implementation
```

## Testing

```bash
# Run quick test
python rag.py

# Run full benchmarks
python rag.py --benchmark

# Test specific component
python -c "from src.retrieval import search; print(search('test', k=1))"
```

## Architecture Principles

1. **Dependency Inversion**: High-level modules depend on abstractions
2. **Single Responsibility**: Each module has one clear job
3. **Open/Closed**: Easy to extend, no need to modify existing code
4. **DRY**: Don't repeat yourself

## Pull Request Guidelines

1. Keep changes focused on one feature/fix
2. Update relevant documentation
3. Add tests if adding new functionality
4. Follow existing code style
5. Update `README.md` if adding user-facing features

## Questions?

See `docs/` folder for detailed architecture information.

