"""Search functions for vector, fulltext, and hybrid retrieval"""

import logging
from typing import Dict, List

from src.config import (
    BYPASS_CLASSIFICATION,
    BYPASS_REWRITING,
    RAG_RERANK_MULTIPLIER,
    RAG_TOP_K,
    SKIP_RERANKING,
    USE_HYBRID_SEARCH,
)
from src.core import DatabaseManager, embed_query_cached, get_embeddings
from src.processing.query import classify_query, rewrite_query

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    k: int = RAG_TOP_K,
    use_rewrite: bool = True,
    reranker=None,
    verbose: bool = True,
) -> List[Dict]:
    """Pure vector search with optional query rewriting and reranking"""
    # Check if query needs retrieval
    if not classify_query(query, verbose=verbose):
        return {
            "message": "Query answered directly - no retrieval needed",
            "type": "direct_answer",
        }

    embeddings = get_embeddings()
    db = DatabaseManager()

    # Apply query rewriting
    optimized_query = query
    if use_rewrite:
        rewrite_result = rewrite_query(query, verbose=verbose)
        optimized_query = rewrite_result["vector_optimized"]
        if verbose:
            print(f"   ✅ Using vector-optimized query for embedding")

    # Generate query embedding from optimized query
    query_embedding = embeddings.embed_query(optimized_query)

    # Get more results for reranking if reranker is specified
    # Retrieve more candidates for reranking, then return top k after reranking
    retrieve_k = k * RAG_RERANK_MULTIPLIER if reranker else k

    # Search database
    results = db.vector_search(query_embedding, retrieve_k, search_chunks=True)
    db.close()

    # Apply reranking if specified
    if reranker and results:
        from src.retrieval.reranking import (
            rerank_bge,
            rerank_claude,
            rerank_cohere,
            rerank_ensemble,
        )

        if verbose:
            print(f"   🔄 Applying {reranker} reranking...")
        if reranker == "cohere":
            results = rerank_cohere(query, results, k)
        elif reranker == "bge":
            results = rerank_bge(query, results, k)
        elif reranker == "claude":
            results = rerank_claude(query, results, k)
        elif reranker == "ensemble":
            results = rerank_ensemble(query, results, k)

    return results


def fulltext_search(
    query: str,
    k: int = RAG_TOP_K,
    use_rewrite: bool = True,
    reranker=None,
    verbose: bool = True,
) -> List[Dict]:
    """Pure full-text search with optional query rewriting and reranking"""
    db = DatabaseManager()

    # Apply query rewriting
    optimized_query = query
    if use_rewrite:
        rewrite_result = rewrite_query(query, verbose=verbose)
        optimized_query = rewrite_result["fulltext_optimized"]
        if verbose:
            print(f"   ✅ Using fulltext-optimized query for search")

    # Get more results for reranking if reranker is specified
    # Retrieve more candidates for reranking, then return top k after reranking
    retrieve_k = k * RAG_RERANK_MULTIPLIER if reranker else k

    # Search database with optimized query
    results = db.fulltext_search(optimized_query, retrieve_k, search_chunks=True)
    db.close()

    # Apply reranking if specified
    if reranker and results:
        from src.retrieval.reranking import (
            rerank_bge,
            rerank_claude,
            rerank_cohere,
            rerank_ensemble,
        )

        if verbose:
            print(f"   🔄 Applying {reranker} reranking...")
        if reranker == "cohere":
            results = rerank_cohere(query, results, k)
        elif reranker == "bge":
            results = rerank_bge(query, results, k)
        elif reranker == "claude":
            results = rerank_claude(query, results, k)
        elif reranker == "ensemble":
            results = rerank_ensemble(query, results, k)

    return results


def hybrid_search(
    query: str,
    k: int = RAG_TOP_K,
    rrf_k: int = 60,  # Increased from 40 to 60 for better RRF fusion
    use_rewrite: bool = True,
    reranker=None,
    verbose: bool = True,
) -> List[Dict]:
    """Hybrid search with method-specific query optimization and optional reranking"""
    # Check if query needs retrieval
    if not classify_query(query, verbose=verbose):
        return {
            "message": "Query answered directly - no retrieval needed",
            "type": "direct_answer",
        }

    embeddings = get_embeddings()
    db = DatabaseManager()

    # Rewrite once, use both optimized versions
    if use_rewrite:
        try:
            rewrite_result = rewrite_query(query, verbose=verbose)
            vector_query = rewrite_result["vector_optimized"]
            fulltext_query = rewrite_result["fulltext_optimized"]
            if verbose:
                print(f"   ✅ Using method-specific optimized queries")
        except Exception as e:
            logger.error(f"Query rewriting failed in hybrid_search: {e}")
            vector_query = fulltext_query = query
            if verbose:
                print(f"   ⚠️  Using original query due to rewriting failure")
    else:
        vector_query = fulltext_query = query

    # Generate embedding from vector-optimized query
    query_embedding = embeddings.embed_query(vector_query)

    # Get more results for reranking if reranker is specified
    # Retrieve more candidates for reranking, then return top k after reranking
    retrieve_k = k * RAG_RERANK_MULTIPLIER if reranker else k

    # Use hybrid search with different queries
    results = db.hybrid_search_rrf(
        query_embedding,
        vector_query,
        fulltext_query,
        retrieve_k,
        rrf_k,
        search_chunks=True,
    )
    db.close()

    # Apply reranking if specified
    if reranker and results:
        from src.retrieval.reranking import (
            rerank_bge,
            rerank_claude,
            rerank_cohere,
            rerank_ensemble,
        )

        if verbose:
            print(f"   🔄 Applying {reranker} reranking...")
        if reranker == "cohere":
            results = rerank_cohere(query, results, k)
        elif reranker == "bge":
            results = rerank_bge(query, results, k)
        elif reranker == "claude":
            results = rerank_claude(query, results, k)
        elif reranker == "ensemble":
            results = rerank_ensemble(query, results, k)

    return results


def search(
    query: str,
    method: str = "vector",
    k: int = RAG_TOP_K,
    use_rewrite: bool = True,
    reranker=None,
    verbose: bool = True,
):
    """Main search interface with optional reranking"""

    if method == "vector":
        return vector_search(query, k, use_rewrite, reranker, verbose=verbose)
    elif method == "fulltext":
        return fulltext_search(query, k, use_rewrite, reranker, verbose=verbose)
    elif method == "hybrid":
        return hybrid_search(
            query, k, use_rewrite=use_rewrite, reranker=reranker, verbose=verbose
        )
    else:
        raise ValueError("Method must be 'vector', 'fulltext', or 'hybrid'")
