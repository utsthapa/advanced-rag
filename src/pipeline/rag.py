"""Main RAG pipeline orchestration"""

import logging
import time
from typing import Dict

from src.generation.answer import generate_answer
from src.processing.compression import compress_documents
from src.processing.query import classify_query
from src.retrieval.search import hybrid_search

logger = logging.getLogger(__name__)


def rag_pipeline(
    query: str,
    k: int = 5,
    use_rewrite: bool = True,
    reranker: str = "cohere",
    use_compression: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Complete end-to-end RAG pipeline with answer generation

    Args:
        query: User's question
        k: Number of documents to retrieve
        use_rewrite: Whether to use query rewriting
        reranker: Reranking strategy ('cohere', 'bge', 'claude', or None)
        use_compression: Whether to compress documents
        verbose: Whether to print progress

    Returns:
        Dict with answer, sources, and metadata
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🤖 RAG PIPELINE")
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(
            f"Config: rewrite={use_rewrite}, reranker={reranker}, compression={use_compression}"
        )

    pipeline_start = time.perf_counter()

    # Step 1: Classification
    if verbose:
        print(f"\n1️⃣  CLASSIFICATION")

    should_retrieve = classify_query(query)

    if not should_retrieve:
        # Direct answer without retrieval
        return {
            "answer": "This query appears to be answerable without searching the scientific database. However, I can only provide answers based on the documents in my database. If you'd like information from the scientific literature, please try a more specific query.",
            "sources": [],
            "query": query,
            "pipeline": "direct_answer",
            "total_time": time.perf_counter() - pipeline_start,
        }

    # Step 2: Query Rewriting (if enabled) + Hybrid Retrieval
    if verbose:
        print(f"\n2️⃣  QUERY REWRITING & HYBRID RETRIEVAL")
        if use_rewrite:
            print(f"   ✍️  Step 2a: Optimizing query for vector and fulltext search")
        print(f"   🔍 Step 2b: Hybrid search (vector similarity + fulltext search)")

    retrieve_start = time.perf_counter()
    # Retrieve more documents before reranking for better quality
    from src.config import RAG_RERANK_MULTIPLIER

    retrieve_k = k * RAG_RERANK_MULTIPLIER if reranker else k

    results = hybrid_search(
        query, k=retrieve_k, use_rewrite=use_rewrite, reranker=None, verbose=verbose
    )

    retrieve_time = time.perf_counter() - retrieve_start

    if isinstance(results, dict) and results.get("type") == "direct_answer":
        return {
            "answer": "This query appears to be answerable without searching the scientific database.",
            "sources": [],
            "query": query,
            "pipeline": "direct_answer",
            "total_time": time.perf_counter() - pipeline_start,
        }

    if verbose:
        print(f"   ✅ Retrieved {len(results)} documents in {retrieve_time:.2f}s")

    # Step 3: Reranking
    rerank_time = 0
    all_rerank_results = {}

    if reranker and results:
        from src.retrieval.reranking import rerank_bge, rerank_claude, rerank_cohere

        # Special mode: run all three rerankers
        if reranker == "all":
            if verbose:
                print(f"\n3️⃣  RERANKING (ALL THREE MODELS)")
                print(f"   🔄 Running Cohere, BGE, and Claude rerankers")

            rerank_start = time.perf_counter()

            # Run all three rerankers in parallel
            try:
                cohere_start = time.perf_counter()
                cohere_results = rerank_cohere(query, results.copy(), k)
                cohere_time = time.perf_counter() - cohere_start
                all_rerank_results["cohere"] = {
                    "results": cohere_results,
                    "time": cohere_time,
                }
                if verbose:
                    print(f"   ✅ Cohere reranked in {cohere_time:.2f}s")
            except Exception as e:
                logger.error(f"Cohere reranking failed: {e}")
                if verbose:
                    print(f"   ❌ Cohere reranking failed: {e}")

            try:
                bge_start = time.perf_counter()
                bge_results = rerank_bge(query, results.copy(), k)
                bge_time = time.perf_counter() - bge_start
                all_rerank_results["bge"] = {"results": bge_results, "time": bge_time}
                if verbose:
                    print(f"   ✅ BGE reranked in {bge_time:.2f}s")
            except Exception as e:
                logger.error(f"BGE reranking failed: {e}")
                if verbose:
                    print(f"   ❌ BGE reranking failed: {e}")

            try:
                claude_start = time.perf_counter()
                claude_results = rerank_claude(query, results.copy(), k)
                claude_time = time.perf_counter() - claude_start
                all_rerank_results["claude"] = {
                    "results": claude_results,
                    "time": claude_time,
                }
                if verbose:
                    print(f"   ✅ Claude reranked in {claude_time:.2f}s")
            except Exception as e:
                logger.error(f"Claude reranking failed: {e}")
                if verbose:
                    print(f"   ❌ Claude reranking failed: {e}")

            rerank_time = time.perf_counter() - rerank_start

            # Use Cohere as default (most reliable)
            if "cohere" in all_rerank_results:
                results = all_rerank_results["cohere"]["results"]
            elif "bge" in all_rerank_results:
                results = all_rerank_results["bge"]["results"]
            elif "claude" in all_rerank_results:
                results = all_rerank_results["claude"]["results"]
            else:
                # Fallback to original results
                results = results[:k]

            if verbose:
                print(f"   ✅ All rerankers completed in {rerank_time:.2f}s")
                print(f"   💡 Using Cohere results for downstream processing")

        else:
            # Single reranker mode
            if verbose:
                print(f"\n3️⃣  RERANKING ({reranker.upper()})")
                print(f"   🔄 Reordering results by relevance using {reranker} model")

            rerank_start = time.perf_counter()
            if reranker == "cohere":
                results = rerank_cohere(query, results, k)
            elif reranker == "bge":
                results = rerank_bge(query, results, k)
            elif reranker == "claude":
                results = rerank_claude(query, results, k)
            rerank_time = time.perf_counter() - rerank_start

            if verbose:
                print(f"   ✅ Reranked to top {k} in {rerank_time:.2f}s")
    else:
        results = results[:k]

    # Filter out documents with very low relevance scores (likely noise/irrelevant)
    if results:
        filtered_results = []
        removed_count = 0
        for doc in results:
            score = doc.get("rerank_score") or doc.get("similarity", 0)
            # Keep documents with any rerank score, or similarity > 0.01
            # This filters out completely irrelevant documents while keeping marginal ones
            if doc.get("rerank_score") is not None or score > 0.01:
                filtered_results.append(doc)
            else:
                removed_count += 1

        if removed_count > 0 and verbose:
            logger.info(f"Filtered out {removed_count} very low-relevance documents")
        results = filtered_results

    # Step 4: Contextual Compression
    compress_time = 0
    if use_compression and results:
        if verbose:
            print(f"\n4️⃣  CONTEXTUAL COMPRESSION")
            print(f"   📦 Compressing documents while preserving relevant information")

        compress_start = time.perf_counter()
        results = compress_documents(query, results, show_stats=verbose)
        compress_time = time.perf_counter() - compress_start

        if verbose:
            print(f"   ✅ Compressed in {compress_time:.2f}s")

    # Step 5: Answer Generation
    if verbose:
        print(f"\n5️⃣  ANSWER GENERATION")
        print(f"   ✨ Generating final answer from compressed context")

    answer_result = generate_answer(query, results, use_compression)

    if verbose:
        print(
            f"   ✅ Generated answer in {answer_result.get('generation_time', 0):.2f}s"
        )

    # Ensure answer field exists - if missing, something went wrong
    if "answer" not in answer_result or not answer_result.get("answer"):
        logger.warning("Answer generation did not return an answer field")
        if "answer" not in answer_result:
            answer_result["answer"] = (
                "Error: Answer was not generated. Please check the logs for details."
            )

    # Compile final result
    total_time = time.perf_counter() - pipeline_start

    result = {
        **answer_result,
        "pipeline": "full_rag",
        "timings": {
            "retrieval": retrieve_time,
            "reranking": rerank_time,
            "compression": compress_time,
            "generation": answer_result.get("generation_time", 0),
            "total": total_time,
        },
        "config": {
            "use_rewrite": use_rewrite,
            "reranker": reranker,
            "use_compression": use_compression,
            "k": k,
        },
        "all_rerank_results": all_rerank_results if all_rerank_results else None,
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f}s")

    return result


def print_rag_answer(result: Dict):
    """Pretty print RAG answer with sources"""
    print(f"\n{'='*80}")
    print(f"💬 ANSWER")
    print(f"{'='*80}")
    print(f"\nQ: {result.get('query', 'Unknown query')}")

    # Ensure answer is always displayed
    answer = result.get("answer", "")
    if not answer or answer.strip() == "":
        answer = "No answer was generated. This may indicate an error in the answer generation step."
        print(f"\n⚠️ {answer}")
    else:
        print(f"\nA: {answer}")

    if result.get("sources"):
        print(f"\n{'─'*80}")
        print(f"📚 SOURCES ({len(result['sources'])} documents)")
        print(f"{'─'*80}")

        for source in result["sources"]:
            num = source["number"]
            score = source["similarity"]
            content_preview = (
                source["content"][:200] + "..."
                if len(source["content"]) > 200
                else source["content"]
            )

            print(f"\n[{num}] Relevance: {score:.3f}")
            print(f"    {content_preview}")

            if source.get("key_points"):
                print(f"    🔑 Key Points:")
                for point in source["key_points"][:3]:
                    print(f"       • {point}")

    # Add relevance warning if scores are very low
    if result.get("avg_relevance", 0) < 0.1:
        print(
            f"\n⚠️  WARNING: Low relevance scores detected (avg: {result.get('avg_relevance', 0):.3f})"
        )
        print(f"    The retrieved documents may not be highly relevant to your query.")

    if result.get("timings"):
        timings = result["timings"]
        print(f"\n{'─'*80}")
        print(f"⏱️  TIMING BREAKDOWN")
        print(f"{'─'*80}")
        print(f"  Retrieval:   {timings.get('retrieval', 0):.2f}s")
        if timings.get("reranking", 0) > 0:
            print(f"  Reranking:   {timings.get('reranking', 0):.2f}s")
        if timings.get("compression", 0) > 0:
            print(f"  Compression: {timings.get('compression', 0):.2f}s")
        print(f"  Generation:  {timings.get('generation', 0):.2f}s")
        print(f"  Total:       {timings.get('total', 0):.2f}s")

    # Show all reranker results if available
    if result.get("all_rerank_results"):
        print(f"\n{'─'*80}")
        print(f"🔄 RERANKER COMPARISON")
        print(f"{'─'*80}")
        all_results = result["all_rerank_results"]

        for reranker_name in ["cohere", "bge", "claude"]:
            if reranker_name in all_results:
                rerank_data = all_results[reranker_name]
                rerank_results = rerank_data["results"]
                rerank_time = rerank_data["time"]

                print(f"\n{reranker_name.upper()} Reranker (took {rerank_time:.2f}s):")
                print(f"  Top {min(3, len(rerank_results))} results:")

                for idx, doc in enumerate(rerank_results[:3], 1):
                    score = doc.get("rerank_score", doc.get("similarity", 0))
                    content_preview = (
                        doc.get("content", "")[:150] + "..."
                        if len(doc.get("content", "")) > 150
                        else doc.get("content", "")
                    )
                    print(f"    [{idx}] Score: {score:.3f}")
                    print(f"        {content_preview}")

    print(f"\n{'='*80}")
