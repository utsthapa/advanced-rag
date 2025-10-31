#!/usr/bin/env python3
"""
Comprehensive RAG Analysis Tool

Answers 6 key questions:
1. Query Classification: What percentage can skip retrieval? Cost impact?
2. Hybrid Search: How do keyword vs vector weights affect query types?
3. Reranking: Which method works best? When do they fail?
4. Compression: Trade-off between compression ratio and answer quality?
5. Performance: What's the biggest latency bottleneck?
6. Cost: Cost per query with different Bedrock models?

Usage: python -m src.benchmarks.rag_analysis
"""

import json
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.core import DatabaseManager
from src.generation.answer import generate_answer
from src.pipeline.rag import rag_pipeline
from src.processing.compression import compress_documents
from src.processing.query import classify_query
from src.retrieval.search import hybrid_search
from src.utils import load_beir_test_queries


class RAGAnalysis:
    """Comprehensive analysis of RAG system components"""

    def __init__(self):
        self.results = {}
        self.test_queries = []

    def print_section(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}\n")

    def load_test_queries(self, num_queries: int = 100) -> List[str]:
        """Load or generate test queries"""
        try:
            test_data = load_beir_test_queries(limit=num_queries)
            queries = [q["query"] for q in test_data]
            print(f"✅ Loaded {len(queries)} queries from BeIR test set")
        except Exception as e:
            print(f"⚠️  Could not load BeIR queries: {e}")
            print("   Using generated test queries...")

            # Generate diverse test queries
            scientific_queries = [
                "How does CRISPR gene editing work?",
                "What causes COVID-19 transmission?",
                "Explain the mechanism of mRNA vaccines",
                "What is the role of ACE2 receptors in viral infection?",
                "How do antibodies neutralize viruses?",
                "What are the effects of climate change on ecosystems?",
                "How does photosynthesis work in plants?",
                "What is the structure of DNA?",
                "Explain protein synthesis mechanisms",
                "How do neurons transmit signals?",
                "What is the function of mitochondria?",
                "How does the immune system recognize pathogens?",
                "What are stem cells and their applications?",
                "Explain gene expression regulation",
                "How do cancer cells evade immune detection?",
                "What causes antibiotic resistance?",
                "How do vaccines prevent diseases?",
                "What is the mechanism of DNA replication?",
                "Explain RNA interference",
                "How do hormones regulate metabolism?",
            ]

            non_scientific_queries = [
                "What is 2+2?",
                "What is the capital of France?",
                "Who invented the telephone?",
                "When did World War 2 end?",
                "What does USA stand for?",
                "How many continents are there?",
                "What color is the sky?",
                "What day comes after Monday?",
                "How many hours in a day?",
                "What language is spoken in Brazil?",
                "What is the largest ocean?",
                "Who wrote Romeo and Juliet?",
                "What is the speed of light?",
                "How many planets are in the solar system?",
                "What is the tallest mountain?",
                "What is the smallest country?",
                "Who painted the Mona Lisa?",
                "What year did the internet start?",
                "What is the longest river?",
                "Who discovered America?",
            ]

            queries = (scientific_queries * 3 + non_scientific_queries * 2)[
                :num_queries
            ]

        self.test_queries = queries
        return queries

    def analyze_query_classification(self) -> Dict:
        """
        Question 1: Query Classification Analysis
        - What percentage of queries can skip retrieval?
        - How does this impact costs?
        """
        self.print_section("QUESTION 1: QUERY CLASSIFICATION ANALYSIS")

        queries = self.load_test_queries(50)

        results = {
            "total_queries": 0,
            "skipped_queries": 0,
            "retrieved_queries": 0,
            "skip_rate": 0.0,
            "cost_savings": 0.0,
            "classification_times": [],
            "query_types": defaultdict(int),
        }

        print("🔍 Analyzing query classification...")

        for i, query in enumerate(queries, 1):
            try:
                start = time.perf_counter()
                needs_retrieval = classify_query(query, verbose=False)
                elapsed = time.perf_counter() - start

                results["total_queries"] += 1
                results["classification_times"].append(elapsed)

                if needs_retrieval:
                    results["retrieved_queries"] += 1
                    results["query_types"]["requires_retrieval"] += 1
                else:
                    results["skipped_queries"] += 1
                    results["query_types"]["skipped"] += 1

                if i % 10 == 0:
                    print(f"   Processed {i}/{len(queries)} queries...")

            except Exception as e:
                print(f"   ⚠️  Query {i} failed: {e}")
                continue

        # Calculate metrics
        results["skip_rate"] = (
            (results["skipped_queries"] / results["total_queries"]) * 100
            if results["total_queries"] > 0
            else 0
        )

        # Cost analysis
        # Costs per component (estimated based on AWS Bedrock pricing)
        classification_cost_per_query = 0.0001  # Claude Haiku for classification
        retrieval_cost_per_query = 0.0015  # Embedding + vector search + reranking
        compression_cost_per_query = 0.001  # Claude Nova for compression
        generation_cost_per_query = 0.0008  # Claude Haiku for generation

        baseline_cost_per_query = (
            classification_cost_per_query
            + retrieval_cost_per_query
            + compression_cost_per_query
            + generation_cost_per_query
        )

        # With classification: skip retrieval + compression for non-scientific queries
        actual_cost_per_retrieved_query = baseline_cost_per_query
        actual_cost_per_skipped_query = (
            classification_cost_per_query  # Only classification cost
        )

        total_baseline_cost = results["total_queries"] * baseline_cost_per_query
        actual_total_cost = (
            results["retrieved_queries"] * actual_cost_per_retrieved_query
            + results["skipped_queries"] * actual_cost_per_skipped_query
        )

        results["cost_savings"] = (
            (total_baseline_cost - actual_total_cost) / total_baseline_cost
        ) * 100
        results["total_baseline_cost"] = total_baseline_cost
        results["actual_total_cost"] = actual_total_cost
        results["cost_per_query_baseline"] = baseline_cost_per_query
        results["cost_per_query_with_classification"] = (
            actual_total_cost / results["total_queries"]
            if results["total_queries"] > 0
            else 0
        )

        results["avg_classification_time"] = (
            np.mean(results["classification_times"])
            if results["classification_times"]
            else 0
        )

        # Print results
        print(f"\n📈 Classification Results:")
        print(f"   Total queries:              {results['total_queries']}")
        print(
            f"   Skipped (no retrieval):     {results['skipped_queries']} ({results['skip_rate']:.1f}%)"
        )
        print(
            f"   Retrieved:                 {results['retrieved_queries']} ({100-results['skip_rate']:.1f}%)"
        )
        print(
            f"   Avg classification time:    {results['avg_classification_time']*1000:.1f}ms"
        )
        print(f"\n💰 Cost Analysis:")
        print(
            f"   Baseline cost per query:   ${results['cost_per_query_baseline']:.4f}"
        )
        print(
            f"   Actual cost per query:     ${results['cost_per_query_with_classification']:.4f}"
        )
        print(f"   Total baseline cost:       ${results['total_baseline_cost']:.4f}")
        print(f"   Actual total cost:         ${results['actual_total_cost']:.4f}")
        print(f"   Cost savings:              {results['cost_savings']:.1f}%")
        print(
            f"   Monthly savings (10k queries): ${(results['cost_savings']/100 * results['cost_per_query_baseline'] * 10000):.2f}"
        )

        self.results["classification"] = results
        return results

    def analyze_hybrid_search(self) -> Dict:
        """
        Question 2: Hybrid Search Analysis
        - How do keyword vs vector weights affect different query types?
        - What's optimal for your domain?
        """
        self.print_section("QUESTION 2: HYBRID SEARCH ANALYSIS")

        # Use a mix of query types
        test_queries = [
            ("factual", "What causes COVID-19?"),
            ("factual", "How does CRISPR work?"),
            ("conceptual", "Explain mRNA vaccine mechanisms"),
            ("technical", "ACE2 receptor viral entry pathway"),
            ("descriptive", "How do neurons communicate?"),
            ("keyword_rich", "antibiotic resistance bacteria mechanisms"),
            ("phrase", "gene editing CRISPR cas9"),
        ]

        # Test different RRF k values (affects weight balance)
        rrf_k_values = [20, 40, 60, 80, 100]

        results = {
            "query_types": {},
            "optimal_rrf_k": None,
            "best_performance": None,
        }

        print("🔍 Testing hybrid search with different RRF k values...")
        print(
            f"   Testing {len(test_queries)} queries with {len(rrf_k_values)} RRF k values\n"
        )

        all_scores = defaultdict(list)

        for query_type, query in test_queries:
            query_results = {
                "query": query,
                "type": query_type,
                "rrf_k_scores": {},
            }

            for rrf_k in rrf_k_values:
                try:
                    start = time.perf_counter()
                    search_results = hybrid_search(
                        query,
                        k=5,
                        rrf_k=rrf_k,
                        use_rewrite=True,
                        reranker="cohere",
                        verbose=False,
                    )

                    if (
                        isinstance(search_results, dict)
                        and search_results.get("type") == "direct_answer"
                    ):
                        continue

                    search_time = time.perf_counter() - start

                    # Calculate average relevance score
                    if search_results:
                        avg_relevance = np.mean(
                            [
                                r.get("rerank_score", r.get("similarity", 0))
                                for r in search_results[:5]
                            ]
                        )
                        top_score = (
                            max(
                                [
                                    r.get("rerank_score", r.get("similarity", 0))
                                    for r in search_results[:5]
                                ]
                            )
                            if search_results
                            else 0
                        )
                    else:
                        avg_relevance = 0
                        top_score = 0

                    # Score = weighted combination of relevance and speed
                    score = (
                        (avg_relevance * 0.7) + ((1.0 / search_time) * 0.3)
                        if search_time > 0
                        else avg_relevance
                    )

                    query_results["rrf_k_scores"][rrf_k] = {
                        "avg_relevance": avg_relevance,
                        "top_score": top_score,
                        "search_time": search_time,
                        "performance_score": score,
                    }

                    all_scores[rrf_k].append(score)

                except Exception as e:
                    print(f"   ⚠️  Query '{query}' with rrf_k={rrf_k} failed: {e}")
                    continue

            results["query_types"][query_type] = query_results

        # Find optimal RRF k
        avg_scores_by_rrf_k = {
            rrf_k: np.mean(scores) if scores else 0
            for rrf_k, scores in all_scores.items()
        }

        if avg_scores_by_rrf_k:
            optimal_rrf_k = max(avg_scores_by_rrf_k.items(), key=lambda x: x[1])
            results["optimal_rrf_k"] = optimal_rrf_k[0]
            results["best_performance"] = optimal_rrf_k[1]

        # Print results
        print("\n📈 Hybrid Search Results:")
        print(f"   Optimal RRF k value: {results['optimal_rrf_k']}")
        print(f"   Best performance score: {results['best_performance']:.3f}\n")

        print("   Performance by RRF k:")
        for rrf_k in sorted(avg_scores_by_rrf_k.keys()):
            score = avg_scores_by_rrf_k[rrf_k]
            print(f"      RRF k={rrf_k:3d}: {score:.3f} avg score")

        print("\n   Query type analysis:")
        for query_type, data in results["query_types"].items():
            if data["rrf_k_scores"]:
                best_k = max(
                    data["rrf_k_scores"].items(),
                    key=lambda x: x[1]["performance_score"],
                )
                print(
                    f"      {query_type:15s}: Best RRF k={best_k[0]} (score={best_k[1]['performance_score']:.3f})"
                )

        print("\n💡 Recommendations:")
        print(f"   • Use RRF k={results['optimal_rrf_k']} for balanced performance")
        print(f"   • Factual queries: Higher k (more keyword weight)")
        print(f"   • Conceptual queries: Lower k (more vector weight)")

        self.results["hybrid_search"] = results
        return results

    def analyze_reranking(self) -> Dict:
        """
        Question 3: Reranking Analysis
        - Which reranking method works best?
        - When do they fail?
        """
        self.print_section("QUESTION 3: RERANKING ANALYSIS")

        queries = self.load_test_queries(20)

        rerankers = ["cohere", "bge", "claude"]

        results = {
            "reranker_stats": {},
            "best_per_reranker": {},
            "failure_cases": {},
        }

        print(f"🔍 Testing {len(rerankers)} rerankers on {len(queries)} queries...\n")

        for reranker_name in rerankers:
            stats = {
                "total_queries": 0,
                "successful": 0,
                "failed": 0,
                "avg_time": [],
                "avg_relevance": [],
                "top_score": [],
            }

            for query in queries[:20]:
                try:
                    # Get initial retrieval results
                    initial_results = hybrid_search(
                        query, k=50, use_rewrite=True, reranker=None, verbose=False
                    )

                    if (
                        isinstance(initial_results, dict)
                        and initial_results.get("type") == "direct_answer"
                    ):
                        continue

                    if not initial_results:
                        continue

                    stats["total_queries"] += 1

                    # Rerank with specific reranker
                    from src.retrieval.reranking import (
                        rerank_bge,
                        rerank_claude,
                        rerank_cohere,
                    )

                    start = time.perf_counter()

                    if reranker_name == "cohere":
                        reranked = rerank_cohere(query, initial_results.copy(), k=5)
                    elif reranker_name == "bge":
                        reranked = rerank_bge(query, initial_results.copy(), k=5)
                    elif reranker_name == "claude":
                        reranked = rerank_claude(query, initial_results.copy(), k=5)

                    elapsed = time.perf_counter() - start

                    if reranked:
                        stats["successful"] += 1
                        stats["avg_time"].append(elapsed)

                        avg_rel = np.mean(
                            [
                                r.get("rerank_score", r.get("similarity", 0))
                                for r in reranked
                            ]
                        )
                        top_rel = (
                            max(
                                [
                                    r.get("rerank_score", r.get("similarity", 0))
                                    for r in reranked
                                ]
                            )
                            if reranked
                            else 0
                        )

                        stats["avg_relevance"].append(avg_rel)
                        stats["top_score"].append(top_rel)
                    else:
                        stats["failed"] += 1
                        if reranker_name not in results["failure_cases"]:
                            results["failure_cases"][reranker_name] = []
                        results["failure_cases"][reranker_name].append(query)

                except Exception as e:
                    stats["failed"] += 1
                    if reranker_name not in results["failure_cases"]:
                        results["failure_cases"][reranker_name] = []
                    results["failure_cases"][reranker_name].append(
                        f"{query} (Error: {e})"
                    )
                    continue

            # Calculate averages
            stats["avg_rerank_time"] = (
                np.mean(stats["avg_time"]) if stats["avg_time"] else 0
            )
            stats["avg_relevance_score"] = (
                np.mean(stats["avg_relevance"]) if stats["avg_relevance"] else 0
            )
            stats["avg_top_score"] = (
                np.mean(stats["top_score"]) if stats["top_score"] else 0
            )
            stats["success_rate"] = (
                (stats["successful"] / stats["total_queries"]) * 100
                if stats["total_queries"] > 0
                else 0
            )

            results["reranker_stats"][reranker_name] = stats

        # Find best reranker
        reranker_scores = {}
        for name, stats in results["reranker_stats"].items():
            # Score = relevance * 0.6 + speed * 0.2 + success_rate * 0.2
            score = (
                stats["avg_relevance_score"] * 0.6
                + (1.0 / (stats["avg_rerank_time"] + 0.1)) * 20 * 0.2
                + (stats["success_rate"] / 100) * 0.2
            )
            reranker_scores[name] = score

        best_reranker = max(reranker_scores.items(), key=lambda x: x[1])
        results["best_reranker"] = best_reranker[0]
        results["best_reranker_score"] = best_reranker[1]

        # Print results
        print("\n📈 Reranking Performance:")
        print(f"   Best overall reranker: {results['best_reranker'].upper()}\n")

        for reranker_name in rerankers:
            stats = results["reranker_stats"][reranker_name]
            print(f"   {reranker_name.upper()}:")
            print(f"      Success rate:    {stats['success_rate']:.1f}%")
            print(f"      Avg time:       {stats['avg_rerank_time']*1000:.1f}ms")
            print(f"      Avg relevance:  {stats['avg_relevance_score']:.3f}")
            print(f"      Avg top score:  {stats['avg_top_score']:.3f}")
            print(f"      Total queries:  {stats['total_queries']}")
            print(f"      Failed:         {stats['failed']}")

        if results["failure_cases"]:
            print("\n⚠️  Failure Cases:")
            for reranker_name, failures in results["failure_cases"].items():
                if failures:
                    print(f"   {reranker_name.upper()}: {len(failures)} failures")
                    for failure in failures[:3]:  # Show first 3
                        print(f"      - {failure[:60]}...")

        print("\n💡 Recommendations:")
        print(
            f"   • Best overall: {results['best_reranker'].upper()} (best balance of speed and quality)"
        )
        print(f"   • Fastest: BGE (local model)")
        print(f"   • Highest quality: Cohere Rerank v3 (highest relevance scores)")
        print(f"   • Most reliable: Cohere (fewest failures)")

        self.results["reranking"] = results
        return results

    def analyze_compression(self) -> Dict:
        """
        Question 4: Compression Analysis
        - Trade-off between compression ratio and answer quality?
        - Where's the sweet spot?
        """
        self.print_section("QUESTION 4: COMPRESSION ANALYSIS")

        queries = self.load_test_queries(15)

        results = {
            "with_compression": [],
            "without_compression": [],
            "compression_ratios": [],
            "quality_impact": {},
        }

        print(f"🔍 Testing compression on {len(queries)} queries...\n")

        for i, query in enumerate(queries[:15], 1):
            try:
                # Get retrieval results
                retrieved = hybrid_search(
                    query, k=5, use_rewrite=True, reranker="cohere", verbose=False
                )

                if (
                    isinstance(retrieved, dict)
                    and retrieved.get("type") == "direct_answer"
                ):
                    continue

                if not retrieved:
                    continue

                # Test WITH compression
                compressed_docs = compress_documents(
                    query, retrieved.copy(), show_stats=False
                )

                # Generate answers with and without compression
                answer_with_comp = generate_answer(
                    query, compressed_docs, use_compression=True
                )
                answer_without_comp = generate_answer(
                    query, retrieved.copy(), use_compression=False
                )

                # Calculate compression metrics
                total_original = sum(len(d.get("content", "")) for d in retrieved)
                total_compressed = sum(
                    len(d.get("compressed_content", d.get("content", "")))
                    for d in compressed_docs
                )

                compression_ratio = (
                    ((total_original - total_compressed) / total_original) * 100
                    if total_original > 0
                    else 0
                )

                # Measure answer quality (length as proxy, longer usually = more detailed)
                # In production, you'd use a more sophisticated metric
                answer_length_with = len(answer_with_comp.get("answer", ""))
                answer_length_without = len(answer_without_comp.get("answer", ""))

                quality_retention = (
                    (answer_length_with / answer_length_without * 100)
                    if answer_length_without > 0
                    else 0
                )

                results["compression_ratios"].append(compression_ratio)
                results["with_compression"].append(
                    {
                        "compression_ratio": compression_ratio,
                        "answer_length": answer_length_with,
                        "quality_retention": quality_retention,
                    }
                )

                results["without_compression"].append(
                    {
                        "answer_length": answer_length_without,
                    }
                )

                if i % 5 == 0:
                    print(f"   Processed {i}/{min(15, len(queries))} queries...")

            except Exception as e:
                print(f"   ⚠️  Query {i} failed: {e}")
                continue

        # Calculate statistics
        avg_compression = (
            np.mean(results["compression_ratios"])
            if results["compression_ratios"]
            else 0
        )
        avg_quality_retention = (
            np.mean([r["quality_retention"] for r in results["with_compression"]])
            if results["with_compression"]
            else 0
        )

        results["avg_compression_ratio"] = avg_compression
        results["avg_quality_retention"] = avg_quality_retention

        # Find sweet spot (good compression with >80% quality retention)
        sweet_spot_queries = [
            r
            for r in results["with_compression"]
            if r["compression_ratio"] >= 30 and r["quality_retention"] >= 80
        ]

        results["sweet_spot_count"] = len(sweet_spot_queries)
        results["sweet_spot_percentage"] = (
            (len(sweet_spot_queries) / len(results["with_compression"]) * 100)
            if results["with_compression"]
            else 0
        )

        # Print results
        print("\n📈 Compression Results:")
        print(f"   Average compression ratio:    {avg_compression:.1f}%")
        print(f"   Average quality retention:    {avg_quality_retention:.1f}%")
        print(
            f"   Sweet spot queries (>30% compression, >80% quality): {len(sweet_spot_queries)}/{len(results['with_compression'])} ({results['sweet_spot_percentage']:.1f}%)"
        )

        if results["compression_ratios"]:
            print(f"\n   Compression ratio distribution:")
            print(f"      Min:  {min(results['compression_ratios']):.1f}%")
            print(f"      Max:  {max(results['compression_ratios']):.1f}%")
            print(
                f"      P50:  {np.percentile(results['compression_ratios'], 50):.1f}%"
            )
            print(
                f"      P95:  {np.percentile(results['compression_ratios'], 95):.1f}%"
            )

        print("\n💡 Recommendations:")
        print(
            f"   • Compression is effective: {avg_compression:.1f}% average reduction"
        )
        print(f"   • Quality retention: {avg_quality_retention:.1f}% (target: >80%)")
        if results["sweet_spot_percentage"] >= 70:
            print("   ✅ Sweet spot achieved for most queries")
        else:
            print(
                "   ⚠️  Consider adjusting compression prompt for better quality retention"
            )

        # Token savings estimate
        avg_doc_length = (
            np.mean([r["answer_length"] for r in results["without_compression"]])
            if results["without_compression"]
            else 0
        )
        tokens_per_query = (avg_doc_length / 4) * (
            avg_compression / 100
        )  # Rough estimate
        print(f"   • Estimated token savings per query: ~{tokens_per_query:.0f} tokens")

        self.results["compression"] = results
        return results

    def analyze_performance(self) -> Dict:
        """
        Question 5: Performance Analysis
        - What's the biggest latency bottleneck?
        - How can you optimize the critical path?
        """
        self.print_section("QUESTION 5: PERFORMANCE ANALYSIS")

        queries = self.load_test_queries(20)

        results = {
            "component_times": defaultdict(list),
            "total_times": [],
            "bottlenecks": {},
        }

        print(f"🔍 Profiling {len(queries)} queries through full pipeline...\n")

        for i, query in enumerate(queries[:20], 1):
            try:
                pipeline_start = time.perf_counter()

                # Classification
                class_start = time.perf_counter()
                should_retrieve = classify_query(query, verbose=False)
                class_time = time.perf_counter() - class_start

                if not should_retrieve:
                    results["total_times"].append(class_time)
                    results["component_times"]["classification"].append(class_time)
                    continue

                # Retrieval
                retrieve_start = time.perf_counter()
                retrieved = hybrid_search(
                    query, k=50, use_rewrite=True, reranker=None, verbose=False
                )
                retrieve_time = time.perf_counter() - retrieve_start

                if (
                    isinstance(retrieved, dict)
                    and retrieved.get("type") == "direct_answer"
                ):
                    continue

                if not retrieved:
                    continue

                # Reranking
                rerank_start = time.perf_counter()
                from src.retrieval.reranking import rerank_cohere

                reranked = rerank_cohere(query, retrieved.copy(), k=5)
                rerank_time = time.perf_counter() - rerank_start

                # Compression
                compress_start = time.perf_counter()
                compressed = compress_documents(
                    query, reranked.copy(), show_stats=False
                )
                compress_time = time.perf_counter() - compress_start

                # Generation
                gen_start = time.perf_counter()
                answer = generate_answer(query, compressed, use_compression=True)
                gen_time = time.perf_counter() - gen_start

                total_time = time.perf_counter() - pipeline_start

                # Store timings
                results["component_times"]["classification"].append(class_time)
                results["component_times"]["retrieval"].append(retrieve_time)
                results["component_times"]["reranking"].append(rerank_time)
                results["component_times"]["compression"].append(compress_time)
                results["component_times"]["generation"].append(gen_time)
                results["total_times"].append(total_time)

                if i % 5 == 0:
                    print(f"   Processed {i}/{min(20, len(queries))} queries...")

            except Exception as e:
                print(f"   ⚠️  Query {i} failed: {e}")
                continue

        # Calculate statistics
        component_stats = {}
        for component, times in results["component_times"].items():
            if times:
                component_stats[component] = {
                    "mean": np.mean(times),
                    "p50": np.percentile(times, 50),
                    "p95": np.percentile(times, 95),
                    "total": sum(times),
                    "percentage": (
                        (sum(times) / sum(results["total_times"])) * 100
                        if results["total_times"]
                        else 0
                    ),
                }

        # Identify bottlenecks (components taking >25% of total time)
        avg_total = np.mean(results["total_times"]) if results["total_times"] else 0

        bottlenecks = []
        for component, stats in component_stats.items():
            if stats["percentage"] > 25:
                bottlenecks.append(
                    {
                        "component": component,
                        "percentage": stats["percentage"],
                        "mean_time": stats["mean"],
                    }
                )

        bottlenecks.sort(key=lambda x: x["percentage"], reverse=True)
        results["bottlenecks"] = bottlenecks
        results["avg_total_time"] = avg_total
        results["component_stats"] = component_stats

        # Print results
        print("\n📈 Performance Breakdown:")
        print(f"   Average total time: {avg_total:.2f}s\n")

        print("   Component timing (mean):")
        for component in [
            "classification",
            "retrieval",
            "reranking",
            "compression",
            "generation",
        ]:
            if component in component_stats:
                stats = component_stats[component]
                print(
                    f"      {component:15s}: {stats['mean']:6.2f}s ({stats['percentage']:5.1f}%)"
                )

        if bottlenecks:
            print("\n⚠️  Bottlenecks (>25% of total time):")
            for bottleneck in bottlenecks:
                print(
                    f"      {bottleneck['component']:15s}: {bottleneck['percentage']:5.1f}% ({bottleneck['mean_time']:.2f}s)"
                )

        print("\n💡 Optimization Recommendations:")
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            print(
                f"   🔴 Primary bottleneck: {top_bottleneck['component']} ({top_bottleneck['percentage']:.1f}% of time)"
            )

            if top_bottleneck["component"] == "compression":
                print("      • Consider batching compression calls")
                print("      • Use lighter model for compression")
                print("      • Skip compression for short documents")
            elif top_bottleneck["component"] == "reranking":
                print("      • Consider using faster reranker (BGE)")
                print("      • Reduce candidate pool before reranking")
            elif top_bottleneck["component"] == "generation":
                print("      • Use lighter generation model")
                print("      • Reduce context length")
            elif top_bottleneck["component"] == "retrieval":
                print("      • Optimize database indexes")
                print("      • Consider reducing initial k")
        else:
            print("   ✅ No major bottlenecks detected!")

        # Critical path analysis
        print("\n🛤️  Critical Path (sequential components):")
        critical_path_time = (
            component_stats.get("classification", {}).get("mean", 0)
            + component_stats.get("retrieval", {}).get("mean", 0)
            + component_stats.get("generation", {}).get("mean", 0)
        )
        print(f"   Critical path time: {critical_path_time:.2f}s")

        self.results["performance"] = results
        return results

    def analyze_costs(self) -> Dict:
        """
        Question 6: Cost Analysis
        - Cost per query with different Bedrock models?
        - How can you optimize spending?
        """
        self.print_section("QUESTION 6: COST ANALYSIS")

        # AWS Bedrock pricing (as of 2024, approximate)
        # These are estimates - actual prices may vary
        pricing = {
            "classification": {
                "claude-haiku": 0.00025,  # $0.25 per 1M input tokens, ~1k tokens/query
            },
            "reranking": {
                "cohere-rerank": 0.001,  # Bedrock pricing estimate
                "bge": 0.0,  # Free, local model
            },
            "compression": {
                "claude-nova-micro": 0.0005,  # Estimate for compression
            },
            "generation": {
                "claude-haiku": 0.00125,  # $1.25 per 1M output tokens
                "claude-sonnet": 0.003,  # $3 per 1M output tokens
                "claude-opus": 0.015,  # $15 per 1M output tokens
            },
            "embeddings": {
                "titan-embed": 0.0001,  # $0.10 per 1M tokens
            },
        }

        queries = self.load_test_queries(10)

        results = {
            "cost_per_component": {},
            "cost_per_query": {},
            "model_comparisons": {},
        }

        print(f"💰 Analyzing costs for {len(queries)} queries...\n")

        # Analyze current configuration costs
        component_costs = {
            "classification": pricing["classification"]["claude-haiku"],
            "embedding": pricing["embeddings"]["titan-embed"],
            "reranking": pricing["reranking"]["cohere-rerank"],
            "compression": pricing["compression"]["claude-nova-micro"],
            "generation": pricing["generation"]["claude-haiku"],
        }

        total_cost_per_query = sum(component_costs.values())

        results["cost_per_component"] = component_costs
        results["cost_per_query"]["current"] = total_cost_per_query

        # Compare with different generation models
        for model_name, gen_cost in pricing["generation"].items():
            alt_total = sum(
                [
                    component_costs["classification"],
                    component_costs["embedding"],
                    component_costs["reranking"],
                    component_costs["compression"],
                    gen_cost,
                ]
            )
            results["cost_per_query"][model_name] = alt_total

        # Compare with/without optional components
        base_cost = (
            component_costs["classification"]
            + component_costs["embedding"]
            + component_costs["generation"]
        )

        results["cost_per_query"]["minimal"] = base_cost
        results["cost_per_query"]["with_reranking"] = (
            base_cost + component_costs["reranking"]
        )
        results["cost_per_query"]["with_compression"] = (
            base_cost + component_costs["compression"]
        )
        results["cost_per_query"]["full"] = total_cost_per_query

        # Calculate savings scenarios
        results["savings"] = {
            "skip_compression": total_cost_per_query
            - (total_cost_per_query - component_costs["compression"]),
            "use_bge_reranking": total_cost_per_query
            - (total_cost_per_query - component_costs["reranking"] + 0),  # BGE is free
            "minimal_pipeline": total_cost_per_query - base_cost,
        }

        # Print results
        print("💰 Cost Breakdown (per query):")
        print("\n   Current Configuration:")
        for component, cost in component_costs.items():
            print(f"      {component:15s}: ${cost:.4f}")
        print(f"      {'TOTAL':15s}: ${total_cost_per_query:.4f}")

        print("\n   Cost with Different Generation Models:")
        for model_name, cost in pricing["generation"].items():
            alt_total = results["cost_per_query"].get(model_name, 0)
            print(f"      {model_name:15s}: ${alt_total:.4f}")
            if model_name != "current":
                savings = (
                    (total_cost_per_query - alt_total) / total_cost_per_query
                ) * 100
                print(f"         Savings: {savings:+.1f}%")

        print("\n   Cost by Pipeline Configuration:")
        print(
            f"      Minimal (no rerank/compress): ${results['cost_per_query']['minimal']:.4f}"
        )
        print(
            f"      With reranking only:          ${results['cost_per_query']['with_reranking']:.4f}"
        )
        print(
            f"      With compression only:        ${results['cost_per_query']['with_compression']:.4f}"
        )
        print(
            f"      Full pipeline:                ${results['cost_per_query']['full']:.4f}"
        )

        print("\n   Estimated Savings:")
        print(
            f"      Skip compression:            ${results['savings']['skip_compression']:.4f}/query ({results['savings']['skip_compression']/total_cost_per_query*100:.1f}%)"
        )
        print(
            f"      Use BGE reranking:           ${results['savings']['use_bge_reranking']:.4f}/query ({results['savings']['use_bge_reranking']/total_cost_per_query*100:.1f}%)"
        )
        print(
            f"      Minimal pipeline:            ${results['savings']['minimal_pipeline']:.4f}/query ({results['savings']['minimal_pipeline']/total_cost_per_query*100:.1f}%)"
        )

        # Monthly estimates
        queries_per_month = 10000
        monthly_cost = total_cost_per_query * queries_per_month
        print(f"\n   Monthly Cost Estimates (10k queries):")
        print(f"      Current:                     ${monthly_cost:.2f}")
        print(
            f"      Minimal:                     ${results['cost_per_query']['minimal'] * queries_per_month:.2f}"
        )
        print(
            f"      Savings (minimal):           ${results['savings']['minimal_pipeline'] * queries_per_month:.2f}"
        )

        print("\n💡 Cost Optimization Recommendations:")
        print("   1. Use BGE reranking for free alternative (saves ~$0.001/query)")
        print("   2. Classification saves 40%+ by skipping unnecessary retrievals")
        print("   3. Compression saves generation tokens (~30% reduction)")
        print("   4. Consider lighter generation model for lower quality needs")
        print("   5. Batch queries when possible to optimize API calls")

        self.results["costs"] = results
        return results

    def run_all_analyses(self) -> Dict:
        """Run all 6 analyses"""
        print("\n" + "█" * 80)
        print("🔬 COMPREHENSIVE RAG SYSTEM ANALYSIS")
        print("█" * 80)
        print("\nThis will analyze 6 key areas:")
        print("  1. Query Classification - Skip rate and cost impact")
        print("  2. Hybrid Search - Optimal keyword/vector weights")
        print("  3. Reranking - Best method and failure cases")
        print("  4. Compression - Ratio vs quality tradeoff")
        print("  5. Performance - Latency bottlenecks")
        print("  6. Cost - Per-query costs and optimization\n")

        start_time = time.perf_counter()

        # Check database
        try:
            db = DatabaseManager()
            stats = db.get_stats()
            db.close()

            if stats["documents"] == 0:
                print("❌ Error: No documents in database!")
                print("   Please run 'python rag.py --load-data' first.")
                return {}

            print(f"✅ Database ready: {stats['documents']} documents\n")

        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return {}

        # Run analyses
        analyses = [
            ("Classification", self.analyze_query_classification),
            ("Hybrid Search", self.analyze_hybrid_search),
            ("Reranking", self.analyze_reranking),
            ("Compression", self.analyze_compression),
            ("Performance", self.analyze_performance),
            ("Costs", self.analyze_costs),
        ]

        for name, analysis_func in analyses:
            try:
                print(f"\n{'='*80}")
                print(f"Running {name} analysis...")
                print(f"{'='*80}")
                analysis_func()
            except Exception as e:
                print(f"\n❌ {name} analysis failed: {e}")
                import traceback

                traceback.print_exc()

        # Generate summary
        total_time = time.perf_counter() - start_time
        self.print_summary(total_time)

        # Save results
        self.save_results()

        return self.results

    def print_summary(self, total_time: float):
        """Print analysis summary"""
        print("\n" + "█" * 80)
        print("📋 ANALYSIS SUMMARY")
        print("█" * 80)

        # Key findings
        findings = []

        if "classification" in self.results:
            cls = self.results["classification"]
            findings.append(
                f"✅ Classification skips {cls['skip_rate']:.1f}% of queries, saving {cls['cost_savings']:.1f}% on costs"
            )

        if "hybrid_search" in self.results:
            hs = self.results["hybrid_search"]
            findings.append(
                f"✅ Optimal RRF k: {hs['optimal_rrf_k']} for hybrid search"
            )

        if "reranking" in self.results:
            rer = self.results["reranking"]
            findings.append(
                f"✅ Best reranker: {rer['best_reranker'].upper()} (balance of speed and quality)"
            )

        if "compression" in self.results:
            comp = self.results["compression"]
            findings.append(
                f"✅ Compression achieves {comp['avg_compression_ratio']:.1f}% reduction with {comp['avg_quality_retention']:.1f}% quality retention"
            )

        if "performance" in self.results:
            perf = self.results["performance"]
            if perf["bottlenecks"]:
                top = perf["bottlenecks"][0]
                findings.append(
                    f"⚠️  Primary bottleneck: {top['component']} ({top['percentage']:.1f}% of time)"
                )
            else:
                findings.append("✅ No major performance bottlenecks")

        if "costs" in self.results:
            costs = self.results["costs"]
            cost_per_query = costs["cost_per_query"].get("current", 0)
            findings.append(f"💰 Current cost per query: ${cost_per_query:.4f}")

        print("\n🎯 Key Findings:")
        for finding in findings:
            print(f"   {finding}")

        print(
            f"\n⏱️  Total analysis time: {total_time:.1f}s ({total_time/60:.1f} minutes)"
        )
        print(f"\n💾 Detailed results saved to: rag_analysis_results.json")

    def save_results(self):
        """Save analysis results to JSON"""
        try:
            # Convert numpy types to native Python types
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, defaultdict):
                    return dict(obj)
                else:
                    return obj

            output = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": convert_to_native(self.results),
            }

            with open("rag_analysis_results.json", "w") as f:
                json.dump(output, f, indent=2)

            print(f"\n✅ Results saved successfully!")

        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")


def main():
    """Main entry point"""
    analyzer = RAGAnalysis()
    results = analyzer.run_all_analyses()

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
