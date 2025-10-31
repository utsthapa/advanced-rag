"""Load testing for RAG system - tests 50+ QPS capacity"""

import asyncio
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from src.config import setup_logging
from src.monitoring import get_metrics_collector
from src.pipeline import rag_pipeline

setup_logging()
logger = logging.getLogger(__name__)


class LoadTestResults:
    """Results from load testing"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies_ms: List[float] = []
        self.errors: List[Dict] = []
        self.start_time = 0
        self.end_time = 0
        self.target_qps = 0
        self.actual_qps = 0

    def calculate_stats(self):
        """Calculate statistics from results"""
        if not self.latencies_ms:
            return {}

        sorted_latencies = sorted(self.latencies_ms)
        n = len(sorted_latencies)

        return {
            "mean": statistics.mean(self.latencies_ms),
            "median": sorted_latencies[n // 2],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "min": min(self.latencies_ms),
            "max": max(self.latencies_ms),
        }


def run_load_test(
    queries: List[str],
    target_qps: int = 50,
    duration_seconds: int = 60,
    max_workers: int = 10,
) -> LoadTestResults:
    """
    Run load test with specified QPS target

    Args:
        queries: List of test queries to use
        target_qps: Target queries per second
        duration_seconds: How long to run the test
        max_workers: Maximum concurrent workers

    Returns:
        LoadTestResults with statistics
    """
    print(f"\n{'='*80}")
    print(f"🚀 LOAD TEST")
    print(f"{'='*80}")
    print(f"Target QPS: {target_qps}")
    print(f"Duration: {duration_seconds}s")
    print(f"Max concurrent workers: {max_workers}")
    print(f"{'='*80}\n")

    results = LoadTestResults()
    results.start_time = time.time()
    results.target_qps = target_qps

    # Query pool - cycle through queries
    query_pool = queries * (target_qps * duration_seconds // len(queries) + 1)

    def process_query(query: str) -> Dict:
        """Process a single query"""
        start = time.perf_counter()
        try:
            # Use minimal pipeline for load testing (focus on retrieval)
            result = rag_pipeline(
                query=query,
                k=5,
                use_rewrite=False,  # Skip rewriting for speed
                reranker="cohere",  # Use fastest reranker
                use_compression=False,  # Skip compression for speed
                verbose=False,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "latency_ms": latency_ms,
                "query": query,
                "num_sources": len(result.get("sources", [])),
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Query failed: {e}")
            return {
                "success": False,
                "latency_ms": latency_ms,
                "query": query,
                "error": str(e),
            }

    # Calculate interval between requests
    request_interval = 1.0 / target_qps

    # Use thread pool for concurrent execution
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    query_index = 0
    start_time = time.time()
    next_request_time = start_time

    print(f"Starting load test at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    print(f"Request interval: {request_interval*1000:.2f}ms\n")

    iteration = 0
    while time.time() - start_time < duration_seconds:
        current_time = time.time()

        # Submit requests at target QPS
        if current_time >= next_request_time:
            query = query_pool[query_index % len(query_pool)]
            future = executor.submit(process_query, query)
            futures.append(future)

            query_index += 1
            next_request_time = start_time + (query_index * request_interval)

            # Process completed futures periodically
            if len(futures) >= max_workers:
                # Check for completed futures
                completed = [f for f in futures if f.done()]
                for future in completed:
                    futures.remove(future)
                    try:
                        result = future.result(timeout=0.1)
                        results.total_requests += 1

                        if result["success"]:
                            results.successful_requests += 1
                            results.latencies_ms.append(result["latency_ms"])
                        else:
                            results.failed_requests += 1
                            results.errors.append(result)

                    except Exception as e:
                        results.failed_requests += 1
                        results.errors.append({"error": str(e)})

        # Small sleep to prevent CPU spinning
        time.sleep(0.001)

        iteration += 1
        if iteration % 1000 == 0:
            elapsed = time.time() - start_time
            current_qps = results.total_requests / elapsed if elapsed > 0 else 0
            print(
                f"  Progress: {elapsed:.1f}s, {results.total_requests} requests, "
                f"{current_qps:.1f} QPS, {results.failed_requests} failures"
            )

    # Wait for remaining futures
    print("\nWaiting for remaining requests to complete...")
    for future in futures:
        try:
            result = future.result(timeout=30)
            results.total_requests += 1

            if result["success"]:
                results.successful_requests += 1
                results.latencies_ms.append(result["latency_ms"])
            else:
                results.failed_requests += 1
                results.errors.append(result)
        except Exception as e:
            results.failed_requests += 1
            results.errors.append({"error": str(e)})

    executor.shutdown(wait=True)
    results.end_time = time.time()

    # Calculate final QPS
    total_time = results.end_time - results.start_time
    results.actual_qps = results.total_requests / total_time if total_time > 0 else 0

    return results


def run_load_test_suite(
    target_qps: int = 50,
    duration_seconds: int = 60,
    max_workers: int = 10,
) -> Dict:
    """
    Run comprehensive load test suite

    Args:
        target_qps: Target queries per second (default: 50)
        duration_seconds: Test duration (default: 60s)
        max_workers: Max concurrent workers

    Returns:
        Dict with test results and pass/fail status
    """
    # Test queries - mix of scientific and general
    test_queries = [
        "How does CRISPR gene editing work?",
        "What causes COVID-19 transmission?",
        "Explain the mechanism of mRNA vaccines",
        "What is the role of ACE2 in viral infection?",
        "How do antibodies neutralize viruses?",
        "What are the effects of climate change?",
        "How does photosynthesis work?",
        "What is the structure of DNA?",
        "Explain protein synthesis",
        "How do neurons transmit signals?",
    ]

    print(f"\n{'█'*80}")
    print(f"🔥 LOAD TEST SUITE - {target_qps} QPS Target")
    print(f"{'█'*80}\n")

    # Run load test
    results = run_load_test(
        queries=test_queries,
        target_qps=target_qps,
        duration_seconds=duration_seconds,
        max_workers=max_workers,
    )

    # Calculate statistics
    stats = results.calculate_stats()
    total_time = results.end_time - results.start_time

    # Print results
    print(f"\n{'='*80}")
    print(f"📊 LOAD TEST RESULTS")
    print(f"{'='*80}\n")

    print(f"Duration:           {total_time:.1f}s")
    print(f"Total Requests:     {results.total_requests}")
    print(f"Successful:         {results.successful_requests}")
    print(f"Failed:             {results.failed_requests}")
    print(
        f"Success Rate:      {(results.successful_requests/results.total_requests*100):.1f}%"
    )
    print(f"\nTarget QPS:        {target_qps}")
    print(f"Actual QPS:         {results.actual_qps:.1f}")
    print(f"QPS Achievement:   {(results.actual_qps/target_qps*100):.1f}%")

    if stats:
        print(f"\n📈 LATENCY STATISTICS (ms):")
        print(f"  Mean:    {stats['mean']:.1f}ms")
        print(f"  Median:  {stats['median']:.1f}ms")
        print(f"  P95:     {stats['p95']:.1f}ms")
        print(f"  P99:     {stats['p99']:.1f}ms")
        print(f"  Min:     {stats['min']:.1f}ms")
        print(f"  Max:     {stats['max']:.1f}ms")

    if results.errors:
        print(f"\n⚠️  ERRORS ({len(results.errors)}):")
        error_types = {}
        for error in results.errors[:10]:  # Show first 10
            err_msg = error.get("error", "Unknown error")
            error_types[err_msg] = error_types.get(err_msg, 0) + 1
        for err_type, count in error_types.items():
            print(f"  {err_type}: {count}")

    # Determine pass/fail
    passed = True
    failures = []

    # Check QPS target
    if results.actual_qps < target_qps * 0.95:  # Allow 5% margin
        passed = False
        failures.append(f"QPS ({results.actual_qps:.1f}) below target ({target_qps})")

    # Check success rate (should be > 99%)
    success_rate = (
        results.successful_requests / results.total_requests * 100
        if results.total_requests > 0
        else 0
    )
    if success_rate < 99:
        passed = False
        failures.append(f"Success rate ({success_rate:.1f}%) below target (99%)")

    # Check retrieval latency p95 (should be < 100ms)
    if stats and stats.get("p95", 0) > 100:
        passed = False
        failures.append(
            f"Retrieval latency p95 ({stats['p95']:.1f}ms) exceeds target (100ms)"
        )

    print(f"\n{'='*80}")
    if passed:
        print(f"✅ LOAD TEST PASSED")
        print(f"{'='*80}")
        print(f"\n✅ System successfully handles {target_qps}+ QPS:")
        print(f"   • Actual QPS: {results.actual_qps:.1f}")
        print(f"   • Success rate: {success_rate:.1f}%")
        if stats:
            print(f"   • Retrieval latency p95: {stats['p95']:.1f}ms")
    else:
        print(f"❌ LOAD TEST FAILED")
        print(f"{'='*80}")
        print(f"\n❌ System did not meet load test requirements:")
        for failure in failures:
            print(f"   • {failure}")

    print(f"{'='*80}\n")

    return {
        "passed": passed,
        "target_qps": target_qps,
        "actual_qps": results.actual_qps,
        "total_requests": results.total_requests,
        "successful_requests": results.successful_requests,
        "failed_requests": results.failed_requests,
        "success_rate": success_rate,
        "latency_stats": stats,
        "failures": failures,
        "duration": total_time,
    }


if __name__ == "__main__":
    # Run load test
    results = run_load_test_suite(target_qps=50, duration_seconds=60, max_workers=10)
