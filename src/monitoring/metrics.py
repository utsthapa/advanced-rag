"""Metrics collection and monitoring for RAG system"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp"""

    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collection for monitoring system performance.

    Tracks:
    - Latency metrics (p50, p95, p99, mean)
    - Throughput (QPS, request counts)
    - Error rates
    - Service health (circuit breaker states, success/failure rates)
    - Cost metrics (tokens, API calls)
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent metrics to keep in rolling window
        """
        self.window_size = window_size
        self.lock = Lock()

        # Latency metrics (in milliseconds)
        self.retrieval_latencies = deque(maxlen=window_size)
        self.reranking_latencies = deque(maxlen=window_size)
        self.compression_latencies = deque(maxlen=window_size)
        self.generation_latencies = deque(maxlen=window_size)
        self.total_latencies = deque(maxlen=window_size)

        # Throughput metrics
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()

        # Error tracking
        self.errors_by_type: Dict[str, int] = {}
        self.errors_by_component: Dict[str, int] = {}

        # Service health
        self.circuit_breaker_states: Dict[str, str] = {}
        self.service_success_rates: Dict[str, float] = {}
        self.service_failure_rates: Dict[str, float] = {}

        # Cost metrics
        self.bedrock_api_calls = 0
        self.tokens_processed = 0
        self.compression_tokens_saved = 0

        # Query classification metrics
        self.classified_queries = 0
        self.skipped_queries = 0

    def record_latency(
        self,
        component: str,
        latency_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Record latency for a component"""
        with self.lock:
            metric_value = MetricValue(
                value=latency_ms, timestamp=time.time(), tags=tags or {}
            )

            if component == "retrieval":
                self.retrieval_latencies.append(metric_value)
            elif component == "reranking":
                self.reranking_latencies.append(metric_value)
            elif component == "compression":
                self.compression_latencies.append(metric_value)
            elif component == "generation":
                self.generation_latencies.append(metric_value)
            elif component == "total":
                self.total_latencies.append(metric_value)

    def record_request(self, success: bool):
        """Record a request (successful or failed)"""
        with self.lock:
            self.request_count += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    def record_error(self, error_type: str, component: str):
        """Record an error"""
        with self.lock:
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
            self.errors_by_component[component] = (
                self.errors_by_component.get(component, 0) + 1
            )

    def record_circuit_breaker_state(self, service: str, state: str):
        """Record circuit breaker state for a service"""
        with self.lock:
            self.circuit_breaker_states[service] = state

    def record_service_metrics(self, service: str, success: bool):
        """Record service-level success/failure"""
        with self.lock:
            if service not in self.service_success_rates:
                self.service_success_rates[service] = 0.0
                self.service_failure_rates[service] = 0.0

            # Simple moving average (last 100 requests)
            if success:
                self.service_success_rates[service] = (
                    self.service_success_rates[service] * 0.99 + 0.01
                )
                self.service_failure_rates[service] = (
                    self.service_failure_rates[service] * 0.99
                )
            else:
                self.service_success_rates[service] = (
                    self.service_success_rates[service] * 0.99
                )
                self.service_failure_rates[service] = (
                    self.service_failure_rates[service] * 0.99 + 0.01
                )

    def record_cost_metrics(
        self, api_calls: int = 0, tokens: int = 0, tokens_saved: int = 0
    ):
        """Record cost-related metrics"""
        with self.lock:
            self.bedrock_api_calls += api_calls
            self.tokens_processed += tokens
            self.compression_tokens_saved += tokens_saved

    def record_classification(self, skipped: bool):
        """Record query classification decision"""
        with self.lock:
            self.classified_queries += 1
            if skipped:
                self.skipped_queries += 1

    def get_latency_stats(self, component: str) -> Dict[str, float]:
        """Get latency statistics for a component"""
        with self.lock:
            if component == "retrieval":
                latencies = self.retrieval_latencies
            elif component == "reranking":
                latencies = self.reranking_latencies
            elif component == "compression":
                latencies = self.compression_latencies
            elif component == "generation":
                latencies = self.generation_latencies
            elif component == "total":
                latencies = self.total_latencies
            else:
                return {}

            if not latencies:
                return {}

            values = [m.value for m in latencies]
            sorted_values = sorted(values)

            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "p50": sorted_values[len(sorted_values) // 2],
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)],
                "min": min(values),
                "max": max(values),
            }

    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics"""
        with self.lock:
            elapsed = time.time() - self.start_time
            qps = self.request_count / elapsed if elapsed > 0 else 0

            return {
                "total_requests": self.request_count,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.request_count
                    if self.request_count > 0
                    else 0
                ),
                "qps": qps,
                "uptime_seconds": elapsed,
            }

    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        with self.lock:
            return {
                "by_type": dict(self.errors_by_type),
                "by_component": dict(self.errors_by_component),
                "total_errors": sum(self.errors_by_type.values()),
            }

    def get_service_health(self) -> Dict:
        """Get service health metrics"""
        with self.lock:
            return {
                "circuit_breaker_states": dict(self.circuit_breaker_states),
                "success_rates": dict(self.service_success_rates),
                "failure_rates": dict(self.service_failure_rates),
            }

    def get_cost_stats(self) -> Dict:
        """Get cost-related statistics"""
        with self.lock:
            return {
                "bedrock_api_calls": self.bedrock_api_calls,
                "tokens_processed": self.tokens_processed,
                "tokens_saved_from_compression": self.compression_tokens_saved,
                "compression_efficiency": (
                    self.compression_tokens_saved / self.tokens_processed * 100
                    if self.tokens_processed > 0
                    else 0
                ),
            }

    def get_classification_stats(self) -> Dict:
        """Get query classification statistics"""
        with self.lock:
            return {
                "total_classified": self.classified_queries,
                "skipped_queries": self.skipped_queries,
                "skip_rate": (
                    self.skipped_queries / self.classified_queries * 100
                    if self.classified_queries > 0
                    else 0
                ),
            }

    def get_all_metrics(self) -> Dict:
        """Get all metrics as a dictionary"""
        return {
            "latency": {
                "retrieval": self.get_latency_stats("retrieval"),
                "reranking": self.get_latency_stats("reranking"),
                "compression": self.get_latency_stats("compression"),
                "generation": self.get_latency_stats("generation"),
                "total": self.get_latency_stats("total"),
            },
            "throughput": self.get_throughput_stats(),
            "errors": self.get_error_stats(),
            "service_health": self.get_service_health(),
            "cost": self.get_cost_stats(),
            "classification": self.get_classification_stats(),
        }

    def reset(self):
        """Reset all metrics (useful for testing)"""
        with self.lock:
            self.retrieval_latencies.clear()
            self.reranking_latencies.clear()
            self.compression_latencies.clear()
            self.generation_latencies.clear()
            self.total_latencies.clear()
            self.request_count = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.start_time = time.time()
            self.errors_by_type.clear()
            self.errors_by_component.clear()
            self.circuit_breaker_states.clear()
            self.service_success_rates.clear()
            self.service_failure_rates.clear()
            self.bedrock_api_calls = 0
            self.tokens_processed = 0
            self.compression_tokens_saved = 0
            self.classified_queries = 0
            self.skipped_queries = 0


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
