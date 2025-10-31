"""Circuit breaker pattern for graceful degradation when Bedrock services are unavailable"""

import logging
import time
from enum import Enum
from threading import Lock
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures when external services are unavailable.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests fail fast without calling service
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes needed in half-open to close circuit
            timeout: Seconds to wait before attempting half-open
            expected_exception: Exception type that indicates failure
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.lock = Lock()

    def call(
        self, func: Callable[[], T], fallback: Optional[Callable[[], T]] = None
    ) -> T:
        """
        Call a function with circuit breaker protection.

        Args:
            func: Function to call
            fallback: Optional fallback function if circuit is open or call fails

        Returns:
            Result of func or fallback
        """
        with self.lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if time.time() - (self.last_failure_time or 0) >= self.timeout:
                    logger.info("Circuit breaker: transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    # Circuit is open, use fallback
                    if fallback:
                        logger.warning(
                            f"Circuit breaker OPEN: using fallback (failed {self.failure_count} times)"
                        )
                        return fallback()
                    raise Exception(
                        f"Circuit breaker is OPEN (service unavailable, {self.failure_count} failures)"
                    )

        # Try to call the function
        try:
            result = func()
            self._record_success()
            return result

        except self.expected_exception as e:
            self._record_failure()
            logger.warning(f"Circuit breaker: recorded failure: {e}")

            # Use fallback if available
            if fallback:
                logger.info("Using fallback function after failure")
                return fallback()

            raise

    def _record_success(self):
        """Record a successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker: transitioning to CLOSED (recovered)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _record_failure(self):
        """Record a failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Failed again in half-open, go back to open
                logger.warning("Circuit breaker: transitioning back to OPEN")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit breaker: transitioning to OPEN (threshold: {self.failure_threshold})"
                    )
                    self.state = CircuitState.OPEN

    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker: manually reset to CLOSED")

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == CircuitState.OPEN


# Global circuit breakers for different services
_bedrock_embeddings_cb = None
_bedrock_classifier_cb = None
_bedrock_rewriter_cb = None
_bedrock_reranker_cb = None
_bedrock_generator_cb = None


def get_bedrock_embeddings_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for embeddings service"""
    global _bedrock_embeddings_cb
    if _bedrock_embeddings_cb is None:
        _bedrock_embeddings_cb = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )
    return _bedrock_embeddings_cb


def get_bedrock_classifier_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for classifier service"""
    global _bedrock_classifier_cb
    if _bedrock_classifier_cb is None:
        _bedrock_classifier_cb = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )
    return _bedrock_classifier_cb


def get_bedrock_rewriter_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for rewriter service"""
    global _bedrock_rewriter_cb
    if _bedrock_rewriter_cb is None:
        _bedrock_rewriter_cb = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )
    return _bedrock_rewriter_cb


def get_bedrock_reranker_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for reranker service"""
    global _bedrock_reranker_cb
    if _bedrock_reranker_cb is None:
        _bedrock_reranker_cb = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )
    return _bedrock_reranker_cb


def get_bedrock_generator_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for answer generator service"""
    global _bedrock_generator_cb
    if _bedrock_generator_cb is None:
        _bedrock_generator_cb = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout=60.0,
        )
    return _bedrock_generator_cb
