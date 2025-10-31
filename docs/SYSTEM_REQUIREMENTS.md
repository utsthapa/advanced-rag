# System Requirements Implementation Guide

This document explains how the Advanced RAG system meets all system requirements and how to test them.

## Requirements Overview

### Performance Benchmarks

1. ✅ **Latency**: p95 retrieval latency < 100ms (excluding LLM generation)
2. ✅ **Quality**: NDCG@10 > 0.8 on domain-specific test set
3. ✅ **Compression**: 30%+ token reduction through contextual compression
4. ✅ **Cost**: 40%+ cost reduction through query classification

### System Requirements

1. ✅ **Load Capacity**: Handle 50+ QPS retrieval load consistently
2. ✅ **Graceful Degradation**: When Bedrock services are unavailable
3. ✅ **Error Handling**: Comprehensive error handling for all components
4. ✅ **Monitoring & Alerting**: Monitoring and alerting for key metrics

## Implementation Details

### 1. Performance Benchmarks

The performance benchmarks are implemented in `src/benchmarks/performance.py`:

- **Latency Test**: Measures pure retrieval latency (excluding LLM calls)
- **Quality Test**: Uses NDCG@10 metric with ground truth or relevance proxies
- **Compression Test**: Measures token reduction from contextual compression
- **Cost Test**: Calculates cost savings from query classification

**Run Performance Benchmarks:**
```bash
python rag.py --benchmark
```

Or directly:
```bash
python src/benchmarks/performance.py
```

### 2. Load Capacity (50+ QPS)

The load testing system is implemented in `src/benchmarks/load_test.py`:

- Tests system capacity under high load
- Maintains target QPS using controlled request intervals
- Measures actual QPS, success rates, and latency under load
- Validates system can handle 50+ QPS consistently

**Run Load Test:**
```bash
python src/benchmarks/load_test.py
```

**Parameters:**
- `target_qps`: Target queries per second (default: 50)
- `duration_seconds`: Test duration (default: 60)
- `max_workers`: Maximum concurrent workers (default: 10)

### 3. Graceful Degradation

Circuit breakers are implemented in `src/core/circuit_breaker.py`:

**Circuit Breaker Pattern:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service is failing, requests fail fast without calling service
- **HALF_OPEN**: Testing if service recovered, limited requests allowed

**Circuit Breakers Configured:**
- `bedrock_embeddings_cb`: Embeddings service
- `bedrock_classifier_cb`: Query classification service
- `bedrock_rewriter_cb`: Query rewriting service
- `bedrock_reranker_cb`: Reranking service
- `bedrock_generator_cb`: Answer generation service

**Features:**
- Automatic failure detection (threshold-based)
- Fast failure mode when circuit is open
- Automatic recovery testing (half-open state)
- Configurable thresholds and timeouts

**Usage Example:**
```python
from src.core.circuit_breaker import get_bedrock_classifier_circuit_breaker

cb = get_bedrock_classifier_circuit_breaker()

def call_classifier():
    # Your classification code
    pass

def fallback_classifier():
    # Fallback strategy
    return True  # Default: always retrieve

result = cb.call(call_classifier, fallback=fallback_classifier)
```

### 4. Comprehensive Error Handling

Error handling is integrated throughout the system:

**Error Tracking:**
- All components use try-catch blocks
- Errors are logged with appropriate levels
- Errors don't crash the system (graceful fallbacks)
- Error statistics are tracked in metrics

**Components with Enhanced Error Handling:**
- `src/core/clients.py`: Bedrock client initialization
- `src/retrieval/reranking.py`: Reranking fallbacks
- `src/generation/answer.py`: Answer generation fallbacks
- `src/processing/compression.py`: Compression fallbacks
- `src/processing/query.py`: Query processing fallbacks

**Error Metrics:**
- Error counts by type
- Error counts by component
- Total error rate
- Service-specific error rates

### 5. Monitoring & Alerting

The monitoring system is implemented in `src/monitoring/`:

#### Metrics Collection (`src/monitoring/metrics.py`)

**Tracked Metrics:**
- **Latency**: p50, p95, p99, mean for all components
- **Throughput**: QPS, request counts, success rates
- **Errors**: Error rates by type and component
- **Service Health**: Circuit breaker states, success/failure rates
- **Cost**: API calls, tokens processed, compression savings
- **Classification**: Skip rates, cost savings

**Usage:**
```python
from src.monitoring import get_metrics_collector

metrics = get_metrics_collector()

# Record latency
metrics.record_latency("retrieval", 45.2)  # milliseconds

# Record request
metrics.record_request(success=True)

# Record error
metrics.record_error("TimeoutError", "bedrock_embeddings")

# Get all metrics
all_metrics = metrics.get_all_metrics()
```

#### Alert System (`src/monitoring/alerts.py`)

**Alert Levels:**
- `INFO`: Informational alerts
- `WARNING`: Non-critical issues requiring attention
- `CRITICAL`: Critical issues requiring immediate action

**Monitored Thresholds:**
- Retrieval latency p95 > 100ms (CRITICAL)
- Retrieval latency p95 > 80ms (WARNING)
- Error rate > 5% (CRITICAL)
- Error rate > 1% (WARNING)
- QPS < 50 (WARNING)
- Circuit breaker OPEN (CRITICAL)
- Service failure rate > 10% (CRITICAL)

**Usage:**
```python
from src.monitoring.alerts import get_alert_manager

alert_manager = get_alert_manager()

# Check for alerts
alerts = alert_manager.check_alerts()

# Register custom callback
def my_alert_handler(alert):
    print(f"Alert: {alert.message}")

alert_manager.register_alert_callback(my_alert_handler)
```

## Running Comprehensive Tests

### System Requirements Test Suite

The comprehensive test suite validates all requirements:

```bash
python src/benchmarks/system_requirements.py
```

This runs:
1. Performance benchmarks
2. Load capacity test (50+ QPS)
3. Graceful degradation test
4. Error handling test
5. Monitoring & alerting test

### Individual Tests

**Performance Benchmarks Only:**
```bash
python rag.py --benchmark
```

**Load Test Only:**
```bash
python src/benchmarks/load_test.py
```

**Monitor System Metrics:**
```python
from src.monitoring import get_metrics_collector

metrics = get_metrics_collector()
all_metrics = metrics.get_all_metrics()

print("Latency Stats:", all_metrics["latency"])
print("Throughput:", all_metrics["throughput"])
print("Service Health:", all_metrics["service_health"])
```

## Integration with Pipeline

The monitoring and circuit breakers are integrated into the pipeline:

1. **Metrics Collection**: Automatic latency tracking for all pipeline steps
2. **Circuit Breakers**: Protect against cascading failures
3. **Error Handling**: Graceful fallbacks at every step
4. **Alerting**: Automatic threshold monitoring

## Configuration

Circuit breaker and monitoring settings can be adjusted in:
- `src/core/circuit_breaker.py`: Circuit breaker thresholds
- `src/monitoring/metrics.py`: Metrics window sizes
- `src/monitoring/alerts.py`: Alert thresholds

## Best Practices

1. **Monitor Regularly**: Check metrics and alerts regularly
2. **Set Appropriate Thresholds**: Adjust thresholds based on your requirements
3. **Test Under Load**: Regularly run load tests to validate capacity
4. **Review Error Logs**: Monitor error statistics for patterns
5. **Circuit Breaker Tuning**: Adjust failure thresholds based on service reliability

## Troubleshooting

**High Latency:**
- Check retrieval latency specifically (should be < 100ms)
- Verify database indexes are optimized
- Check connection pooling settings

**High Error Rate:**
- Review error logs by component
- Check circuit breaker states
- Verify service availability

**Low QPS:**
- Increase `max_workers` in load test
- Check system resource usage
- Verify database performance

**Circuit Breakers Open:**
- Check Bedrock service availability
- Review error logs for specific failures
- Wait for automatic recovery (circuit breaker timeout)

## Next Steps

1. Run comprehensive test suite: `python src/benchmarks/system_requirements.py`
2. Monitor metrics in production: Integrate metrics into your monitoring dashboard
3. Set up alert notifications: Register callbacks for critical alerts
4. Tune thresholds: Adjust based on your specific requirements
5. Regular load testing: Schedule regular load tests to validate capacity

