# System Requirements Testing Guide

This guide explains how to test all system requirements for the Advanced RAG system.

## Quick Start

Run the comprehensive system requirements test:

```bash
python src/benchmarks/system_requirements.py
```

Or use the integrated CLI:

```bash
python rag.py --benchmark
# Then select option 3: System requirements (comprehensive)
```

## Test Components

### 1. Performance Benchmarks

**What it tests:**
- ✅ Latency: p95 retrieval latency < 100ms (excluding LLM generation)
- ✅ Quality: NDCG@10 > 0.8 on domain-specific test set
- ✅ Compression: 30%+ token reduction through contextual compression
- ✅ Cost: 40%+ cost reduction through query classification

**Run individually:**
```bash
python src/benchmarks/performance.py
```

**What to expect:**
- Each benchmark reports pass/fail status
- Detailed metrics for each requirement
- Results saved to `benchmark_results.json`

### 2. Load Capacity Test

**What it tests:**
- ✅ Handle 50+ QPS retrieval load consistently
- ✅ Success rate ≥ 99%
- ✅ Retrieval latency p95 < 100ms under load

**Run individually:**
```bash
python src/benchmarks/load_test.py
```

**Parameters:**
- `target_qps`: Target queries per second (default: 50)
- `duration_seconds`: Test duration (default: 60)
- `max_workers`: Maximum concurrent workers (default: 10)

**What to expect:**
- Real-time QPS reporting during test
- Final statistics: actual QPS, success rate, latency distribution
- Pass/fail based on targets

**Example output:**
```
🚀 LOAD TEST
Target QPS: 50
Duration: 60s
Max concurrent workers: 10

Actual QPS: 52.3
Success Rate: 99.8%
Latency p95: 85.2ms

✅ LOAD TEST PASSED
```

### 3. Graceful Degradation Test

**What it tests:**
- ✅ Circuit breakers configured for all Bedrock services
- ✅ Circuit breakers transition states correctly
- ✅ Fallback mechanisms available

**What it checks:**
- Circuit breaker instances exist
- Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
- Automatic failure detection and recovery

**How it works:**
Circuit breakers protect against cascading failures:
- After 5 failures → Circuit opens (fails fast)
- After 60s timeout → Circuit goes half-open (test recovery)
- After 2 successes → Circuit closes (recovered)

**Monitor circuit breaker states:**
```python
from src.core.circuit_breaker import (
    get_bedrock_embeddings_circuit_breaker,
    get_bedrock_classifier_circuit_breaker,
)

cb = get_bedrock_embeddings_circuit_breaker()
print(f"State: {cb.get_state().value}")
```

### 4. Error Handling Test

**What it tests:**
- ✅ All components have error handling
- ✅ Errors are tracked and logged
- ✅ System continues operating despite errors
- ✅ Error statistics available

**What it checks:**
- Error tracking system operational
- Error counts by type and component
- Error rates within acceptable thresholds

**Monitor errors:**
```python
from src.monitoring import get_metrics_collector

metrics = get_metrics_collector()
error_stats = metrics.get_error_stats()

print("Total errors:", error_stats["total_errors"])
print("By component:", error_stats["by_component"])
```

### 5. Monitoring & Alerting Test

**What it tests:**
- ✅ Metrics collection for all key components
- ✅ Latency metrics (p50, p95, p99, mean)
- ✅ Throughput metrics (QPS, success rate)
- ✅ Alert generation for threshold violations

**What it checks:**
- Metrics collector operational
- All metric categories available
- Alert manager functional
- Active alerts tracked

**Monitor metrics:**
```python
from src.monitoring import get_metrics_collector, get_alert_manager

# Get all metrics
metrics = get_metrics_collector()
all_metrics = metrics.get_all_metrics()

print("Latency:", all_metrics["latency"]["retrieval"])
print("Throughput:", all_metrics["throughput"])
print("Service Health:", all_metrics["service_health"])

# Check alerts
alert_manager = get_alert_manager()
alerts = alert_manager.check_alerts()
print(f"Active alerts: {len(alerts)}")
```

## Running All Tests

### Option 1: Comprehensive Test Suite

Run all tests in one command:

```bash
python src/benchmarks/system_requirements.py
```

This runs:
1. Performance benchmarks (~5-10 minutes)
2. Load capacity test (~1 minute)
3. Graceful degradation test (~5 seconds)
4. Error handling test (~5 seconds)
5. Monitoring & alerting test (~5 seconds)

**Total time:** ~6-11 minutes

### Option 2: Interactive CLI

```bash
python rag.py --benchmark
```

Then select:
- `1` - Performance benchmarks only
- `2` - Load test only
- `3` - System requirements (comprehensive) ← **Recommended**

### Option 3: Individual Tests

Run each test separately for focused testing:

```bash
# Performance benchmarks
python src/benchmarks/performance.py

# Load test
python src/benchmarks/load_test.py

# System requirements (includes all)
python src/benchmarks/system_requirements.py
```

## Expected Results

### Performance Benchmarks

```
✅ PASS - Retrieval Latency (P95 < 100ms): P95=85.2ms
✅ PASS - Retrieval Quality (Score > 0.8): Mean Quality Score=0.852
✅ PASS - Compression Ratio (>30%): Overall compression=35.2%
✅ PASS - Cost Reduction (>40%): Cost reduction=42.5%
```

### Load Capacity

```
✅ LOAD TEST PASSED
Actual QPS: 52.3
Success Rate: 99.8%
Retrieval latency p95: 85.2ms
```

### System Requirements

```
✅ PASS - Load Capacity (50+ QPS): QPS: 52.3/50, Success: 99.8%, P95: 85.2ms
✅ PASS - Graceful Degradation (Circuit Breakers): Circuit breakers configured: True
✅ PASS - Error Handling (Comprehensive): Error tracking enabled: True
✅ PASS - Monitoring & Alerting: Latency: True, Throughput: True, Alerts: True
```

## Troubleshooting

### High Latency Issues

**Problem:** p95 latency > 100ms

**Solutions:**
1. Check database indexes:
   ```bash
   python optimize_db.py
   ```
2. Reduce retrieval `k` value
3. Check network latency to Bedrock services
4. Verify database connection pooling

### Low QPS Issues

**Problem:** Can't achieve 50+ QPS

**Solutions:**
1. Increase `max_workers` in load test
2. Check system resources (CPU, memory)
3. Verify database performance
4. Check for bottlenecks in pipeline

### Circuit Breaker Issues

**Problem:** Circuit breakers stuck in OPEN state

**Solutions:**
1. Verify Bedrock service availability
2. Check AWS credentials and permissions
3. Review error logs for specific failures
4. Wait for automatic recovery (60s timeout)

### High Error Rates

**Problem:** Error rate > 1%

**Solutions:**
1. Review error logs by component
2. Check service health metrics
3. Verify network connectivity
4. Check for rate limiting issues

## Monitoring in Production

### Continuous Monitoring

Set up metrics monitoring:

```python
from src.monitoring import get_metrics_collector, get_alert_manager

metrics = get_metrics_collector()
alert_manager = get_alert_manager()

# Check metrics periodically
def check_system_health():
    all_metrics = metrics.get_all_metrics()
    alerts = alert_manager.check_alerts()

    # Report critical alerts
    for alert in alerts:
        if alert.level == "critical":
            send_notification(alert)
```

### Alert Notifications

Register custom alert handlers:

```python
from src.monitoring.alerts import get_alert_manager

def send_slack_alert(alert):
    # Send alert to Slack
    pass

def send_email_alert(alert):
    # Send alert via email
    pass

alert_manager = get_alert_manager()
alert_manager.register_alert_callback(send_slack_alert)
alert_manager.register_alert_callback(send_email_alert)
```

## Best Practices

1. **Regular Testing**: Run system requirements tests weekly
2. **Load Testing**: Run load tests before deploying to production
3. **Monitor Continuously**: Set up monitoring dashboard
4. **Review Alerts**: Check alerts daily
5. **Tune Thresholds**: Adjust based on actual requirements
6. **Document Changes**: Document any threshold changes

## Next Steps

1. Run comprehensive test: `python src/benchmarks/system_requirements.py`
2. Review results in `system_requirements_test_results.json`
3. Set up monitoring dashboard (integrate metrics)
4. Configure alert notifications
5. Schedule regular testing

For more details, see `docs/SYSTEM_REQUIREMENTS.md`.

