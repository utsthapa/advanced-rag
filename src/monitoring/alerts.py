"""Alerting system for key metrics"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class AlertLevel:
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a single alert"""

    level: str
    component: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float

    def __init__(
        self,
        level: str,
        component: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
    ):
        import time

        self.level = level
        self.component = component
        self.message = message
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold = threshold
        self.timestamp = time.time()


class AlertManager:
    """
    Manages alerts for key metrics and system health.

    Monitors:
    - Latency thresholds (p95 < 100ms for retrieval)
    - Error rates (should be < 1%)
    - Circuit breaker states (should be CLOSED)
    - Throughput (should handle 50+ QPS)
    - Service health degradation
    """

    def __init__(self):
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        self.alert_callbacks: List[callable] = []

    def check_alerts(self) -> List[Alert]:
        """Check all metrics and generate alerts"""
        metrics = get_metrics_collector()
        all_metrics = metrics.get_all_metrics()
        alerts = []

        # Check latency alerts
        retrieval_latency = all_metrics["latency"].get("retrieval", {})
        if retrieval_latency:
            p95 = retrieval_latency.get("p95", 0)
            if p95 > 100:
                alerts.append(
                    Alert(
                        level=AlertLevel.CRITICAL,
                        component="retrieval",
                        message=f"Retrieval latency p95 ({p95:.1f}ms) exceeds target (100ms)",
                        metric_name="retrieval_latency_p95",
                        current_value=p95,
                        threshold=100,
                    )
                )
            elif p95 > 80:
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        component="retrieval",
                        message=f"Retrieval latency p95 ({p95:.1f}ms) approaching target (100ms)",
                        metric_name="retrieval_latency_p95",
                        current_value=p95,
                        threshold=100,
                    )
                )

        # Check error rate
        throughput = all_metrics.get("throughput", {})
        if throughput:
            total_requests = throughput.get("total_requests", 0)
            failed_requests = throughput.get("failed_requests", 0)
            if total_requests > 0:
                error_rate = (failed_requests / total_requests) * 100
                if error_rate > 5:
                    alerts.append(
                        Alert(
                            level=AlertLevel.CRITICAL,
                            component="system",
                            message=f"Error rate ({error_rate:.1f}%) exceeds threshold (5%)",
                            metric_name="error_rate",
                            current_value=error_rate,
                            threshold=5,
                        )
                    )
                elif error_rate > 1:
                    alerts.append(
                        Alert(
                            level=AlertLevel.WARNING,
                            component="system",
                            message=f"Error rate ({error_rate:.1f}%) exceeds target (1%)",
                            metric_name="error_rate",
                            current_value=error_rate,
                            threshold=1,
                        )
                    )

        # Check QPS
        qps = throughput.get("qps", 0)
        if qps > 0 and qps < 50:
            # Only alert if we're processing requests but below target
            if total_requests > 100:  # Only alert after enough requests
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        component="throughput",
                        message=f"QPS ({qps:.1f}) below target (50 QPS)",
                        metric_name="qps",
                        current_value=qps,
                        threshold=50,
                    )
                )

        # Check circuit breaker states
        service_health = all_metrics.get("service_health", {})
        cb_states = service_health.get("circuit_breaker_states", {})
        for service, state in cb_states.items():
            if state == "open":
                alerts.append(
                    Alert(
                        level=AlertLevel.CRITICAL,
                        component=service,
                        message=f"Circuit breaker is OPEN - service unavailable",
                        metric_name="circuit_breaker_state",
                        current_value=0,
                        threshold=1,
                    )
                )

        # Check service failure rates
        failure_rates = service_health.get("failure_rates", {})
        for service, rate in failure_rates.items():
            if rate > 0.1:  # > 10% failure rate
                alerts.append(
                    Alert(
                        level=AlertLevel.CRITICAL,
                        component=service,
                        message=f"Service failure rate ({rate*100:.1f}%) is high",
                        metric_name="service_failure_rate",
                        current_value=rate * 100,
                        threshold=10,
                    )
                )
            elif rate > 0.05:  # > 5% failure rate
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        component=service,
                        message=f"Service failure rate ({rate*100:.1f}%) is elevated",
                        metric_name="service_failure_rate",
                        current_value=rate * 100,
                        threshold=5,
                    )
                )

        # Store and notify
        for alert in alerts:
            self._add_alert(alert)
            self._notify_alert(alert)

        return alerts

    def _add_alert(self, alert: Alert):
        """Add alert to history"""
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

    def _notify_alert(self, alert: Alert):
        """Notify about an alert"""
        # Log the alert
        log_msg = f"[{alert.level.upper()}] {alert.component}: {alert.message}"
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: callable):
        """Register a callback to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def get_recent_alerts(
        self, level: Optional[str] = None, limit: int = 50
    ) -> List[Alert]:
        """Get recent alerts, optionally filtered by level"""
        alerts = self.alert_history[-limit:]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return alerts

    def get_active_critical_alerts(self) -> List[Alert]:
        """Get active critical alerts"""
        return [a for a in self.alert_history if a.level == AlertLevel.CRITICAL]


# Global alert manager instance
_alert_manager = None


def get_alert_manager():
    """Get or create global alert manager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
