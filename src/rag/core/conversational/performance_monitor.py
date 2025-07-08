#!/usr/bin/env python3
"""
Conversational Performance Monitor - Phase 3: Learning Integration & Monitoring

This module provides comprehensive monitoring and alerting capabilities for the
hybrid LLM + template conversational intelligence system, focusing on performance
tracking, cost optimization, and system health monitoring.

Key Features:
- Real-time performance monitoring with configurable alerts
- Cost tracking and optimization recommendations
- System health monitoring with early warning detection
- Privacy-compliant audit trail generation
- Learning effectiveness measurement and trending

Component Reuse Strategy:
- Leverages existing audit logging infrastructure (100% reuse)
- Uses existing privacy-safe logging mechanisms (100% reuse) 
- Integrates with existing performance tracking (enhancement only)
- Reuses existing monitoring patterns and methodologies
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque

# Reuse existing imports
from .handler import ConversationalPattern
from .learning_integrator import LearningFeedback, RoutingStrategy

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics for monitoring."""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    LLM_USAGE_RATE = "llm_usage_rate"
    COST_PER_QUERY = "cost_per_query"
    SYSTEM_HEALTH = "system_health"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class RoutingDecision:
    """Simplified routing decision for monitoring (avoiding circular import)."""
    strategy_used: RoutingStrategy
    llm_enhancement_used: bool = False
    processing_time: float = 0.0
    
    @property
    def strategy(self) -> RoutingStrategy:
        """Compatibility property."""
        return self.strategy_used


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    pattern_type: Optional[ConversationalPattern] = None
    routing_strategy: Optional[RoutingStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'pattern_type': self.pattern_type.value if self.pattern_type else None,
            'routing_strategy': self.routing_strategy.value if self.routing_strategy else None,
            'metadata': self.metadata
        }


@dataclass 
class Alert:
    """System alert for monitoring conditions."""
    level: AlertLevel
    message: str
    metric_type: MetricType
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    pattern_type: Optional[ConversationalPattern] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'level': self.level.value,
            'message': self.message,
            'metric_type': self.metric_type.value,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'pattern_type': self.pattern_type.value if self.pattern_type else None
        }


class ConversationalPerformanceMonitor:
    """
    Comprehensive monitoring system for hybrid conversational intelligence.
    
    This component provides real-time monitoring, alerting, and performance
    tracking for the hybrid LLM + template system, with focus on cost
    optimization and system health.
    
    Component Reuse:
    - Uses existing audit logging infrastructure
    - Leverages existing privacy-safe logging mechanisms
    - Integrates with existing performance tracking patterns
    - Maintains consistency with existing monitoring approaches
    """
    
    def __init__(self, 
                 alert_callback: Optional[Callable[[Alert], None]] = None,
                 cost_per_llm_call: float = 0.001,  # Configurable cost tracking
                 monitoring_window_minutes: int = 60):
        """
        Initialize performance monitor.
        
        Args:
            alert_callback: Optional callback function for alert notifications
            cost_per_llm_call: Cost per LLM API call for cost tracking
            monitoring_window_minutes: Window for rolling metrics calculation
        """
        self.alert_callback = alert_callback
        self.cost_per_llm_call = cost_per_llm_call
        self.monitoring_window = timedelta(minutes=monitoring_window_minutes)
        
        # Metric storage (rolling windows)
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.pattern_metrics: Dict[ConversationalPattern, Dict[MetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )
        
        # Configurable thresholds
        self.thresholds = {
            MetricType.RESPONSE_TIME: {'warning': 2000, 'error': 5000},  # ms
            MetricType.SUCCESS_RATE: {'warning': 0.8, 'error': 0.7},     # ratio
            MetricType.LLM_USAGE_RATE: {'warning': 0.4, 'error': 0.6},   # ratio
            MetricType.COST_PER_QUERY: {'warning': 0.01, 'error': 0.02}, # dollars
            MetricType.SYSTEM_HEALTH: {'warning': 0.9, 'error': 0.8},    # ratio
            MetricType.USER_SATISFACTION: {'warning': 0.7, 'error': 0.6} # ratio
        }
        
        # Performance tracking
        self.total_queries = 0
        self.total_llm_calls = 0
        self.total_cost = 0.0
        self.start_time = datetime.now()
        
        logger.info("Performance monitor initialized with cost tracking and alerting")

    async def record_interaction(self, feedback: LearningFeedback,
                               routing_decision: RoutingDecision) -> None:
        """
        Record a conversational interaction for monitoring.
        
        Args:
            feedback: Learning feedback from the interaction
            routing_decision: Routing decision details
        """
        try:
            self.total_queries += 1
            
            # Record response time metric
            await self._record_metric(
                MetricType.RESPONSE_TIME,
                feedback.response_time_ms,
                feedback.pattern_type,
                feedback.routing_strategy
            )
            
            # Record success rate metric
            success_value = 1.0 if feedback.was_helpful else 0.0
            await self._record_metric(
                MetricType.SUCCESS_RATE,
                success_value,
                feedback.pattern_type,
                feedback.routing_strategy
            )
            
            # Record LLM usage and cost if applicable
            if feedback.llm_used:
                self.total_llm_calls += 1
                self.total_cost += self.cost_per_llm_call
                
                await self._record_metric(
                    MetricType.COST_PER_QUERY,
                    self.cost_per_llm_call,
                    feedback.pattern_type,
                    feedback.routing_strategy
                )
            
            # Calculate and record LLM usage rate
            llm_usage_rate = self.total_llm_calls / max(1, self.total_queries)
            await self._record_metric(MetricType.LLM_USAGE_RATE, llm_usage_rate)
            
            # Record user satisfaction if available
            if feedback.user_satisfaction is not None:
                await self._record_metric(
                    MetricType.USER_SATISFACTION,
                    feedback.user_satisfaction,
                    feedback.pattern_type,
                    feedback.routing_strategy
                )
            
            # Calculate system health score
            health_score = await self._calculate_system_health()
            await self._record_metric(MetricType.SYSTEM_HEALTH, health_score)
            
            # Check for alert conditions
            await self._check_alert_conditions()
            
            logger.debug(f"Recorded interaction: LLM used: {feedback.llm_used}, "
                        f"Success: {feedback.was_helpful}, "
                        f"Response time: {feedback.response_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")

    async def _record_metric(self, metric_type: MetricType, value: float,
                           pattern_type: Optional[ConversationalPattern] = None,
                           routing_strategy: Optional[RoutingStrategy] = None) -> None:
        """Record a metric measurement."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            pattern_type=pattern_type,
            routing_strategy=routing_strategy
        )
        
        # Store in global metrics
        self.metrics[metric_type].append(metric)
        
        # Store in pattern-specific metrics if pattern provided
        if pattern_type:
            self.pattern_metrics[pattern_type][metric_type].append(metric)

    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        health_components = []
        
        # Response time health (last 10 measurements)
        recent_response_times = list(self.metrics[MetricType.RESPONSE_TIME])[-10:]
        if recent_response_times:
            avg_response_time = sum(m.value for m in recent_response_times) / len(recent_response_times)
            # Health decreases as response time increases
            time_health = max(0, 1 - (avg_response_time / 5000))  # 5000ms = 0 health
            health_components.append(time_health)
        
        # Success rate health (last 20 measurements)
        recent_success = list(self.metrics[MetricType.SUCCESS_RATE])[-20:]
        if recent_success:
            success_rate = sum(m.value for m in recent_success) / len(recent_success)
            health_components.append(success_rate)
        
        # Cost efficiency health
        if self.total_queries > 0:
            avg_cost_per_query = self.total_cost / self.total_queries
            # Health decreases as cost per query increases
            cost_health = max(0, 1 - (avg_cost_per_query / 0.05))  # $0.05 = 0 health
            health_components.append(cost_health)
        
        # Overall health is average of components
        return sum(health_components) / max(1, len(health_components))

    async def _check_alert_conditions(self) -> None:
        """Check current metrics against thresholds and generate alerts."""
        current_time = datetime.now()
        
        # Check each metric type
        for metric_type, thresholds in self.thresholds.items():
            recent_metrics = [m for m in self.metrics[metric_type] 
                            if current_time - m.timestamp <= self.monitoring_window]
            
            if not recent_metrics:
                continue
            
            # Calculate current value (average for most metrics, latest for rates)
            if metric_type in [MetricType.LLM_USAGE_RATE, MetricType.SYSTEM_HEALTH]:
                current_value = recent_metrics[-1].value
            else:
                current_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Check thresholds
            alert_level = None
            threshold_value = None
            
            if metric_type in [MetricType.SUCCESS_RATE, MetricType.SYSTEM_HEALTH, MetricType.USER_SATISFACTION]:
                # Lower values are worse
                if current_value < thresholds['error']:
                    alert_level = AlertLevel.ERROR
                    threshold_value = thresholds['error']
                elif current_value < thresholds['warning']:
                    alert_level = AlertLevel.WARNING
                    threshold_value = thresholds['warning']
            else:
                # Higher values are worse
                if current_value > thresholds['error']:
                    alert_level = AlertLevel.ERROR
                    threshold_value = thresholds['error']
                elif current_value > thresholds['warning']:
                    alert_level = AlertLevel.WARNING
                    threshold_value = thresholds['warning']
            
            # Generate alert if threshold exceeded
            if alert_level:
                await self._generate_alert(alert_level, metric_type, current_value, threshold_value)

    async def _generate_alert(self, level: AlertLevel, metric_type: MetricType,
                            current_value: float, threshold_value: float) -> None:
        """Generate and handle an alert."""
        # Check if we've already alerted on this recently (avoid spam)
        recent_alerts = [a for a in self.alerts[-10:] 
                        if a.metric_type == metric_type and a.level == level]
        if recent_alerts:
            last_alert = recent_alerts[-1]
            if datetime.now() - last_alert.timestamp < timedelta(minutes=15):
                return  # Don't spam alerts
        
        # Create alert message
        message = await self._create_alert_message(metric_type, current_value, threshold_value, level)
        
        alert = Alert(
            level=level,
            message=message,
            metric_type=metric_type,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        self.alerts.append(alert)
        
        # Log alert using existing privacy-safe logging
        logger.warning(f"Performance Alert [{level.value.upper()}]: {message}")
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    async def _create_alert_message(self, metric_type: MetricType, current_value: float,
                                  threshold_value: float, level: AlertLevel) -> str:
        """Create human-readable alert message."""
        if metric_type == MetricType.RESPONSE_TIME:
            return f"Response time {current_value:.0f}ms exceeds {level.value} threshold {threshold_value:.0f}ms"
        elif metric_type == MetricType.SUCCESS_RATE:
            return f"Success rate {current_value:.2%} below {level.value} threshold {threshold_value:.2%}"
        elif metric_type == MetricType.LLM_USAGE_RATE:
            return f"LLM usage rate {current_value:.2%} exceeds {level.value} threshold {threshold_value:.2%}"
        elif metric_type == MetricType.COST_PER_QUERY:
            return f"Cost per query ${current_value:.4f} exceeds {level.value} threshold ${threshold_value:.4f}"
        elif metric_type == MetricType.SYSTEM_HEALTH:
            return f"System health {current_value:.2%} below {level.value} threshold {threshold_value:.2%}"
        elif metric_type == MetricType.USER_SATISFACTION:
            return f"User satisfaction {current_value:.2%} below {level.value} threshold {threshold_value:.2%}"
        else:
            return f"Metric {metric_type.value} value {current_value:.3f} triggers {level.value} alert"

    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance analytics and recommendations
        """
        current_time = datetime.now()
        uptime = current_time - self.start_time
        
        report = {
            'summary': {
                'total_queries': self.total_queries,
                'total_llm_calls': self.total_llm_calls,
                'total_cost': self.total_cost,
                'llm_usage_rate': self.total_llm_calls / max(1, self.total_queries),
                'avg_cost_per_query': self.total_cost / max(1, self.total_queries),
                'uptime_hours': uptime.total_seconds() / 3600,
                'queries_per_hour': self.total_queries / max(1, uptime.total_seconds() / 3600)
            },
            'current_metrics': {},
            'pattern_performance': {},
            'recent_alerts': [],
            'recommendations': []
        }
        
        # Current metrics (last window)
        for metric_type in MetricType:
            recent_metrics = [m for m in self.metrics[metric_type] 
                            if current_time - m.timestamp <= self.monitoring_window]
            if recent_metrics:
                if metric_type in [MetricType.LLM_USAGE_RATE, MetricType.SYSTEM_HEALTH]:
                    value = recent_metrics[-1].value
                else:
                    value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                
                report['current_metrics'][metric_type.value] = {
                    'value': value,
                    'sample_count': len(recent_metrics),
                    'threshold_warning': self.thresholds[metric_type]['warning'],
                    'threshold_error': self.thresholds[metric_type]['error']
                }
        
        # Pattern-specific performance
        for pattern_type, pattern_metrics in self.pattern_metrics.items():
            pattern_report = {}
            for metric_type, metrics in pattern_metrics.items():
                if metrics:
                    recent_metrics = [m for m in metrics 
                                    if current_time - m.timestamp <= self.monitoring_window]
                    if recent_metrics:
                        avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                        pattern_report[metric_type.value] = {
                            'value': avg_value,
                            'sample_count': len(recent_metrics)
                        }
            
            if pattern_report:
                report['pattern_performance'][pattern_type.value] = pattern_report
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts[-20:] 
                        if current_time - a.timestamp <= timedelta(hours=24)]
        report['recent_alerts'] = [alert.to_dict() for alert in recent_alerts]
        
        # Generate recommendations
        await self._generate_performance_recommendations(report)
        
        return report

    async def _generate_performance_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate actionable performance recommendations."""
        recommendations = []
        
        summary = report['summary']
        current_metrics = report['current_metrics']
        
        # Cost optimization recommendations
        if summary['llm_usage_rate'] > 0.3:
            avg_cost = summary['avg_cost_per_query']
            if avg_cost > 0.01:
                recommendations.append(
                    f"High LLM usage ({summary['llm_usage_rate']:.1%}) and cost "
                    f"(${avg_cost:.4f}/query). Consider raising confidence thresholds."
                )
        
        # Performance optimization recommendations
        if 'response_time' in current_metrics:
            avg_response_time = current_metrics['response_time']['value']
            if avg_response_time > 1000:
                recommendations.append(
                    f"Average response time high ({avg_response_time:.0f}ms). "
                    f"Consider optimizing template selection or LLM timeouts."
                )
        
        # Success rate recommendations
        if 'success_rate' in current_metrics:
            success_rate = current_metrics['success_rate']['value']
            if success_rate < 0.8:
                recommendations.append(
                    f"Success rate low ({success_rate:.1%}). "
                    f"Review pattern templates and LLM prompts for improvement."
                )
        
        # System health recommendations
        if 'system_health' in current_metrics:
            health_score = current_metrics['system_health']['value']
            if health_score < 0.8:
                recommendations.append(
                    f"System health score low ({health_score:.1%}). "
                    f"Review recent alerts and consider scaling resources."
                )
        
        report['recommendations'] = recommendations

    def update_thresholds(self, metric_type: MetricType, 
                         warning_threshold: float, error_threshold: float) -> None:
        """Update monitoring thresholds for a metric type."""
        self.thresholds[metric_type] = {
            'warning': warning_threshold,
            'error': error_threshold
        }
        logger.info(f"Updated thresholds for {metric_type.value}: "
                   f"warning={warning_threshold}, error={error_threshold}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        return {
            'status': 'active',
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_metrics_recorded': sum(len(metrics) for metrics in self.metrics.values()),
            'total_alerts_generated': len(self.alerts),
            'monitoring_window_minutes': self.monitoring_window.total_seconds() / 60,
            'cost_per_llm_call': self.cost_per_llm_call,
            'thresholds_configured': len(self.thresholds)
        }


# Example usage and integration
class PerformanceMonitoringExample:
    """Example usage of the performance monitor."""
    
    @staticmethod
    async def demonstrate_monitoring():
        """Demonstrate performance monitoring capabilities."""
        # Custom alert handler
        def alert_handler(alert: Alert):
            print(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # Initialize monitor
        monitor = ConversationalPerformanceMonitor(
            alert_callback=alert_handler,
            cost_per_llm_call=0.002
        )
        
        # Simulate some interactions
        
        for i in range(10):
            feedback = LearningFeedback(
                query=f"Test query {i}",
                pattern_type=ConversationalPattern.GREETING,
                routing_strategy=RoutingStrategy.TEMPLATE_FIRST,
                llm_used=(i % 3 == 0),  # LLM used every 3rd query
                was_helpful=(i % 5 != 0),  # Fails every 5th query
                response_time_ms=50 + (i * 10),
                confidence_score=0.8
            )
            
            routing_decision = RoutingDecision(
                strategy=RoutingStrategy.TEMPLATE_FIRST,
                confidence=0.8,
                reasoning="Test routing",
                llm_enhanced=feedback.llm_used
            )
            
            await monitor.record_interaction(feedback, routing_decision)
        
        # Get performance report
        report = await monitor.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    # Run example
    asyncio.run(PerformanceMonitoringExample.demonstrate_monitoring())
