"""
Real-time Query Monitoring for Phase 3 Enhancement

This module provides monitoring and analytics for query generation patterns,
logic issues, and correction effectiveness. Helps identify trends and areas
for improvement in the query logic validation system.

Key Features:
- Real-time monitoring of query generation patterns
- Logic issue detection and alerting
- Correction effectiveness tracking
- Performance metrics and analytics
- Anomaly detection for unusual query patterns

Example Usage:
    # Initialize monitor
    monitor = QueryLogicMonitor()
    
    # Log query analysis
    await monitor.log_query_analysis(
        question="What feedback about courses?",
        generated_sql="SELECT * FROM evaluation...",
        result_count=25,
        validation_applied=False,
        issues=[]
    )
    
    # Get analytics
    analytics = monitor.get_analytics()
    print(f"Query success rate: {analytics['success_rate']}")
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import re

from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis result for a single query."""
    timestamp: datetime
    question: str
    generated_sql: str
    result_count: int
    execution_time: float
    validation_applied: bool
    validation_issues: List[Dict[str, Any]]
    table_usage: List[str]
    logic_issues: List[str]
    success: bool


@dataclass
class MonitoringStats:
    """Aggregated monitoring statistics."""
    total_queries: int = 0
    successful_queries: int = 0
    validation_corrections: int = 0
    critical_issues_detected: int = 0
    empty_result_queries: int = 0
    average_execution_time: float = 0.0
    table_usage_distribution: Dict[str, int] = field(default_factory=dict)
    common_issues: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    correction_effectiveness: float = 0.0


class QueryLogicMonitor:
    """
    Monitors query generation for logic issues and tracks corrections.
    
    Provides real-time monitoring, analytics, and alerting for query
    generation patterns and validation effectiveness.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize query monitor.
        
        Args:
            max_history: Maximum number of query analyses to keep in memory
        """
        self.max_history = max_history
        
        # Query history (rolling window)
        self.query_history: deque = deque(maxlen=max_history)
        
        # Real-time statistics
        self.stats = MonitoringStats()
        
        # Issue patterns for detection
        self.logic_issue_patterns = [
            {
                "name": "incorrect_table_join",
                "pattern": r"rag_user_feedback.*JOIN.*learning_content",
                "severity": "critical",
                "description": "Incorrect join between rag_user_feedback and learning_content"
            },
            {
                "name": "semantic_mismatch",
                "pattern": r"LIKE.*\|\|.*name.*\|\|",
                "severity": "critical", 
                "description": "Semantic mismatch using LIKE with concatenated names"
            },
            {
                "name": "missing_where_clause",
                "pattern": r"SELECT.*FROM.*(?!.*WHERE)",
                "severity": "warning",
                "description": "Query without WHERE clause may return too many results"
            },
            {
                "name": "potential_cartesian_product",
                "pattern": r"FROM.*,.*(?!.*WHERE.*=)",
                "severity": "warning",
                "description": "Potential cartesian product without proper join conditions"
            }
        ]
        
        # Alert thresholds
        self.alert_thresholds = {
            "empty_result_rate": 0.3,  # 30% empty results
            "critical_issue_rate": 0.1,  # 10% critical issues
            "low_success_rate": 0.8,  # Below 80% success
            "high_correction_rate": 0.2  # Above 20% corrections needed
        }
    
    async def log_query_analysis(
        self,
        question: str,
        generated_sql: str,
        result_count: int,
        execution_time: float = 0.0,
        validation_applied: bool = False,
        validation_issues: Optional[List[Dict[str, Any]]] = None,
        success: bool = True
    ) -> None:
        """
        Log query analysis for monitoring and analytics.
        
        Args:
            question: Original natural language question
            generated_sql: Generated SQL query
            result_count: Number of rows returned
            execution_time: Query execution time in seconds
            validation_applied: Whether validation correction was applied
            validation_issues: List of validation issues found
            success: Whether the query executed successfully
        """
        try:
            # Detect logic issues
            logic_issues = await self._detect_logic_issues(generated_sql)
            
            # Extract table usage
            table_usage = self._extract_table_usage(generated_sql)
            
            # Create analysis record
            analysis = QueryAnalysis(
                timestamp=datetime.utcnow(),
                question=question,
                generated_sql=generated_sql,
                result_count=result_count,
                execution_time=execution_time,
                validation_applied=validation_applied,
                validation_issues=validation_issues or [],
                table_usage=table_usage,
                logic_issues=logic_issues,
                success=success
            )
            
            # Add to history
            self.query_history.append(analysis)
            
            # Update statistics
            await self._update_statistics(analysis)
            
            # Check for alerts
            await self._check_alerts(analysis)
            
            # Log significant events
            if validation_applied:
                logger.info(f"Query validation correction applied for: {question[:50]}...")
            
            if logic_issues:
                logger.warning(f"Logic issues detected in query: {logic_issues}")
            
            if result_count == 0 and "feedback" in question.lower():
                await self._alert_empty_feedback_query(analysis)
            
        except Exception as e:
            logger.error(f"Error logging query analysis: {e}")
    
    async def _detect_logic_issues(self, sql_query: str) -> List[str]:
        """
        Detect logic issues in SQL query using pattern matching.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            List of detected issue names
        """
        issues = []
        
        for pattern_info in self.logic_issue_patterns:
            if re.search(pattern_info["pattern"], sql_query, re.IGNORECASE):
                issues.append(pattern_info["name"])
                
                # Log the issue
                logger.warning(
                    f"Logic issue detected: {pattern_info['name']} - "
                    f"{pattern_info['description']}"
                )
        
        return issues
    
    def _extract_table_usage(self, sql_query: str) -> List[str]:
        """
        Extract table names used in SQL query.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            List of table names found in the query
        """
        # Common table names in our system
        known_tables = [
            'users', 'learning_content', 'attendance', 'evaluation',
            'rag_embeddings', 'rag_user_feedback'
        ]
        
        tables_used = []
        sql_lower = sql_query.lower()
        
        for table in known_tables:
            if re.search(rf'\b{table}\b', sql_lower):
                tables_used.append(table)
        
        return tables_used
    
    async def _update_statistics(self, analysis: QueryAnalysis) -> None:
        """Update monitoring statistics with new analysis."""
        
        # Basic counts
        self.stats.total_queries += 1
        if analysis.success:
            self.stats.successful_queries += 1
        if analysis.validation_applied:
            self.stats.validation_corrections += 1
        if analysis.result_count == 0:
            self.stats.empty_result_queries += 1
        
        # Critical issues
        critical_issues = [
            issue for issue in analysis.validation_issues 
            if issue.get("severity") == "critical"
        ]
        if critical_issues:
            self.stats.critical_issues_detected += 1
        
        # Table usage
        for table in analysis.table_usage:
            self.stats.table_usage_distribution[table] = (
                self.stats.table_usage_distribution.get(table, 0) + 1
            )
        
        # Common issues tracking
        for issue in analysis.logic_issues:
            self.stats.common_issues[issue] = (
                self.stats.common_issues.get(issue, 0) + 1
            )
        
        # Calculate rates
        if self.stats.total_queries > 0:
            self.stats.success_rate = self.stats.successful_queries / self.stats.total_queries
            self.stats.correction_effectiveness = (
                self.stats.validation_corrections / self.stats.total_queries
            )
        
        # Average execution time (rolling average)
        if analysis.execution_time > 0:
            current_avg = self.stats.average_execution_time
            total = self.stats.total_queries
            self.stats.average_execution_time = (
                (current_avg * (total - 1) + analysis.execution_time) / total
            )
    
    async def _check_alerts(self, analysis: QueryAnalysis) -> None:
        """Check if any alert conditions are met."""
        
        # Check empty result rate for feedback queries
        if "feedback" in analysis.question.lower() and analysis.result_count == 0:
            feedback_queries = [
                a for a in self.query_history 
                if "feedback" in a.question.lower()
            ]
            if len(feedback_queries) >= 5:  # Minimum sample size
                empty_rate = sum(1 for a in feedback_queries if a.result_count == 0) / len(feedback_queries)
                if empty_rate > self.alert_thresholds["empty_result_rate"]:
                    await self._send_alert(
                        "high_empty_feedback_rate",
                        f"High empty result rate for feedback queries: {empty_rate:.2%}"
                    )
        
        # Check critical issue rate
        if self.stats.total_queries >= 10:  # Minimum sample size
            critical_rate = self.stats.critical_issues_detected / self.stats.total_queries
            if critical_rate > self.alert_thresholds["critical_issue_rate"]:
                await self._send_alert(
                    "high_critical_issue_rate",
                    f"High critical issue rate: {critical_rate:.2%}"
                )
        
        # Check overall success rate
        if (self.stats.total_queries >= 10 and 
            self.stats.success_rate < self.alert_thresholds["low_success_rate"]):
            await self._send_alert(
                "low_success_rate",
                f"Low query success rate: {self.stats.success_rate:.2%}"
            )
    
    async def _alert_empty_feedback_query(self, analysis: QueryAnalysis) -> None:
        """Alert on empty results for feedback queries."""
        logger.warning(
            f"Empty feedback query result - Question: '{analysis.question}', "
            f"Tables used: {analysis.table_usage}, "
            f"Issues: {analysis.logic_issues}"
        )
        
        # Check if this might be the original problem
        if ("learning" in analysis.question.lower() and 
            "rag_user_feedback" in analysis.table_usage and
            "learning_content" in analysis.table_usage):
            
            await self._send_alert(
                "original_problem_detected",
                f"Potential original problem pattern detected: {analysis.question[:100]}"
            )
    
    async def _send_alert(self, alert_type: str, message: str) -> None:
        """Send alert for monitoring issues."""
        logger.error(f"QUERY MONITOR ALERT [{alert_type}]: {message}")
        
        # In a production system, this would integrate with alerting systems
        # like Slack, email, PagerDuty, etc.
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics and statistics.
        
        Returns:
            Dictionary with monitoring analytics
        """
        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        
        analytics = {
            "overview": {
                "total_queries": self.stats.total_queries,
                "success_rate": self.stats.success_rate,
                "average_execution_time": self.stats.average_execution_time,
                "validation_corrections": self.stats.validation_corrections,
                "correction_rate": self.stats.correction_effectiveness
            },
            "issues": {
                "critical_issues_detected": self.stats.critical_issues_detected,
                "empty_result_queries": self.stats.empty_result_queries,
                "common_issues": dict(self.stats.common_issues),
                "empty_result_rate": (
                    self.stats.empty_result_queries / max(self.stats.total_queries, 1)
                )
            },
            "table_usage": dict(self.stats.table_usage_distribution),
            "recent_trends": self._analyze_recent_trends(recent_queries),
            "recommendations": self._generate_recommendations()
        }
        
        return analytics
    
    def _analyze_recent_trends(self, recent_queries: List[QueryAnalysis]) -> Dict[str, Any]:
        """Analyze trends in recent queries."""
        if not recent_queries:
            return {}
        
        # Success rate trend
        recent_success_rate = sum(1 for q in recent_queries if q.success) / len(recent_queries)
        
        # Validation rate trend
        recent_validation_rate = sum(1 for q in recent_queries if q.validation_applied) / len(recent_queries)
        
        # Most common tables
        table_counts = defaultdict(int)
        for query in recent_queries:
            for table in query.table_usage:
                table_counts[table] += 1
        
        return {
            "recent_success_rate": recent_success_rate,
            "recent_validation_rate": recent_validation_rate,
            "most_used_tables": dict(sorted(table_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "trend_direction": "improving" if recent_success_rate > self.stats.success_rate else "declining"
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        if self.stats.correction_effectiveness > 0.15:
            recommendations.append(
                "High validation correction rate detected. Consider improving query generation prompts."
            )
        
        if self.stats.empty_result_queries / max(self.stats.total_queries, 1) > 0.2:
            recommendations.append(
                "High empty result rate. Review table relationships and data availability."
            )
        
        if "incorrect_table_join" in self.stats.common_issues:
            recommendations.append(
                "Frequent incorrect table joins detected. Review schema guidance and table descriptions."
            )
        
        if self.stats.success_rate < 0.9:
            recommendations.append(
                "Query success rate below 90%. Consider additional validation rules or query templates."
            )
        
        return recommendations
    
    def export_analytics(self, filepath: str) -> None:
        """Export analytics to JSON file."""
        try:
            analytics = self.get_analytics()
            with open(filepath, 'w') as f:
                json.dump(analytics, f, indent=2, default=str)
            logger.info(f"Analytics exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
    
    def reset_stats(self) -> None:
        """Reset monitoring statistics."""
        self.query_history.clear()
        self.stats = MonitoringStats()
