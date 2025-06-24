"""
Circuit breaker pattern and fallback mechanisms for robust LLM classification.

This module implements sophisticated resilience patterns including:
- Circuit breaker pattern to prevent cascading failures
- Exponential backoff with jitter for retry logic
- Real-time metrics collection and monitoring
- Graceful degradation when LLM is unavailable
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .data_structures import (
    CircuitBreaker, FallbackMetrics, RetryConfig, CircuitBreakerState
)
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class CircuitBreakerManager:
    """
    Manages circuit breaker functionality for LLM classification failures.
    
    Implements the circuit breaker pattern to provide fault tolerance
    and prevent cascading failures when the LLM service is degraded.
    """
    
    def __init__(self, settings=None):
        """
        Initialize circuit breaker manager.
        
        Args:
            settings: Configuration settings (optional)
        """
        self.settings = settings
        
        # Initialize circuit breaker with configurable parameters
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(settings, 'classification_failure_threshold', 5) if settings else 5,
            recovery_timeout=getattr(settings, 'classification_recovery_timeout', 60.0) if settings else 60.0,
            half_open_max_calls=getattr(settings, 'classification_half_open_calls', 3) if settings else 3
        )
        
        # Initialize retry configuration
        self.retry_config = RetryConfig(
            max_retries=getattr(settings, 'classification_max_retries', 3) if settings else 3,
            base_delay=getattr(settings, 'classification_base_delay', 1.0) if settings else 1.0,
            max_delay=getattr(settings, 'classification_max_delay', 30.0) if settings else 30.0
        )
        
        # Initialize metrics tracking
        self.metrics = FallbackMetrics()
    
    def can_execute_llm_call(self) -> bool:
        """
        Check if LLM call is allowed by circuit breaker.
        
        Returns:
            True if LLM call is allowed, False otherwise
        """
        return self.circuit_breaker.can_execute()
    
    def record_llm_success(self, response_time: float) -> None:
        """
        Record successful LLM operation.
        
        Args:
            response_time: Time taken for the operation in seconds
        """
        self.circuit_breaker.record_success()
        self.metrics.record_llm_success(response_time)
        
        logger.debug(f"LLM success recorded: {response_time:.3f}s, circuit breaker: {self.circuit_breaker.state.value}")
    
    def record_llm_failure(self) -> None:
        """Record failed LLM operation."""
        self.circuit_breaker.record_failure()
        self.metrics.record_llm_failure()
        
        logger.warning(
            f"LLM failure recorded: {self.circuit_breaker.failure_count}/{self.circuit_breaker.failure_threshold}, "
            f"circuit breaker: {self.circuit_breaker.state.value}"
        )
    
    def record_circuit_breaker_block(self) -> None:
        """Record circuit breaker intervention."""
        self.metrics.record_circuit_breaker_block()
        logger.info("Circuit breaker blocked LLM call")
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive circuit breaker statistics.
        
        Returns:
            Dictionary with circuit breaker status and metrics
        """
        return {
            "state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count,
            "failure_threshold": self.circuit_breaker.failure_threshold,
            "recovery_timeout": self.circuit_breaker.recovery_timeout,
            "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None,
            "half_open_attempts": self.circuit_breaker.half_open_attempts,
            "half_open_max_calls": self.circuit_breaker.half_open_max_calls,
            "time_until_recovery": self._get_time_until_recovery()
        }
    
    def _get_time_until_recovery(self) -> Optional[float]:
        """
        Get time until circuit breaker recovery attempt.
        
        Returns:
            Seconds until recovery, None if not applicable
        """
        if (self.circuit_breaker.state == CircuitBreakerState.OPEN and 
            self.circuit_breaker.last_failure_time):
            
            recovery_time = self.circuit_breaker.last_failure_time + timedelta(
                seconds=self.circuit_breaker.recovery_timeout
            )
            time_until_recovery = (recovery_time - datetime.now()).total_seconds()
            return max(0, time_until_recovery)
        
        return None


class RetryManager:
    """
    Manages retry logic with exponential backoff for failed operations.
    
    Provides sophisticated retry mechanisms with jitter to prevent
    thundering herd problems and configurable backoff strategies.
    """
    
    def __init__(self, retry_config: RetryConfig):
        """
        Initialize retry manager.
        
        Args:
            retry_config: Configuration for retry behavior
        """
        self.retry_config = retry_config
        self.metrics = FallbackMetrics()
    
    async def execute_with_retry(self, operation, *args, session_id: Optional[str] = None, **kwargs):
        """
        Execute operation with retry logic.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            session_id: Optional session identifier for logging
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of successful operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                if attempt > 0:
                    # Apply exponential backoff
                    delay = self.retry_config.get_delay(attempt - 1)
                    logger.info(
                        f"Retrying operation after {delay:.2f}s delay "
                        f"(attempt {attempt + 1}/{self.retry_config.max_retries + 1}, "
                        f"session: {session_id or 'anonymous'})"
                    )
                    await asyncio.sleep(delay)
                    self.metrics.record_retry()
                
                # Attempt operation
                result = await operation(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        f"Operation succeeded on attempt {attempt + 1} "
                        f"(session: {session_id or 'anonymous'})"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Operation attempt {attempt + 1} failed: {e} "
                    f"(session: {session_id or 'anonymous'})"
                )
                
                # If this was the last attempt, we'll raise the exception
                if attempt == self.retry_config.max_retries:
                    logger.error(
                        f"All retry attempts exhausted for operation "
                        f"(session: {session_id or 'anonymous'})"
                    )
                    break
        
        # All attempts failed
        raise last_exception
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get retry mechanism statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        return {
            "config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
                "backoff_multiplier": self.retry_config.backoff_multiplier,
                "jitter_enabled": self.retry_config.jitter
            },
            "metrics": {
                "total_retries": self.metrics.retry_attempts,
                "retry_rate": self.metrics.retry_attempts / max(1, self.metrics.total_attempts)
            }
        }


class FallbackManager:
    """
    Manages fallback mechanisms and metrics collection.
    
    Coordinates multiple fallback strategies and provides comprehensive
    monitoring of system resilience and performance.
    """
    
    def __init__(self, settings=None):
        """
        Initialize fallback manager.
        
        Args:
            settings: Configuration settings (optional)
        """
        self.circuit_breaker_manager = CircuitBreakerManager(settings)
        self.retry_manager = RetryManager(self.circuit_breaker_manager.retry_config)
        self.metrics = FallbackMetrics()
    
    def record_classification_attempt(self) -> None:
        """Record a classification attempt."""
        self.metrics.record_attempt()
        self.circuit_breaker_manager.metrics.record_attempt()
    
    def record_fallback_activation(self, response_time: float) -> None:
        """
        Record activation of fallback classification.
        
        Args:
            response_time: Time taken for fallback classification
        """
        self.metrics.record_fallback(response_time)
        logger.info(f"Fallback classification activated: {response_time:.3f}s")
    
    def can_attempt_llm_classification(self) -> bool:
        """
        Check if LLM classification attempt is allowed.
        
        Returns:
            True if LLM classification can be attempted
        """
        can_execute = self.circuit_breaker_manager.can_execute_llm_call()
        
        if not can_execute:
            self.circuit_breaker_manager.record_circuit_breaker_block()
        
        return can_execute
    
    async def execute_llm_with_fallback(
        self, 
        llm_operation, 
        fallback_operation,
        *args,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """
        Execute LLM operation with comprehensive fallback.
        
        Args:
            llm_operation: Primary LLM operation to attempt
            fallback_operation: Fallback operation if LLM fails
            *args: Positional arguments
            session_id: Optional session identifier
            **kwargs: Keyword arguments
            
        Returns:
            Result from successful operation (LLM or fallback)
        """
        # Check circuit breaker before attempting LLM
        if not self.can_attempt_llm_classification():
            logger.info(f"Using fallback due to circuit breaker (session: {session_id or 'anonymous'})")
            
            fallback_start = time.time()
            result = await fallback_operation(*args, **kwargs)
            fallback_time = time.time() - fallback_start
            
            self.record_fallback_activation(fallback_time)
            return result
        
        # Attempt LLM with retry logic
        try:
            llm_start = time.time()
            result = await self.retry_manager.execute_with_retry(
                llm_operation, 
                *args, 
                session_id=session_id,
                **kwargs
            )
            llm_time = time.time() - llm_start
            
            self.circuit_breaker_manager.record_llm_success(llm_time)
            return result
            
        except Exception as e:
            self.circuit_breaker_manager.record_llm_failure()
            
            logger.warning(
                f"LLM operation failed after retries, using fallback: {e} "
                f"(session: {session_id or 'anonymous'})"
            )
            
            # Use fallback
            fallback_start = time.time()
            result = await fallback_operation(*args, **kwargs)
            fallback_time = time.time() - fallback_start
            
            self.record_fallback_activation(fallback_time)
            return result
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive fallback system metrics.
        
        Returns:
            Dictionary with all fallback system statistics
        """
        circuit_breaker_stats = self.circuit_breaker_manager.get_circuit_breaker_stats()
        retry_stats = self.retry_manager.get_retry_stats()
        
        # Combine metrics from all components
        combined_metrics = FallbackMetrics()
        combined_metrics.total_attempts = (
            self.metrics.total_attempts + 
            self.circuit_breaker_manager.metrics.total_attempts + 
            self.retry_manager.metrics.total_attempts
        )
        combined_metrics.llm_successes = (
            self.metrics.llm_successes + 
            self.circuit_breaker_manager.metrics.llm_successes
        )
        combined_metrics.llm_failures = (
            self.metrics.llm_failures + 
            self.circuit_breaker_manager.metrics.llm_failures
        )
        combined_metrics.circuit_breaker_blocks = (
            self.metrics.circuit_breaker_blocks + 
            self.circuit_breaker_manager.metrics.circuit_breaker_blocks
        )
        combined_metrics.retry_attempts = (
            self.metrics.retry_attempts + 
            self.retry_manager.metrics.retry_attempts
        )
        combined_metrics.fallback_activations = self.metrics.fallback_activations
        
        return {
            "circuit_breaker": circuit_breaker_stats,
            "retry_mechanism": retry_stats,
            "combined_metrics": {
                "total_attempts": combined_metrics.total_attempts,
                "llm_success_rate": combined_metrics.get_llm_success_rate(),
                "circuit_breaker_blocks": combined_metrics.circuit_breaker_blocks,
                "retry_attempts": combined_metrics.retry_attempts,
                "fallback_activations": combined_metrics.fallback_activations,
                "system_health": self._calculate_system_health(combined_metrics)
            }
        }
    
    def _calculate_system_health(self, metrics: FallbackMetrics) -> Dict[str, Any]:
        """
        Calculate overall system health metrics.
        
        Args:
            metrics: Combined fallback metrics
            
        Returns:
            System health assessment
        """
        llm_success_rate = metrics.get_llm_success_rate()
        circuit_breaker_state = self.circuit_breaker_manager.circuit_breaker.state
        
        # Determine health status
        if llm_success_rate > 0.9 and circuit_breaker_state == CircuitBreakerState.CLOSED:
            health_status = "excellent"
        elif llm_success_rate > 0.7 and circuit_breaker_state != CircuitBreakerState.OPEN:
            health_status = "good"
        elif llm_success_rate > 0.5:
            health_status = "degraded"
        else:
            health_status = "poor"
        
        return {
            "status": health_status,
            "llm_availability": llm_success_rate,
            "circuit_breaker_state": circuit_breaker_state.value,
            "fallback_dependency": metrics.fallback_activations / max(1, metrics.total_attempts),
            "recommendation": self._get_health_recommendation(health_status, llm_success_rate)
        }
    
    def _get_health_recommendation(self, health_status: str, llm_success_rate: float) -> str:
        """
        Get recommendation based on system health.
        
        Args:
            health_status: Current health status
            llm_success_rate: LLM success rate
            
        Returns:
            Health recommendation string
        """
        if health_status == "excellent":
            return "System operating optimally"
        elif health_status == "good":
            return "System stable with minor issues"
        elif health_status == "degraded":
            return "Monitor LLM service, consider scaling or maintenance"
        else:
            return "Critical: LLM service unavailable, operating on fallback only"
    
    def reset_metrics(self) -> None:
        """Reset all fallback metrics."""
        self.metrics = FallbackMetrics()
        self.circuit_breaker_manager.metrics = FallbackMetrics()
        self.retry_manager.metrics = FallbackMetrics()
        
        logger.info("Fallback system metrics reset")
