"""
Circuit Breaker Pattern Implementation for Query Classification.

This module implements the Circuit Breaker pattern to provide resilience
and fault tolerance for LLM-based query classification. It includes
retry logic, exponential backoff, and comprehensive metrics collection.
"""

import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Base for exponential backoff
    jitter: bool = True  # Add random jitter to prevent thundering herd
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt with exponential backoff.
        
        Args:
            attempt: The attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter if enabled
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class CircuitBreaker:
    """
    Circuit breaker implementation for LLM classification protection.
    
    Implements the Circuit Breaker pattern to prevent cascading failures
    when the LLM service is experiencing issues. Provides automatic
    recovery and graceful degradation.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opened_count = 0
        
        logger.info(f"Circuit breaker initialized: threshold={failure_threshold}, recovery_timeout={recovery_timeout}s")
    
    def can_execute(self) -> bool:
        """
        Check if circuit breaker allows execution.
        
        Returns:
            True if execution is allowed, False otherwise
        """
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time.timestamp() >= self.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.total_calls += 1
        self.successful_calls += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            # If we've had enough successful calls, close the circuit
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.total_calls += 1
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open state immediately opens circuit
            self.state = CircuitBreakerState.OPEN
            self.circuit_opened_count += 1
            logger.warning("Circuit breaker opened due to failure in HALF_OPEN state")
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if we've exceeded failure threshold
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.circuit_opened_count += 1
                logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": (self.successful_calls / max(1, self.total_calls)) * 100,
            "circuit_opened_count": self.circuit_opened_count,
            "half_open_calls": self.half_open_calls if self.state == CircuitBreakerState.HALF_OPEN else 0
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset to CLOSED state")


class FallbackMetrics:
    """
    Comprehensive metrics collection for fallback system monitoring.
    
    Tracks various aspects of the classification system's resilience
    and performance, including retry attempts, circuit breaker activations,
    and response times.
    """
    
    def __init__(self):
        """Initialize fallback metrics collection."""
        # Attempt tracking
        self.total_attempts = 0
        self.llm_successes = 0
        self.llm_failures = 0
        
        # Circuit breaker metrics
        self.circuit_breaker_blocks = 0
        
        # Retry metrics
        self.retry_attempts = 0
        
        # Fallback activation metrics
        self.fallback_activations = 0
        
        # Performance metrics
        self.classification_times: List[float] = []
        self.llm_response_times: List[float] = []
        self.retry_times: List[float] = []
        
        # Time tracking
        self.start_time = time.time()
        
        logger.info("Fallback metrics initialized")
    
    def record_attempt(self) -> None:
        """Record a classification attempt."""
        self.total_attempts += 1
    
    def record_llm_success(self, response_time: float) -> None:
        """Record a successful LLM classification."""
        self.llm_successes += 1
        self.llm_response_times.append(response_time)
    
    def record_llm_failure(self) -> None:
        """Record a failed LLM classification."""
        self.llm_failures += 1
    
    def record_circuit_breaker_block(self) -> None:
        """Record a circuit breaker block."""
        self.circuit_breaker_blocks += 1
    
    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_attempts += 1
    
    def record_fallback(self, fallback_time: float) -> None:
        """Record a fallback activation."""
        self.fallback_activations += 1
        self.retry_times.append(fallback_time)
    
    def get_llm_success_rate(self) -> float:
        """Get LLM success rate as percentage."""
        total_llm_attempts = self.llm_successes + self.llm_failures
        if total_llm_attempts == 0:
            return 100.0
        return (self.llm_successes / total_llm_attempts) * 100
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average response times for different operations."""
        return {
            "average_classification_time": (
                sum(self.classification_times) / len(self.classification_times)
                if self.classification_times else 0.0
            ),
            "average_llm_response_time": (
                sum(self.llm_response_times) / len(self.llm_response_times)
                if self.llm_response_times else 0.0
            ),
            "average_retry_time": (
                sum(self.retry_times) / len(self.retry_times)
                if self.retry_times else 0.0
            )
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics statistics."""
        uptime = time.time() - self.start_time
        
        return {
            # Basic metrics
            "total_attempts": self.total_attempts,
            "llm_successes": self.llm_successes,
            "llm_failures": self.llm_failures,
            "llm_success_rate": self.get_llm_success_rate(),
            
            # Resilience metrics
            "circuit_breaker_blocks": self.circuit_breaker_blocks,
            "retry_attempts": self.retry_attempts,
            "fallback_activations": self.fallback_activations,
            
            # Performance metrics
            **self.get_average_times(),
            
            # System health
            "uptime_seconds": uptime,
            "system_healthy": self.llm_failures < 5 and self.circuit_breaker_blocks < 3,
            
            # Efficiency metrics
            "classification_efficiency": (
                (self.llm_successes / max(1, self.total_attempts)) * 100
            ),
            "retry_rate": (
                (self.retry_attempts / max(1, self.total_attempts)) * 100
            ),
            "fallback_rate": (
                (self.fallback_activations / max(1, self.total_attempts)) * 100
            )
        }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.total_attempts = 0
        self.llm_successes = 0
        self.llm_failures = 0
        self.circuit_breaker_blocks = 0
        self.retry_attempts = 0
        self.fallback_activations = 0
        
        self.classification_times.clear()
        self.llm_response_times.clear()
        self.retry_times.clear()
        
        self.start_time = time.time()
        
        logger.info("Fallback metrics reset")


async def retry_with_exponential_backoff(
    func,
    config: RetryConfig,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with exponential backoff retry logic.
    
    Args:
        func: Async function to retry
        config: Retry configuration
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):  # +1 for initial attempt
        try:
            return await func(*args, **kwargs)
        
        except Exception as e:
            last_exception = e
            
            if attempt == config.max_retries:
                # Last attempt failed
                logger.error(f"All {config.max_retries + 1} attempts failed: {e}")
                break
            
            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # Add jitter if enabled
            if config.jitter:
                import random
                delay = delay * (0.5 + random.random() * 0.5)
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    # Re-raise the last exception
    raise last_exception
