"""
Data structures and type definitions for query classification system.

This module contains all the data classes, enums, and type definitions used
throughout the query classification system. Centralizing these definitions
improves maintainability and provides clear contracts between modules.

Example Usage:
    # Create a classification result
    result = ClassificationResult(
        classification="SQL",
        confidence="HIGH",
        reasoning="High-confidence SQL patterns detected",
        processing_time=0.025,
        method_used="rule_based",
        pattern_matches={"sql_high": 3, "sql_medium": 1}
    )
    
    # Create query complexity analysis
    complexity = QueryComplexityAnalysis(
        word_count=8,
        keyword_density=0.375,
        domain_specificity_score=0.9,
        ambiguity_indicators=[],
        structural_complexity=0.6,
        overall_complexity_score=0.7
    )
    
    # Use confidence calibration result
    calibration = ConfidenceCalibrationResult(
        original_confidence="MEDIUM",
        calibrated_confidence="HIGH",
        adjustment_reasoning="High pattern strength and domain specificity",
        confidence_score=0.85,
        complexity_analysis=complexity
    )
    
    # Track classification statistics
    stats = ClassificationStatistics(
        total_attempts=100,
        successful_classifications=95,
        accuracy_by_confidence={"HIGH": 0.98, "MEDIUM": 0.92, "LOW": 0.75}
    )
"""

from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


# Type definitions
ClassificationType = Literal["SQL", "VECTOR", "HYBRID", "CLARIFICATION_NEEDED", "CONVERSATIONAL"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class QueryComplexityAnalysis:
    """Analysis of query complexity for confidence calibration."""
    word_count: int
    keyword_density: float
    domain_specificity_score: float
    ambiguity_indicators: List[str]
    structural_complexity: float
    overall_complexity_score: float


@dataclass
class ConfidenceCalibrationResult:
    """Result of confidence calibration with detailed reasoning."""
    original_confidence: ConfidenceLevel
    calibrated_confidence: ConfidenceLevel
    confidence_score: float  # 0.0 to 1.0
    calibration_factors: Dict[str, float]
    complexity_analysis: QueryComplexityAnalysis
    historical_accuracy: float
    adjustment_reasoning: str


@dataclass
class ClassificationResult:
    """
    Result of query classification with confidence and reasoning.
    
    Attributes:
        classification: The determined category (SQL, VECTOR, HYBRID, CLARIFICATION_NEEDED)
        confidence: Confidence level (HIGH, MEDIUM, LOW)
        reasoning: Explanation of classification decision
        processing_time: Time taken for classification in seconds
        method_used: Which classification method was used (rule_based, llm_based, fallback)
        anonymized_query: PII-anonymized version of the original query
        pattern_matches: Pattern match counts by confidence level (for rule-based)
        calibration_reasoning: Reasoning for confidence calibration adjustments
        feedback_table_suggestion: Suggested table for feedback queries (Phase 2 enhancement)
        feedback_confidence: Confidence in feedback table classification (Phase 2 enhancement)
    """
    classification: ClassificationType
    confidence: ConfidenceLevel
    reasoning: str
    processing_time: float
    method_used: Literal["rule_based", "llm_based", "fallback", "conversational"]
    anonymized_query: Optional[str] = None
    pattern_matches: Optional[Dict[str, int]] = None
    calibration_reasoning: Optional[str] = None
    feedback_table_suggestion: Optional[str] = None
    feedback_confidence: Optional[float] = None


class ClassificationMethod(Enum):
    """Enumeration of available classification methods."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    FALLBACK = "fallback"


class CircuitBreakerState(Enum):
    """Circuit breaker states for LLM failure management."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # LLM failures detected, bypassing LLM
    HALF_OPEN = "half_open"  # Testing if LLM is back online


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for LLM classification failures.
    
    Implements the circuit breaker pattern to prevent cascading failures
    when the LLM service is unavailable or performing poorly.
    """
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    half_open_max_calls: int = 3  # Max calls to test in half-open state
    
    # State tracking
    state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    half_open_attempts: int = field(default=0)
    
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to attempt recovery
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_attempts = 0
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_attempts < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Recovery successful, close the circuit
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            self.half_open_attempts = 0
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Half-open test failed, go back to open
            self.state = CircuitBreakerState.OPEN
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.failure_threshold):
            # Too many failures, open the circuit
            self.state = CircuitBreakerState.OPEN


@dataclass
class FallbackMetrics:
    """Metrics tracking for fallback system performance."""
    total_attempts: int = 0
    llm_successes: int = 0
    llm_failures: int = 0
    circuit_breaker_blocks: int = 0
    retry_attempts: int = 0
    fallback_activations: int = 0
    
    # Timing metrics
    classification_times: List[float] = field(default_factory=list)
    llm_response_times: List[float] = field(default_factory=list)
    fallback_response_times: List[float] = field(default_factory=list)
    
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
        """Record a circuit breaker intervention."""
        self.circuit_breaker_blocks += 1
    
    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_attempts += 1
    
    def record_fallback(self, response_time: float) -> None:
        """Record a fallback classification."""
        self.fallback_activations += 1
        self.fallback_response_times.append(response_time)
    
    def get_llm_success_rate(self) -> float:
        """Calculate LLM success rate."""
        total_llm_attempts = self.llm_successes + self.llm_failures
        if total_llm_attempts == 0:
            return 1.0
        return self.llm_successes / total_llm_attempts
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average response times for different methods."""
        def safe_average(times_list: List[float]) -> float:
            return sum(times_list) / len(times_list) if times_list else 0.0
        
        return {
            "classification": safe_average(self.classification_times),
            "llm_response": safe_average(self.llm_response_times),
            "fallback_response": safe_average(self.fallback_response_times)
        }


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt using exponential backoff.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        import random
        
        # Calculate exponential backoff
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            # Add up to 25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Ensure minimum delay
