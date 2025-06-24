"""
Confidence Calibration System for Query Classification

This module implements sophisticated confidence calibration with adaptive thresholds,
historical accuracy tracking, and multi-dimensional scoring for the RAG query classifier.

Key Features:
- Query complexity analysis (structural, semantic, domain-specific)
- Historical accuracy tracking per classification type
- Pattern strength and reliability metrics
- Contextual factors (APS domain specificity, ambiguity markers)
- Dynamic confidence threshold adjustment
- Real-time calibration statistics and monitoring

"""

from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
import re
import time

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type definitions
ClassificationType = Literal["SQL", "VECTOR", "HYBRID", "CLARIFICATION_NEEDED"]
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


class ConfidenceCalibrator:
    """
    Sophisticated confidence calibration system with adaptive thresholds.
    
    This system provides multi-dimensional confidence scoring based on:
    1. Query complexity analysis (structural, semantic, domain-specific)
    2. Historical accuracy tracking per classification type
    3. Pattern strength and reliability metrics
    4. Contextual factors (APS domain specificity, ambiguity markers)
    """
    
    def __init__(self):
        """Initialize confidence calibrator with tracking systems."""
        # Historical accuracy tracking
        self._accuracy_history = {
            "SQL": {"correct": 0, "total": 0, "recent_accuracy": []},
            "VECTOR": {"correct": 0, "total": 0, "recent_accuracy": []},
            "HYBRID": {"correct": 0, "total": 0, "recent_accuracy": []},
            "CLARIFICATION_NEEDED": {"correct": 0, "total": 0, "recent_accuracy": []}
        }
        
        # Confidence calibration weights
        self._calibration_weights = {
            "query_complexity": 0.25,
            "historical_accuracy": 0.30,
            "pattern_strength": 0.25,
            "domain_specificity": 0.20
        }
        
        # Dynamic confidence thresholds (can be adjusted based on performance)
        self._confidence_thresholds = {
            "HIGH": {"min": 0.80, "adaptive_adjustment": 0.0},
            "MEDIUM": {"min": 0.50, "adaptive_adjustment": 0.0},
            "LOW": {"min": 0.0, "adaptive_adjustment": 0.0}
        }
        
        # APS domain-specific indicators
        self._aps_domain_indicators = [
            "executive level", "el1", "el2", "aps", "agency", "department", 
            "portfolio", "capability framework", "professional development",
            "mandatory training", "compliance", "participant", "delegate",
            "facilitator", "virtual learning", "face-to-face", "blended"
        ]
        
        # Ambiguity markers that reduce confidence
        self._ambiguity_markers = [
            "maybe", "perhaps", "might be", "could be", "not sure",
            "unclear", "confusing", "ambiguous", "general", "broad",
            "anything", "something", "everything", "help", "assist"
        ]
        
        # Pattern strength indicators
        self._pattern_strength_cache = {}
    
    def analyze_query_complexity(self, query: str) -> QueryComplexityAnalysis:
        """
        Perform comprehensive query complexity analysis.
        
        Args:
            query: Query text to analyze
            
        Returns:
            QueryComplexityAnalysis with detailed complexity metrics
        """
        query_lower = query.lower()
        words = query.split()
        
        # Basic metrics
        word_count = len(words)
        
        # Keyword density (APS-specific terms / total words)
        aps_keywords_found = [indicator for indicator in self._aps_domain_indicators if indicator in query_lower]
        keyword_density = len(aps_keywords_found) / max(1, word_count)
        
        # Domain specificity score
        domain_specificity_score = min(1.0, len(aps_keywords_found) * 0.2)
        
        # Ambiguity indicators
        ambiguity_found = [marker for marker in self._ambiguity_markers if marker in query_lower]
        
        # Structural complexity (sentence structure, conjunctions, etc.)
        structural_indicators = ["and", "or", "but", "with", "across", "between", "compare", "analyze"]
        structural_complexity = len([word for word in words if word.lower() in structural_indicators]) / max(1, word_count)
        
        # Overall complexity score (0.0 = simple, 1.0 = very complex)
        complexity_factors = {
            "length": min(1.0, word_count / 20),  # Normalize to ~20 words max
            "keyword_density": keyword_density,
            "ambiguity": len(ambiguity_found) * 0.1,
            "structure": structural_complexity
        }
        
        overall_complexity_score = (
            complexity_factors["length"] * 0.2 +
            complexity_factors["ambiguity"] * 0.4 +  # High weight for ambiguity
            complexity_factors["structure"] * 0.3 +
            (1 - complexity_factors["keyword_density"]) * 0.1  # Less keywords = more complex
        )
        
        return QueryComplexityAnalysis(
            word_count=word_count,
            keyword_density=keyword_density,
            domain_specificity_score=domain_specificity_score,
            ambiguity_indicators=ambiguity_found,
            structural_complexity=structural_complexity,
            overall_complexity_score=min(1.0, overall_complexity_score)
        )
    
    def get_historical_accuracy(self, classification: ClassificationType) -> float:
        """
        Get historical accuracy for a classification type.
        
        Args:
            classification: Classification type to get accuracy for
            
        Returns:
            Historical accuracy as float (0.0 to 1.0)
        """
        history = self._accuracy_history.get(classification, {"correct": 0, "total": 0})
        
        if history["total"] == 0:
            return 0.85  # Default optimistic accuracy for new classifications
        
        base_accuracy = history["correct"] / history["total"]
        
        # Use recent accuracy if available (more weight on recent performance)
        recent_accuracy = history.get("recent_accuracy", [])
        if recent_accuracy:
            recent_avg = sum(recent_accuracy[-10:]) / len(recent_accuracy[-10:])  # Last 10 results
            # Weighted average: 70% recent, 30% historical
            return base_accuracy * 0.3 + recent_avg * 0.7
        
        return base_accuracy
    
    def calculate_pattern_strength(self, query: str, classification: ClassificationType, pattern_matches: Dict[str, int]) -> float:
        """
        Calculate pattern strength based on matched patterns and their reliability.
        
        Args:
            query: Query text
            classification: Predicted classification
            pattern_matches: Dictionary of pattern match counts by confidence level
            
        Returns:
            Pattern strength score (0.0 to 1.0)
        """
        # Weight pattern matches by confidence level
        high_weight = pattern_matches.get("high_confidence", 0) * 3
        medium_weight = pattern_matches.get("medium_confidence", 0) * 2
        low_weight = pattern_matches.get("low_confidence", 0) * 1
        
        total_weighted_score = high_weight + medium_weight + low_weight
        
        # Normalize based on query length (longer queries should have more patterns)
        query_length_factor = len(query.split()) / 10  # Normalize to ~10 words
        normalized_score = total_weighted_score / max(1, query_length_factor)
        
        # Cap at 1.0 and apply diminishing returns for very high scores
        pattern_strength = min(1.0, normalized_score / 5)  # Divide by 5 for realistic scaling
        
        return pattern_strength
    
    def calibrate_confidence(
        self,
        raw_confidence: ConfidenceLevel,
        classification: ClassificationType,
        query: str,
        pattern_matches: Optional[Dict[str, int]] = None,
        method_used: str = "unknown"
    ) -> ConfidenceCalibrationResult:
        """
        Perform sophisticated confidence calibration with multi-dimensional analysis.
        
        Args:
            raw_confidence: Original confidence level
            classification: Predicted classification
            query: Query text
            pattern_matches: Pattern match counts by confidence level
            method_used: Classification method used
            
        Returns:
            ConfidenceCalibrationResult with calibrated confidence and detailed analysis
        """
        # Step 1: Analyze query complexity
        complexity_analysis = self.analyze_query_complexity(query)
        
        # Step 2: Get historical accuracy
        historical_accuracy = self.get_historical_accuracy(classification)
        
        # Step 3: Calculate pattern strength
        pattern_strength = 0.5  # Default for non-rule-based methods
        if pattern_matches and method_used == "rule_based":
            pattern_strength = self.calculate_pattern_strength(query, classification, pattern_matches)
        
        # Step 4: Convert raw confidence to numeric score
        confidence_to_score = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.3}
        base_score = confidence_to_score.get(raw_confidence, 0.5)
        
        # Step 5: Apply calibration factors
        calibration_factors = {
            "base_confidence": base_score,
            "complexity_penalty": -complexity_analysis.overall_complexity_score * 0.2,
            "historical_boost": (historical_accuracy - 0.5) * 0.3,  # Boost/penalty based on history
            "pattern_strength_boost": (pattern_strength - 0.5) * 0.2,
            "domain_specificity_boost": complexity_analysis.domain_specificity_score * 0.15,
            "ambiguity_penalty": -len(complexity_analysis.ambiguity_indicators) * 0.1
        }
        
        # Calculate weighted calibrated score
        calibrated_score = base_score
        for factor, weight in self._calibration_weights.items():
            if factor == "query_complexity":
                calibrated_score += calibration_factors["complexity_penalty"] * weight
            elif factor == "historical_accuracy":
                calibrated_score += calibration_factors["historical_boost"] * weight
            elif factor == "pattern_strength":
                calibrated_score += calibration_factors["pattern_strength_boost"] * weight
            elif factor == "domain_specificity":
                calibrated_score += calibration_factors["domain_specificity_boost"] * weight
        
        # Apply ambiguity penalty (always reduces confidence)
        calibrated_score += calibration_factors["ambiguity_penalty"]
        
        # Ensure score stays within bounds
        calibrated_score = max(0.0, min(1.0, calibrated_score))
        
        # Step 6: Convert back to confidence level with adaptive thresholds
        high_threshold = self._confidence_thresholds["HIGH"]["min"] + self._confidence_thresholds["HIGH"]["adaptive_adjustment"]
        medium_threshold = self._confidence_thresholds["MEDIUM"]["min"] + self._confidence_thresholds["MEDIUM"]["adaptive_adjustment"]
        
        if calibrated_score >= high_threshold:
            calibrated_confidence = "HIGH"
        elif calibrated_score >= medium_threshold:
            calibrated_confidence = "MEDIUM"
        else:
            calibrated_confidence = "LOW"
        
        # Step 7: Generate adjustment reasoning
        adjustments = []
        if calibration_factors["complexity_penalty"] < -0.05:
            adjustments.append(f"reduced due to query complexity ({complexity_analysis.overall_complexity_score:.2f})")
        if calibration_factors["historical_boost"] > 0.05:
            adjustments.append(f"increased due to good historical accuracy ({historical_accuracy:.2f})")
        elif calibration_factors["historical_boost"] < -0.05:
            adjustments.append(f"reduced due to poor historical accuracy ({historical_accuracy:.2f})")
        if calibration_factors["pattern_strength_boost"] > 0.05:
            adjustments.append(f"increased due to strong pattern matches ({pattern_strength:.2f})")
        if complexity_analysis.ambiguity_indicators:
            adjustments.append(f"reduced due to ambiguity indicators: {', '.join(complexity_analysis.ambiguity_indicators[:3])}")
        if complexity_analysis.domain_specificity_score > 0.5:
            adjustments.append(f"increased due to APS domain specificity ({complexity_analysis.domain_specificity_score:.2f})")
        
        adjustment_reasoning = f"Confidence {raw_confidence} → {calibrated_confidence} ({calibrated_score:.3f})"
        if adjustments:
            adjustment_reasoning += f": {'; '.join(adjustments)}"
        else:
            adjustment_reasoning += ": no significant adjustments needed"
        
        return ConfidenceCalibrationResult(
            original_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_score=calibrated_score,
            calibration_factors=calibration_factors,
            complexity_analysis=complexity_analysis,
            historical_accuracy=historical_accuracy,
            adjustment_reasoning=adjustment_reasoning
        )
    
    def record_classification_outcome(
        self,
        classification: ClassificationType,
        was_correct: bool,
        confidence_score: float
    ) -> None:
        """
        Record the outcome of a classification for historical accuracy tracking.
        
        Args:
            classification: The classification that was made
            was_correct: Whether the classification was correct
            confidence_score: The confidence score that was assigned
        """
        if classification not in self._accuracy_history:
            self._accuracy_history[classification] = {"correct": 0, "total": 0, "recent_accuracy": []}
        
        history = self._accuracy_history[classification]
        history["total"] += 1
        
        if was_correct:
            history["correct"] += 1
            history["recent_accuracy"].append(1.0)
        else:
            history["recent_accuracy"].append(0.0)
        
        # Keep only recent history (last 50 results)
        if len(history["recent_accuracy"]) > 50:
            history["recent_accuracy"] = history["recent_accuracy"][-50:]
        
        logger.debug(
            f"Recorded classification outcome: {classification} "
            f"({'correct' if was_correct else 'incorrect'}), "
            f"historical accuracy: {history['correct']}/{history['total']} "
            f"({(history['correct']/history['total']*100):.1f}%)"
        )
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive calibration statistics for monitoring.
        
        Returns:
            Dictionary with calibration system statistics
        """
        stats = {
            "accuracy_by_classification": {},
            "calibration_weights": self._calibration_weights,
            "confidence_thresholds": self._confidence_thresholds,
            "total_classifications": 0
        }
        
        for classification, history in self._accuracy_history.items():
            total = history["total"]
            correct = history["correct"]
            recent_accuracy = history.get("recent_accuracy", [])
            
            stats["accuracy_by_classification"][classification] = {
                "total": total,
                "correct": correct,
                "accuracy": (correct / total * 100) if total > 0 else 0,
                "recent_accuracy": (sum(recent_accuracy[-10:]) / len(recent_accuracy[-10:]) * 100) if recent_accuracy else 0,
                "sample_size": len(recent_accuracy)
            }
            
            stats["total_classifications"] += total
        
        return stats
    
    def adjust_confidence_thresholds(self, performance_metrics: Dict[str, float]) -> None:
        """
        Dynamically adjust confidence thresholds based on system performance.
        
        Args:
            performance_metrics: Performance metrics to base adjustments on
        """
        # Simple adaptive threshold adjustment
        # If accuracy is consistently high, we can be more confident
        # If accuracy is low, we should be more conservative
        
        overall_accuracy = performance_metrics.get("overall_accuracy", 0.8)
        
        if overall_accuracy > 0.9:
            # High accuracy - can afford to be more confident
            self._confidence_thresholds["HIGH"]["adaptive_adjustment"] = -0.05
            self._confidence_thresholds["MEDIUM"]["adaptive_adjustment"] = -0.05
        elif overall_accuracy < 0.7:
            # Low accuracy - be more conservative
            self._confidence_thresholds["HIGH"]["adaptive_adjustment"] = 0.05
            self._confidence_thresholds["MEDIUM"]["adaptive_adjustment"] = 0.05
        else:
            # Reset adjustments for normal performance
            self._confidence_thresholds["HIGH"]["adaptive_adjustment"] = 0.0
            self._confidence_thresholds["MEDIUM"]["adaptive_adjustment"] = 0.0
        
        logger.info(f"Adjusted confidence thresholds based on accuracy {overall_accuracy:.3f}")
    
    def reset_calibration_data(self) -> None:
        """Reset all calibration data for testing or fresh start."""
        self._accuracy_history = {
            "SQL": {"correct": 0, "total": 0, "recent_accuracy": []},
            "VECTOR": {"correct": 0, "total": 0, "recent_accuracy": []},
            "HYBRID": {"correct": 0, "total": 0, "recent_accuracy": []},
            "CLARIFICATION_NEEDED": {"correct": 0, "total": 0, "recent_accuracy": []}
        }
        self._pattern_strength_cache = {}
        
        # Reset adaptive adjustments
        for confidence_level in self._confidence_thresholds:
            self._confidence_thresholds[confidence_level]["adaptive_adjustment"] = 0.0
        
        logger.info("Confidence calibration data reset")


# Demo and testing functions for Milestone 3
async def demonstrate_confidence_calibration():
    """
    Demonstrate confidence calibration capabilities with APS-specific examples.
    """
    print("=== Milestone 3: Confidence Calibration System Demo ===\n")
    
    calibrator = ConfidenceCalibrator()
    
    # Test queries with different complexity levels
    test_queries = [
        # Simple, domain-specific queries
        ("How many EL1 staff completed training?", "SQL", "HIGH"),
        ("What feedback did participants provide?", "VECTOR", "MEDIUM"),
        
        # Complex queries with ambiguity
        ("Maybe analyze some training effectiveness stuff?", "HYBRID", "LOW"),
        ("Could you perhaps help with general training information?", "CLARIFICATION_NEEDED", "LOW"),
        
        # Domain-specific but complex
        ("Correlate Executive Level 2 completion rates with participant satisfaction feedback across all agencies", "HYBRID", "MEDIUM"),
    ]
    
    print("1. Testing Query Complexity Analysis:")
    for query, classification, expected_confidence in test_queries:
        complexity = calibrator.analyze_query_complexity(query)
        
        print(f"  Query: {query}")
        print(f"  Word count: {complexity.word_count}")
        print(f"  Domain specificity: {complexity.domain_specificity_score:.2f}")
        print(f"  Ambiguity indicators: {complexity.ambiguity_indicators}")
        print(f"  Overall complexity: {complexity.overall_complexity_score:.2f}")
        print()
    
    print("2. Testing Confidence Calibration:")
    for query, classification, raw_confidence in test_queries:
        # Simulate pattern matches for rule-based classifications
        pattern_matches = {"high_confidence": 1, "medium_confidence": 0, "low_confidence": 0} if classification == "SQL" else None
        
        result = calibrator.calibrate_confidence(
            raw_confidence=raw_confidence,
            classification=classification,
            query=query,
            pattern_matches=pattern_matches,
            method_used="rule_based" if pattern_matches else "llm_based"
        )
        
        print(f"  Query: {query}")
        print(f"  Original confidence: {result.original_confidence}")
        print(f"  Calibrated confidence: {result.calibrated_confidence}")
        print(f"  Confidence score: {result.confidence_score:.3f}")
        print(f"  Adjustment reasoning: {result.adjustment_reasoning}")
        print()
    
    print("3. Testing Historical Accuracy Tracking:")
    
    # Simulate some classification outcomes
    test_outcomes = [
        ("SQL", True, 0.9),
        ("SQL", True, 0.8),
        ("SQL", False, 0.7),
        ("VECTOR", True, 0.8),
        ("VECTOR", True, 0.9),
        ("HYBRID", False, 0.6),
    ]
    
    for classification, was_correct, confidence_score in test_outcomes:
        calibrator.record_classification_outcome(classification, was_correct, confidence_score)
    
    # Show calibration statistics
    stats = calibrator.get_calibration_stats()
    print("  Calibration Statistics:")
    for classification, accuracy_data in stats["accuracy_by_classification"].items():
        if accuracy_data["total"] > 0:
            print(f"    {classification}: {accuracy_data['accuracy']:.1f}% accuracy ({accuracy_data['correct']}/{accuracy_data['total']})")
    
    print("\n4. Testing Adaptive Threshold Adjustment:")
    
    # Test threshold adjustment with different performance levels
    calibrator.adjust_confidence_thresholds({"overall_accuracy": 0.95})  # High performance
    print("  High performance (95% accuracy) - thresholds lowered for more confidence")
    
    calibrator.adjust_confidence_thresholds({"overall_accuracy": 0.65})  # Low performance  
    print("  Low performance (65% accuracy) - thresholds raised for more conservative confidence")
    
    calibrator.adjust_confidence_thresholds({"overall_accuracy": 0.80})  # Normal performance
    print("  Normal performance (80% accuracy) - thresholds reset to defaults")


if __name__ == "__main__":
    """Test the confidence calibration system."""
    import asyncio
    
    print("Starting Confidence Calibration System Demo...")
    print("=" * 60)
    
    asyncio.run(demonstrate_confidence_calibration())
    
    print("\n" + "=" * 60)
    print("Milestone 3: Confidence Calibration System - IMPLEMENTED ✅")
