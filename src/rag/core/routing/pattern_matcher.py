"""
Rule-based pattern matching for query classification.

This module provides sophisticated rule-based classification using
weighted regex patterns specific to the Australian Public Service
learning analytics domain.

Example Usage:
    # Initialize pattern matcher
    matcher = PatternMatcher()
    
    # Classify queries using APS-specific patterns
    result = matcher.classify_query("How many Level 6 users completed training?")
    if result:
        print(f"Classification: {result.classification}")  # SQL
        print(f"Confidence: {result.confidence}")          # HIGH
        print(f"Reasoning: {result.reasoning}")            # Pattern match details
    
    # Check feedback-related queries
    result = matcher.classify_query("What did participants say about the course?")
    if result:
        print(f"Classification: {result.classification}")  # VECTOR
        print(f"Confidence: {result.confidence}")          # HIGH
    
    # Test hybrid analytical queries
    result = matcher.classify_query("Analyze satisfaction trends across agencies")
    if result:
        print(f"Classification: {result.classification}")  # HYBRID
        print(f"Confidence: {result.confidence}")          # MEDIUM/HIGH
    
    # Get pattern statistics
    stats = matcher.get_pattern_stats()
    print(f"Total patterns loaded: {stats['total_patterns']}")
    print(f"SQL patterns: {stats['sql_patterns']}")
    print(f"VECTOR patterns: {stats['vector_patterns']}")
"""

import re
from typing import Dict, Any, Optional
from .data_structures import ClassificationResult, ClassificationType
from .aps_patterns import aps_patterns, aps_pattern_weights
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class PatternMatcher:
    """
    Enhanced rule-based pattern matcher with APS domain knowledge.
    
    Uses weighted regex patterns to provide fast and accurate classification
    for obvious queries, serving as the first stage in the classification pipeline.
    """
    
    def __init__(self):
        """Initialize pattern matcher with APS-specific patterns."""
        # Get patterns from APS patterns module
        self.compiled_patterns = aps_patterns.compiled_patterns
        self.compiled_weighted_patterns = aps_pattern_weights.get_all_weighted_patterns()
        
        # Pattern statistics for monitoring
        self.pattern_usage_stats = {
            "SQL": {"high": 0, "medium": 0, "low": 0},
            "VECTOR": {"high": 0, "medium": 0, "low": 0},
            "HYBRID": {"high": 0, "medium": 0, "low": 0}
        }
        
        self.total_classifications = 0
        self.successful_classifications = 0
    
    def classify_query(self, query: str) -> Optional[ClassificationResult]:
        """
        Perform enhanced rule-based classification using weighted regex patterns.
        
        Uses pattern weighting system to provide more accurate confidence scoring
        based on Australian Public Service domain knowledge.
        
        Args:
            query: Query text to classify
            
        Returns:
            ClassificationResult if confident match found, None otherwise
        """
        self.total_classifications += 1
        query_lower = query.lower()
        
        # Calculate weighted scores for each category
        weighted_scores = {}
        pattern_details = {}
        
        for category in ["SQL", "VECTOR", "HYBRID"]:
            category_score = 0
            matched_patterns = []
            
            # High confidence patterns (weight: 3)
            high_patterns = self.compiled_weighted_patterns[category].get("high_confidence", [])
            for pattern in high_patterns:
                if pattern.search(query_lower):
                    category_score += 3
                    matched_patterns.append(("high", pattern.pattern))
                    self.pattern_usage_stats[category]["high"] += 1
            
            # Medium confidence patterns (weight: 2)
            medium_patterns = self.compiled_weighted_patterns[category].get("medium_confidence", [])
            for pattern in medium_patterns:
                if pattern.search(query_lower):
                    category_score += 2
                    matched_patterns.append(("medium", pattern.pattern))
                    self.pattern_usage_stats[category]["medium"] += 1
            
            # Low confidence patterns (weight: 1)
            low_patterns = self.compiled_weighted_patterns[category].get("low_confidence", [])
            for pattern in low_patterns:
                if pattern.search(query_lower):
                    category_score += 1
                    matched_patterns.append(("low", pattern.pattern))
                    self.pattern_usage_stats[category]["low"] += 1
            
            weighted_scores[category] = category_score
            pattern_details[category] = matched_patterns
        
        # Find the category with highest weighted score
        max_score = max(weighted_scores.values())
        
        if max_score == 0:
            logger.debug(f"No pattern matches found for query: {query[:50]}...")
            return None  # No pattern matches found
        
        # Find categories with maximum score
        top_categories = [cat for cat, score in weighted_scores.items() if score == max_score]
        
        if len(top_categories) > 1:
            logger.debug(f"Ambiguous pattern matching - multiple categories tied: {top_categories}")
            return None  # Ambiguous - multiple categories tied
        
        classification = top_categories[0]
        
        # Enhanced confidence determination based on weighted scores and pattern mix
        matched_patterns = pattern_details[classification]
        high_count = sum(1 for level, _ in matched_patterns if level == "high")
        medium_count = sum(1 for level, _ in matched_patterns if level == "medium")
        low_count = sum(1 for level, _ in matched_patterns if level == "low")
        
        # Prioritize presence of high-confidence patterns
        if high_count > 0:
            confidence = "HIGH"  # Any high-confidence pattern should give HIGH confidence
        elif max_score >= 6:  # Multiple medium-confidence patterns
            confidence = "HIGH"
        elif max_score >= 3:  # At least one medium-confidence pattern or multiple low
            confidence = "MEDIUM"
        elif max_score >= 2:  # Single medium or multiple low confidence patterns
            confidence = "MEDIUM"
        else:  # Only low confidence patterns
            confidence = "LOW"
        
        reasoning_parts = []
        if high_count > 0:
            reasoning_parts.append(f"{high_count} high-confidence")
        if medium_count > 0:
            reasoning_parts.append(f"{medium_count} medium-confidence")
        if low_count > 0:
            reasoning_parts.append(f"{low_count} low-confidence")
        
        reasoning = f"Enhanced rule-based: {', '.join(reasoning_parts)} pattern(s) for {classification} (score: {max_score})"
        
        # Calculate pattern match counts for confidence calibration
        pattern_match_counts = {
            "high_confidence": high_count,
            "medium_confidence": medium_count,
            "low_confidence": low_count
        }
        
        self.successful_classifications += 1
        
        logger.debug(
            f"Pattern match successful: {classification} with {confidence} confidence "
            f"(score: {max_score}, patterns: {reasoning_parts})"
        )
        
        return ClassificationResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=0.0,  # Will be set by caller
            method_used="rule_based",
            pattern_matches=pattern_match_counts
        )
    
    def get_pattern_coverage_analysis(self, queries: list) -> Dict[str, Any]:
        """
        Analyze pattern coverage across a set of queries.
        
        Args:
            queries: List of query strings to analyze
            
        Returns:
            Dictionary with pattern coverage statistics
        """
        coverage_stats = {
            "total_queries": len(queries),
            "matched_queries": 0,
            "unmatched_queries": [],
            "category_coverage": {"SQL": 0, "VECTOR": 0, "HYBRID": 0},
            "confidence_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "pattern_utilization": {}
        }
        
        # Initialize pattern utilization tracking
        for category in ["SQL", "VECTOR", "HYBRID"]:
            coverage_stats["pattern_utilization"][category] = {
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0
            }
        
        # Analyze each query
        for query in queries:
            result = self.classify_query(query)
            
            if result:
                coverage_stats["matched_queries"] += 1
                coverage_stats["category_coverage"][result.classification] += 1
                coverage_stats["confidence_distribution"][result.confidence] += 1
                
                # Track pattern utilization
                if result.pattern_matches:
                    for confidence_level, count in result.pattern_matches.items():
                        if count > 0:
                            coverage_stats["pattern_utilization"][result.classification][confidence_level] += 1
            else:
                coverage_stats["unmatched_queries"].append(query[:100])  # Truncate for readability
        
        # Calculate percentages
        total = len(queries)
        if total > 0:
            coverage_stats["match_rate"] = coverage_stats["matched_queries"] / total
            
            for category in coverage_stats["category_coverage"]:
                coverage_stats["category_coverage"][category] = coverage_stats["category_coverage"][category] / total
            
            for confidence in coverage_stats["confidence_distribution"]:
                coverage_stats["confidence_distribution"][confidence] = coverage_stats["confidence_distribution"][confidence] / total
        
        return coverage_stats
    
    def get_pattern_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pattern matching performance statistics.
        
        Returns:
            Dictionary with pattern matching statistics
        """
        success_rate = self.successful_classifications / max(1, self.total_classifications)
        
        # Calculate pattern utilization rates
        pattern_utilization = {}
        for category in self.pattern_usage_stats:
            total_category_usage = sum(self.pattern_usage_stats[category].values())
            pattern_utilization[category] = {
                "total_usage": total_category_usage,
                "high_confidence_rate": self.pattern_usage_stats[category]["high"] / max(1, total_category_usage),
                "medium_confidence_rate": self.pattern_usage_stats[category]["medium"] / max(1, total_category_usage),
                "low_confidence_rate": self.pattern_usage_stats[category]["low"] / max(1, total_category_usage)
            }
        
        # Pattern efficiency analysis
        most_used_category = max(self.pattern_usage_stats.keys(), 
                                key=lambda k: sum(self.pattern_usage_stats[k].values()))
        
        return {
            "total_classifications": self.total_classifications,
            "successful_classifications": self.successful_classifications,
            "success_rate": success_rate,
            "pattern_utilization": pattern_utilization,
            "most_active_category": most_used_category,
            "pattern_counts": {
                "SQL": len(aps_patterns.sql_patterns),
                "VECTOR": len(aps_patterns.vector_patterns),
                "HYBRID": len(aps_patterns.hybrid_patterns)
            },
            "weighted_pattern_counts": {
                category: {
                    confidence: len(patterns)
                    for confidence, patterns in aps_pattern_weights.get_weighted_patterns_for_category(category).items()
                }
                for category in ["SQL", "VECTOR", "HYBRID"]
            }
        }
    
    def validate_patterns(self) -> Dict[str, Any]:
        """
        Validate pattern compilation and detect potential issues.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "compiled_patterns_valid": True,
            "weighted_patterns_valid": True,
            "pattern_compilation_errors": [],
            "potential_issues": [],
            "recommendations": []
        }
        
        # Validate basic patterns
        try:
            for category, patterns in self.compiled_patterns.items():
                for i, pattern in enumerate(patterns):
                    if not hasattr(pattern, 'search'):
                        validation_results["compiled_patterns_valid"] = False
                        validation_results["pattern_compilation_errors"].append(
                            f"{category} pattern {i}: not properly compiled"
                        )
        except Exception as e:
            validation_results["compiled_patterns_valid"] = False
            validation_results["pattern_compilation_errors"].append(f"Pattern validation error: {e}")
        
        # Validate weighted patterns
        try:
            for category in ["SQL", "VECTOR", "HYBRID"]:
                weighted_patterns = self.compiled_weighted_patterns.get(category, {})
                for confidence_level in ["high_confidence", "medium_confidence", "low_confidence"]:
                    patterns = weighted_patterns.get(confidence_level, [])
                    for i, pattern in enumerate(patterns):
                        if not hasattr(pattern, 'search'):
                            validation_results["weighted_patterns_valid"] = False
                            validation_results["pattern_compilation_errors"].append(
                                f"{category} {confidence_level} pattern {i}: not properly compiled"
                            )
        except Exception as e:
            validation_results["weighted_patterns_valid"] = False
            validation_results["pattern_compilation_errors"].append(f"Weighted pattern validation error: {e}")
        
        # Check for potential issues
        self._check_pattern_overlaps(validation_results)
        self._check_pattern_coverage(validation_results)
        
        # Generate recommendations
        if validation_results["pattern_compilation_errors"]:
            validation_results["recommendations"].append("Fix pattern compilation errors before deployment")
        
        if validation_results["potential_issues"]:
            validation_results["recommendations"].append("Review identified pattern issues for optimization")
        
        if not validation_results["potential_issues"] and not validation_results["pattern_compilation_errors"]:
            validation_results["recommendations"].append("Pattern validation passed - system ready for production")
        
        return validation_results
    
    def _check_pattern_overlaps(self, validation_results: Dict[str, Any]) -> None:
        """Check for potential pattern overlaps that could cause ambiguity."""
        # This is a simplified check - in practice, you might want more sophisticated overlap detection
        test_queries = [
            "count the number of users",  # Should clearly match SQL
            "what did people say about the course",  # Should clearly match VECTOR
            "analyze satisfaction trends across agencies"  # Should clearly match HYBRID
        ]
        
        for query in test_queries:
            matches = []
            for category in ["SQL", "VECTOR", "HYBRID"]:
                patterns = self.compiled_patterns.get(category, [])
                if any(pattern.search(query.lower()) for pattern in patterns):
                    matches.append(category)
            
            if len(matches) > 1:
                validation_results["potential_issues"].append(
                    f"Query '{query}' matches multiple categories: {matches}"
                )
    
    def _check_pattern_coverage(self, validation_results: Dict[str, Any]) -> None:
        """Check pattern coverage for common query types."""
        # Check if we have reasonable coverage for each category
        for category in ["SQL", "VECTOR", "HYBRID"]:
            pattern_count = len(self.compiled_patterns.get(category, []))
            weighted_pattern_count = sum(
                len(patterns) 
                for patterns in self.compiled_weighted_patterns.get(category, {}).values()
            )
            
            if pattern_count < 5:
                validation_results["potential_issues"].append(
                    f"Low pattern count for {category}: {pattern_count} patterns"
                )
            
            if weighted_pattern_count < pattern_count:
                validation_results["potential_issues"].append(
                    f"Weighted patterns incomplete for {category}: {weighted_pattern_count}/{pattern_count}"
                )
    
    def reset_statistics(self) -> None:
        """Reset pattern matching statistics."""
        self.pattern_usage_stats = {
            "SQL": {"high": 0, "medium": 0, "low": 0},
            "VECTOR": {"high": 0, "medium": 0, "low": 0},
            "HYBRID": {"high": 0, "medium": 0, "low": 0}
        }
        self.total_classifications = 0
        self.successful_classifications = 0
        
        logger.info("Pattern matcher statistics reset")
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded patterns.
        
        Returns:
            Dictionary with pattern counts for monitoring
        """
        stats = {
            "sql_patterns": len(self.compiled_patterns.get("SQL", [])),
            "vector_patterns": len(self.compiled_patterns.get("VECTOR", [])),
            "hybrid_patterns": len(self.compiled_patterns.get("HYBRID", [])),
            "total_patterns": sum(len(patterns) for patterns in self.compiled_patterns.values()),
            "weighted_patterns": len(self.compiled_weighted_patterns),
            "usage_stats": self.pattern_usage_stats,
            "total_classifications": self.total_classifications,
            "successful_classifications": self.successful_classifications
        }
        return stats
