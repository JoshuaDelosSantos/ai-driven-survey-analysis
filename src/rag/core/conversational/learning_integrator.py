#!/usr/bin/env python3
"""
Conversational Learning Integrator - Phase 3: Learning Integration & Monitoring

This module bridges LLM performance data back to the existing learning system,
enabling the conversational intelligence to continuously improve its routing
decisions based on real performance feedback.

Key Features:
- Bridges LLM effectiveness data to existing PatternLearningData
- Updates routing preferences based on comparative performance
- Maintains existing learning system integrity
- Provides learning-driven threshold adjustment
- Enables continuous improvement of hybrid routing decisions

Component Reuse Strategy:
- Leverages existing PatternLearningData structure (100% reuse)
- Uses existing feedback mechanisms (100% reuse)
- Preserves existing success rate calculations (100% reuse)
- Enhances existing user satisfaction tracking (enhancement only)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Reuse existing imports
from .handler import ConversationalHandler, ConversationalPattern, PatternLearningData

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for learning feedback (avoiding circular import)."""
    TEMPLATE_ONLY = "template_only"
    VECTOR_ENHANCED = "vector_enhanced"
    LLM_ENHANCED = "llm_enhanced"
    TEMPLATE_FIRST = "template_first"
    FALLBACK = "fallback"


class LearningUpdateType(Enum):
    """Types of learning updates for tracking purposes."""
    TEMPLATE_SUCCESS = "template_success"
    TEMPLATE_FAILURE = "template_failure" 
    LLM_SUCCESS = "llm_success"
    LLM_FAILURE = "llm_failure"
    HYBRID_SUCCESS = "hybrid_success"
    HYBRID_FAILURE = "hybrid_failure"


@dataclass
class LearningFeedback:
    """Structured feedback for learning integration."""
    query: str
    pattern_type: ConversationalPattern
    routing_strategy: RoutingStrategy
    llm_used: bool
    was_helpful: bool
    response_time_ms: float
    confidence_score: float
    user_satisfaction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        return {
            'query_length': len(self.query.split()),
            'pattern_type': self.pattern_type.value,
            'routing_strategy': self.routing_strategy.value,
            'llm_used': self.llm_used,
            'was_helpful': self.was_helpful,
            'response_time_ms': self.response_time_ms,
            'confidence_score': self.confidence_score,
            'user_satisfaction': self.user_satisfaction,
            'timestamp': self.timestamp.isoformat()
        }


class ConversationalLearningIntegrator:
    """
    Bridges LLM performance back to existing learning system.
    
    This component enhances the existing learning system by feeding LLM
    performance data back into the pattern learning mechanisms, enabling
    the system to improve its routing decisions over time.
    
    Component Reuse:
    - Uses existing ConversationalHandler pattern learning system
    - Leverages existing PatternLearningData structure
    - Preserves existing feedback mechanisms
    - Enhances existing success rate calculations
    """
    
    def __init__(self, conversational_handler: ConversationalHandler):
        """
        Initialize learning integrator with existing handler.
        
        Args:
            conversational_handler: Existing ConversationalHandler instance
        """
        self.handler = conversational_handler  # Reuse existing system
        self.learning_history: List[LearningFeedback] = []
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        self.adaptation_rate = 0.2  # How quickly to adapt to new performance data
        
        # Learning thresholds (configurable)
        self.min_samples_for_preference = 5
        self.significant_difference_threshold = 0.15
        self.preference_stability_window = 10
        
        logger.info("Learning integrator initialized with existing handler")

    async def update_learning_with_feedback(self, feedback: LearningFeedback) -> None:
        """
        Update existing learning system with LLM performance feedback.
        
        This method integrates LLM performance data into the existing
        PatternLearningData structure, enabling continuous improvement
        of routing decisions.
        
        Args:
            feedback: Structured feedback about response performance
        """
        try:
            # Store feedback for analysis
            self.learning_history.append(feedback)
            
            # Create pattern key using existing handler methodology
            pattern_key = f"{feedback.pattern_type.value}_{len(feedback.query.split())}"
            
            # Get or create learning data using existing system
            if pattern_key not in self.handler.pattern_learning:
                # Initialize with existing handler method
                self._initialize_pattern_learning(pattern_key, feedback.pattern_type)
            
            pattern_data = self.handler.pattern_learning[pattern_key]
            
            # Update LLM-specific metrics
            await self._update_llm_metrics(pattern_data, feedback)
            
            # Update routing preferences based on comparative performance
            await self._update_routing_preferences(pattern_data, feedback)
            
            # Use existing feedback mechanism to update success rates
            method_used = "llm_enhanced" if feedback.llm_used else "template"
            pattern_data.update_success_rate(feedback.was_helpful, method_used)
            
            # Update performance cache for quick access
            self._update_performance_cache(pattern_key, feedback)
            
            logger.debug(f"Learning updated for pattern {pattern_key}: "
                        f"LLM used: {feedback.llm_used}, "
                        f"Helpful: {feedback.was_helpful}")
            
        except Exception as e:
            logger.error(f"Failed to update learning with feedback: {e}")
            raise

    def _initialize_pattern_learning(self, pattern_key: str, 
                                    pattern_type: ConversationalPattern) -> None:
        """Initialize new pattern learning data using existing handler structure."""
        # Create the pattern learning data directly if it doesn't exist
        from datetime import datetime
        if pattern_key not in self.handler.pattern_learning:
            self.handler.pattern_learning[pattern_key] = PatternLearningData(
                pattern=pattern_key,
                frequency=1,
                success_rate=0.8,
                last_used=datetime.now(),
                feedback_scores=[],
                template_effectiveness={},
                context_success={},
                user_satisfaction=0.8
            )

    async def _update_llm_metrics(self, pattern_data: PatternLearningData, 
                                feedback: LearningFeedback) -> None:
        """Update LLM-specific performance metrics."""
        if feedback.llm_used:
            pattern_data.llm_usage_count += 1
            
            # Update LLM effectiveness using exponential moving average
            current_effectiveness = 1.0 if feedback.was_helpful else 0.0
            pattern_data.llm_effectiveness = (
                pattern_data.llm_effectiveness * (1 - self.adaptation_rate) +
                current_effectiveness * self.adaptation_rate
            )
            
            # Track response time performance
            if hasattr(pattern_data, 'llm_avg_response_time'):
                pattern_data.llm_avg_response_time = (
                    pattern_data.llm_avg_response_time * 0.8 +
                    feedback.response_time_ms * 0.2
                )
            else:
                pattern_data.llm_avg_response_time = feedback.response_time_ms

    async def _update_routing_preferences(self, pattern_data: PatternLearningData,
                                        feedback: LearningFeedback) -> None:
        """Update routing preferences based on comparative performance."""
        # Only update preferences after sufficient data
        if pattern_data.llm_usage_count < self.min_samples_for_preference:
            return
            
        # Calculate performance difference
        llm_performance = pattern_data.llm_effectiveness
        template_performance = pattern_data.success_rate
        performance_diff = llm_performance - template_performance
        
        # Update preference based on significant performance differences
        if abs(performance_diff) > self.significant_difference_threshold:
            if performance_diff > 0:
                # LLM significantly better
                pattern_data.template_vs_llm_preference = "llm"
                logger.info(f"Pattern {feedback.pattern_type.value}: "
                           f"Preference updated to LLM (diff: {performance_diff:.3f})")
            else:
                # Template significantly better
                pattern_data.template_vs_llm_preference = "template"
                logger.info(f"Pattern {feedback.pattern_type.value}: "
                           f"Preference updated to template (diff: {performance_diff:.3f})")
        else:
            # Performance similar - use hybrid approach
            pattern_data.template_vs_llm_preference = "hybrid"

    def _update_performance_cache(self, pattern_key: str, feedback: LearningFeedback) -> None:
        """Update performance cache for quick threshold adjustments."""
        if pattern_key not in self.performance_cache:
            self.performance_cache[pattern_key] = {
                'recent_llm_success': [],
                'recent_template_success': [],
                'last_updated': time.time()
            }
        
        cache = self.performance_cache[pattern_key]
        success_score = 1.0 if feedback.was_helpful else 0.0
        
        if feedback.llm_used:
            cache['recent_llm_success'].append(success_score)
            # Keep only recent samples
            cache['recent_llm_success'] = cache['recent_llm_success'][-10:]
        else:
            cache['recent_template_success'].append(success_score)
            cache['recent_template_success'] = cache['recent_template_success'][-10:]
        
        cache['last_updated'] = time.time()

    async def get_adaptive_threshold(self, pattern_type: ConversationalPattern,
                                   query_length: int) -> float:
        """
        Get adaptive confidence threshold for LLM routing.
        
        Returns dynamically adjusted threshold based on learning history.
        
        Args:
            pattern_type: Type of conversational pattern
            query_length: Length of query in words
            
        Returns:
            Adaptive confidence threshold for LLM routing decision
        """
        pattern_key = f"{pattern_type.value}_{query_length}"
        
        # Default threshold
        base_threshold = 0.7
        
        # Check if we have learning data
        if pattern_key not in self.handler.pattern_learning:
            return base_threshold
        
        pattern_data = self.handler.pattern_learning[pattern_key]
        
        # Adjust threshold based on learning preference
        if pattern_data.template_vs_llm_preference == "llm":
            # Lower threshold to favor LLM more often
            return max(0.5, base_threshold - 0.2)
        elif pattern_data.template_vs_llm_preference == "template":
            # Higher threshold to favor templates more often
            return min(0.9, base_threshold + 0.2)
        else:
            # Hybrid - adjust based on relative performance
            if hasattr(pattern_data, 'llm_effectiveness'):
                performance_factor = pattern_data.llm_effectiveness - pattern_data.success_rate
                adjustment = performance_factor * 0.3  # Scale adjustment
                return max(0.4, min(0.9, base_threshold - adjustment))
        
        return base_threshold

    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights about learning performance and patterns.
        
        Returns:
            Dictionary containing learning analytics and insights
        """
        insights = {
            'total_feedback_samples': len(self.learning_history),
            'pattern_performance': {},
            'routing_effectiveness': {},
            'recent_trends': {},
            'recommendations': []
        }
        
        # Analyze pattern performance
        for pattern_key, pattern_data in self.handler.pattern_learning.items():
            if hasattr(pattern_data, 'llm_effectiveness'):
                insights['pattern_performance'][pattern_key] = {
                    'template_success_rate': pattern_data.success_rate,
                    'llm_effectiveness': pattern_data.llm_effectiveness,
                    'llm_usage_count': pattern_data.llm_usage_count,
                    'preference': pattern_data.template_vs_llm_preference,
                    'performance_difference': pattern_data.llm_effectiveness - pattern_data.success_rate
                }
        
        # Analyze routing effectiveness
        recent_feedback = self.learning_history[-100:]  # Last 100 samples
        if recent_feedback:
            llm_success = sum(1 for f in recent_feedback if f.llm_used and f.was_helpful)
            llm_total = sum(1 for f in recent_feedback if f.llm_used)
            template_success = sum(1 for f in recent_feedback if not f.llm_used and f.was_helpful)
            template_total = sum(1 for f in recent_feedback if not f.llm_used)
            
            insights['routing_effectiveness'] = {
                'llm_success_rate': llm_success / max(1, llm_total),
                'template_success_rate': template_success / max(1, template_total),
                'llm_usage_rate': llm_total / len(recent_feedback),
                'overall_success_rate': (llm_success + template_success) / len(recent_feedback)
            }
        
        # Generate recommendations
        await self._generate_recommendations(insights)
        
        return insights

    async def _generate_recommendations(self, insights: Dict[str, Any]) -> None:
        """Generate actionable recommendations based on learning data."""
        recommendations = []
        
        # Check for patterns where LLM is significantly better
        for pattern_key, perf in insights['pattern_performance'].items():
            if perf['performance_difference'] > 0.2 and perf['llm_usage_count'] > 5:
                recommendations.append(
                    f"Pattern {pattern_key}: Consider lowering LLM threshold "
                    f"(LLM outperforming by {perf['performance_difference']:.2f})"
                )
            elif perf['performance_difference'] < -0.2 and perf['llm_usage_count'] > 5:
                recommendations.append(
                    f"Pattern {pattern_key}: Consider raising LLM threshold "
                    f"(templates outperforming by {abs(perf['performance_difference']):.2f})"
                )
        
        # Check overall routing effectiveness
        if 'routing_effectiveness' in insights:
            routing = insights['routing_effectiveness']
            if routing['llm_success_rate'] < 0.6 and routing['llm_usage_rate'] > 0.3:
                recommendations.append(
                    "LLM success rate low but usage high - consider raising confidence thresholds"
                )
            elif routing['llm_success_rate'] > 0.8 and routing['llm_usage_rate'] < 0.1:
                recommendations.append(
                    "LLM success rate high but usage low - consider lowering confidence thresholds"
                )
        
        insights['recommendations'] = recommendations

    async def reset_learning_for_pattern(self, pattern_type: ConversationalPattern) -> None:
        """Reset learning data for a specific pattern type."""
        pattern_keys = [key for key in self.handler.pattern_learning.keys() 
                       if key.startswith(pattern_type.value)]
        
        for pattern_key in pattern_keys:
            if pattern_key in self.handler.pattern_learning:
                pattern_data = self.handler.pattern_learning[pattern_key]
                pattern_data.llm_effectiveness = 0.8  # Reset to default
                pattern_data.llm_usage_count = 0
                pattern_data.template_vs_llm_preference = "template"
                
        logger.info(f"Reset learning data for pattern type: {pattern_type.value}")

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of current learning state."""
        total_patterns = len(self.handler.pattern_learning)
        llm_patterns = sum(1 for data in self.handler.pattern_learning.values() 
                          if hasattr(data, 'llm_usage_count') and data.llm_usage_count > 0)
        
        return {
            'total_patterns_tracked': total_patterns,
            'patterns_with_llm_data': llm_patterns,
            'total_feedback_samples': len(self.learning_history),
            'learning_adaptation_rate': self.adaptation_rate,
            'performance_cache_size': len(self.performance_cache),
            'status': 'active' if self.learning_history else 'initialized'
        }


# Example usage and integration points
class LearningIntegrationExample:
    """Example usage of the learning integrator."""
    
    @staticmethod
    async def demonstrate_learning_integration():
        """Demonstrate how to integrate learning feedback."""
        # Initialize with existing handler (component reuse)
        handler = ConversationalHandler()
        integrator = ConversationalLearningIntegrator(handler)
        
        # Example feedback after a conversational interaction
        feedback = LearningFeedback(
            query="Hello, how can you help me?",
            pattern_type=ConversationalPattern.GREETING,
            routing_strategy=RoutingStrategy.TEMPLATE_FIRST,
            llm_used=False,
            was_helpful=True,
            response_time_ms=45.0,
            confidence_score=0.95
        )
        
        # Update learning system
        await integrator.update_learning_with_feedback(feedback)
        
        # Get adaptive threshold for future routing
        threshold = await integrator.get_adaptive_threshold(
            ConversationalPattern.GREETING, 
            query_length=6
        )
        
        print(f"Adaptive threshold for greetings: {threshold}")
        
        # Get learning insights
        insights = await integrator.get_learning_insights()
        print(f"Learning insights: {insights}")


if __name__ == "__main__":
    # Run example
    asyncio.run(LearningIntegrationExample.demonstrate_learning_integration())
