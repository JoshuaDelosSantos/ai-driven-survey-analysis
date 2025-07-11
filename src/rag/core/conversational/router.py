"""
Conversational Router

This module provides intelligent routing for conversational queries, orchestrating
template-based responses with optional LLM enhancement while maximizing component reuse.

Key Features:
- Preserves existing ConversationalHandler as primary system
- Uses vector-based confidence enhancement from Phase 1
- Applies LLM enhancement only when needed (conservative approach)
- Integrates learning data for intelligent routing decisions
- Maximum component reuse, minimal new code

Classes:
- ConversationalRouter: Main orchestration class
- RoutingDecision: Decision metadata for audit and learning
- RoutingStrategy: Strategy enumeration for routing logic

Integration:
- Primary: Existing ConversationalHandler (preserved functionality)
- Enhancement: ConversationalPatternClassifier (Phase 1 vector confidence)
- Fallback: ConversationalLLMEnhancer (Phase 2 LLM enhancement)
- Learning: Existing pattern learning system (preserved and enhanced)

Example Usage:
    router = ConversationalRouter()
    await router.initialize()
    
    # Route query with intelligent enhancement
    response = await router.route_conversational_query(
        query="G'day! Can you help me understand our survey trends?",
        context={"user_id": "analyst_123", "session_id": "session_456"}
    )
    
    # Check routing decision
    if response.routing_metadata:
        print(f"Strategy used: {response.routing_metadata.strategy_used}")
        print(f"Enhancement applied: {response.routing_metadata.llm_enhancement_used}")
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .handler import ConversationalHandler, ConversationalPattern, ConversationalResponse
from .pattern_classifier import ConversationalPatternClassifier, ClassificationResult
from .llm_enhancer import ConversationalLLMEnhancer, EnhancedResponse
from .learning_integrator import ConversationalLearningIntegrator, LearningFeedback
from .performance_monitor import ConversationalPerformanceMonitor
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class RoutingStrategy(Enum):
    """Strategy used for routing conversational queries."""
    TEMPLATE_ONLY = "template_only"
    VECTOR_ENHANCED = "vector_enhanced"
    LLM_ENHANCED = "llm_enhanced"
    LEARNING_DRIVEN = "learning_driven"
    FALLBACK = "fallback"


@dataclass
class RoutingDecision:
    """Metadata about routing decision for audit and learning."""
    strategy_used: RoutingStrategy
    template_confidence: float
    vector_enhanced_confidence: Optional[float] = None
    llm_enhancement_used: bool = False
    pattern_detected: Optional[ConversationalPattern] = None
    processing_time: float = 0.0
    learning_data_influenced: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class RoutedResponse:
    """Response with routing metadata for comprehensive tracking."""
    content: str
    confidence: float
    pattern_type: ConversationalPattern
    routing_metadata: RoutingDecision
    original_response: Optional[ConversationalResponse] = None
    enhanced_data: Optional[Dict[str, Any]] = None


class ConversationalRouter:
    """
    Intelligent router that orchestrates conversational query handling using
    all existing components with minimal new functionality.
    
    Routing Logic:
    1. Use existing ConversationalHandler for primary detection and response
    2. Enhance confidence using ConversationalPatternClassifier (vector-based)
    3. Check existing learning data for routing guidance
    4. Apply LLM enhancement only when confidence is low and learning suggests it
    5. Fall back gracefully to template responses
    """
    
    def __init__(self, 
                 handler: Optional[ConversationalHandler] = None,
                 pattern_classifier: Optional[ConversationalPatternClassifier] = None,
                 llm_enhancer: Optional[ConversationalLLMEnhancer] = None,
                 learning_integrator: Optional[ConversationalLearningIntegrator] = None,
                 performance_monitor: Optional[ConversationalPerformanceMonitor] = None,
                 enable_learning: bool = True,
                 enable_monitoring: bool = True):
        """Initialize the router with existing and Phase 3 components."""
        self.handler = handler
        self.pattern_classifier = pattern_classifier
        self.llm_enhancer = llm_enhancer
        self.learning_integrator = learning_integrator
        self.performance_monitor = performance_monitor
        self.enable_learning = enable_learning
        self.enable_monitoring = enable_monitoring
        self.is_initialized = False
        
        # Routing thresholds (conservative approach)
        self.vector_enhancement_threshold = 0.6
        self.llm_enhancement_threshold = 0.7
        self.learning_influence_threshold = 0.8
        
    async def initialize(self) -> None:
        """Initialize all routing components."""
        if self.is_initialized:
            return
            
        try:
            # Initialize existing ConversationalHandler
            if self.handler is None:
                self.handler = ConversationalHandler()
                # Handler initializes itself
                
            # Initialize Phase 1 pattern classifier
            if self.pattern_classifier is None:
                self.pattern_classifier = ConversationalPatternClassifier()
                await self.pattern_classifier.initialize()
                
            # Initialize Phase 2 LLM enhancer
            if self.llm_enhancer is None:
                self.llm_enhancer = ConversationalLLMEnhancer()
                await self.llm_enhancer.initialize()
                
            # Initialize Phase 3 learning integrator
            if self.learning_integrator is None and self.enable_learning:
                self.learning_integrator = ConversationalLearningIntegrator(self.handler)
                
            # Initialize Phase 3 performance monitor
            if self.performance_monitor is None and self.enable_monitoring:
                self.performance_monitor = ConversationalPerformanceMonitor()
                
            self.is_initialized = True
            logger.info("ConversationalRouter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationalRouter: {e}")
            raise
            
    async def route_conversational_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> RoutedResponse:
        """
        Route conversational query using intelligent multi-stage approach.
        
        Args:
            query: User query to route
            context: Optional context (user_id, session_id, etc.)
            
        Returns:
            RoutedResponse with routing decision and enhanced response
        """
        if not self.is_initialized:
            await self.initialize()
            
        start_time = datetime.now()
        routing_decision = RoutingDecision(
            strategy_used=RoutingStrategy.TEMPLATE_ONLY,
            template_confidence=0.0
        )
        
        try:
            # Stage 1: Primary detection using existing handler
            is_conversational, pattern_type, base_confidence = self.handler.is_conversational_query(query)
            
            if not is_conversational:
                # Not conversational - return early with indication
                processing_time = (datetime.now() - start_time).total_seconds()
                routing_decision.processing_time = processing_time
                routing_decision.fallback_reason = "Not identified as conversational"
                
                return RoutedResponse(
                    content="I can help you with data analysis questions. Could you please ask about survey data, user statistics, or learning content?",
                    confidence=0.1,
                    pattern_type=ConversationalPattern.UNKNOWN,
                    routing_metadata=routing_decision
                )
            
            routing_decision.pattern_detected = pattern_type
            routing_decision.template_confidence = base_confidence
            
            # Stage 2: Get template response using existing system
            template_response = self.handler.handle_conversational_query(query)
            
            # Stage 3: Vector-based confidence enhancement (Phase 1)
            vector_confidence = await self._enhance_confidence_with_vectors(
                query, base_confidence, pattern_type
            )
            routing_decision.vector_enhanced_confidence = vector_confidence
            
            # Update strategy if vector enhancement was significant
            if vector_confidence > base_confidence + 0.1:
                routing_decision.strategy_used = RoutingStrategy.VECTOR_ENHANCED
            
            # Stage 4: Check learning data influence (existing system)
            should_use_llm, learning_influenced = self._check_learning_guidance(
                query, pattern_type, vector_confidence
            )
            routing_decision.learning_data_influenced = learning_influenced
            
            if learning_influenced:
                routing_decision.strategy_used = RoutingStrategy.LEARNING_DRIVEN
            
            # Stage 5: Apply LLM enhancement if needed (Phase 2)
            final_response = template_response
            final_confidence = vector_confidence
            
            if should_use_llm and vector_confidence < self.llm_enhancement_threshold:
                enhanced_response = await self.llm_enhancer.enhance_response_if_needed(
                    query=query,
                    template_response=template_response.content,
                    confidence=vector_confidence,
                    pattern_type=pattern_type,
                    context=context
                )
                
                if enhanced_response.enhancement_used:
                    final_response = ConversationalResponse(
                        content=enhanced_response.content,
                        confidence=enhanced_response.confidence,
                        pattern_type=enhanced_response.pattern_type,
                        context=template_response.context,
                        generated_at=datetime.now()
                    )
                    final_confidence = enhanced_response.confidence
                    routing_decision.strategy_used = RoutingStrategy.LLM_ENHANCED
                    routing_decision.llm_enhancement_used = True
            
            # Stage 6: Update learning data (integrate with existing system)
            await self._update_learning_data(
                query, pattern_type, routing_decision, final_confidence
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            routing_decision.processing_time = processing_time
            
            return RoutedResponse(
                content=final_response.content,
                confidence=final_confidence,
                pattern_type=pattern_type,
                routing_metadata=routing_decision,
                original_response=template_response,
                enhanced_data={
                    "vector_similarity": getattr(vector_confidence, 'similarity_score', None),
                    "llm_processing_time": getattr(enhanced_response, 'llm_processing_time', 0.0) if 'enhanced_response' in locals() else 0.0
                }
            )
            
        except Exception as e:
            logger.error(f"Routing failed, using fallback: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Fallback to basic template response
            fallback_response = self.handler.handle_conversational_query(query)
            routing_decision.strategy_used = RoutingStrategy.FALLBACK
            routing_decision.fallback_reason = f"Routing error: {str(e)}"
            routing_decision.processing_time = processing_time
            
            return RoutedResponse(
                content=fallback_response.content,
                confidence=fallback_response.confidence,
                pattern_type=fallback_response.pattern_type,
                routing_metadata=routing_decision,
                original_response=fallback_response
            )
    
    async def _enhance_confidence_with_vectors(
        self, 
        query: str, 
        base_confidence: float, 
        pattern_type: ConversationalPattern
    ) -> float:
        """Enhance confidence using vector similarity (Phase 1 integration)."""
        try:
            if base_confidence < self.vector_enhancement_threshold:
                # Use Phase 1 pattern classifier for confidence boost
                classification_result = await self.pattern_classifier.classify_with_vector_boost(
                    query=query,
                    template_confidence=base_confidence,
                    pattern_type=pattern_type
                )
                return classification_result.boosted_confidence
            return base_confidence
            
        except Exception as e:
            logger.warning(f"Vector confidence enhancement failed: {e}")
            return base_confidence
    
    def _check_learning_guidance(
        self, 
        query: str, 
        pattern_type: ConversationalPattern, 
        confidence: float
    ) -> Tuple[bool, bool]:
        """Check existing learning data for routing guidance."""
        try:
            # Use existing learning system from handler
            if hasattr(self.handler, 'pattern_learning') and self.handler.pattern_learning:
                pattern_key = f"{pattern_type.value}_{len(query.split())}"
                
                if pattern_key in self.handler.pattern_learning:
                    pattern_data = self.handler.pattern_learning[pattern_key]
                    
                    # Use existing should_try_llm method if available
                    if hasattr(pattern_data, 'should_try_llm'):
                        should_use_llm = pattern_data.should_try_llm()
                        return should_use_llm, True
                    
                    # Fallback logic using existing success_rate
                    if hasattr(pattern_data, 'success_rate'):
                        # If templates are working well, don't use LLM
                        if pattern_data.success_rate > self.learning_influence_threshold:
                            return False, True
                        # If templates are failing, consider LLM
                        elif pattern_data.success_rate < 0.6 and confidence < 0.7:
                            return True, True
            
            # No learning data influence
            return confidence < self.llm_enhancement_threshold, False
            
        except Exception as e:
            logger.warning(f"Learning guidance check failed: {e}")
            return confidence < self.llm_enhancement_threshold, False
    
    async def _update_learning_data(
        self, 
        query: str, 
        pattern_type: ConversationalPattern, 
        routing_decision: RoutingDecision, 
        final_confidence: float
    ) -> None:
        """Update learning data and monitor performance (Phase 3 integration)."""
        try:
            processing_time_ms = routing_decision.processing_time * 1000
            
            # Phase 3: Create learning feedback
            if self.learning_integrator and self.enable_learning:
                feedback = LearningFeedback(
                    query=query,
                    pattern_type=pattern_type,
                    routing_strategy=routing_decision.strategy_used,
                    llm_used=routing_decision.llm_enhancement_used,
                    was_helpful=True,  # Default to True, will be updated with real feedback
                    response_time_ms=processing_time_ms,
                    confidence_score=final_confidence
                )
                
                # Update learning system
                await self.learning_integrator.update_learning_with_feedback(feedback)
                
                # Get adaptive threshold for future routing
                adaptive_threshold = await self.learning_integrator.get_adaptive_threshold(
                    pattern_type, len(query.split())
                )
                
                # Update router thresholds based on learning
                if abs(adaptive_threshold - self.llm_enhancement_threshold) > 0.1:
                    self.llm_enhancement_threshold = adaptive_threshold
                    logger.info(f"Updated LLM threshold to {adaptive_threshold:.2f} based on learning")
            
            # Phase 3: Record performance metrics
            if self.performance_monitor and self.enable_monitoring:
                await self.performance_monitor.record_interaction(feedback, routing_decision)
                
            logger.debug(f"Learning updated for pattern {pattern_type}: "
                        f"strategy={routing_decision.strategy_used}, "
                        f"LLM used={routing_decision.llm_enhancement_used}")
            
        except Exception as e:
            logger.warning(f"Learning data update failed: {e}")

    async def provide_feedback(self, query: str, was_helpful: bool, 
                             user_satisfaction: Optional[float] = None) -> None:
        """
        Provide feedback on a recent response to improve learning.
        
        Args:
            query: The original query
            was_helpful: Whether the response was helpful
            user_satisfaction: Optional satisfaction score (0.0-1.0)
        """
        if not (self.learning_integrator and self.enable_learning):
            return
            
        try:
            # Find recent feedback for this query (simple matching)
            if hasattr(self.learning_integrator, 'learning_history'):
                recent_feedback = [f for f in self.learning_integrator.learning_history[-10:] 
                                 if f.query == query]
                if recent_feedback:
                    # Update the most recent matching feedback
                    latest_feedback = recent_feedback[-1]
                    latest_feedback.was_helpful = was_helpful
                    latest_feedback.user_satisfaction = user_satisfaction
                    
                    # Re-update learning with corrected feedback
                    await self.learning_integrator.update_learning_with_feedback(latest_feedback)
                    
                    logger.info(f"Updated feedback for query: helpful={was_helpful}, "
                               f"satisfaction={user_satisfaction}")
                    
        except Exception as e:
            logger.error(f"Failed to provide feedback: {e}")

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights and performance analytics."""
        insights = {}
        
        if self.learning_integrator and self.enable_learning:
            insights['learning'] = await self.learning_integrator.get_learning_insights()
            
        if self.performance_monitor and self.enable_monitoring:
            insights['performance'] = await self.performance_monitor.get_performance_report()
            
        return insights

    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics for monitoring and optimization."""
        stats = {
            "router_initialized": self.is_initialized,
            "components_available": {
                "handler": self.handler is not None,
                "pattern_classifier": self.pattern_classifier is not None,
                "llm_enhancer": self.llm_enhancer is not None,
                "learning_integrator": self.learning_integrator is not None,
                "performance_monitor": self.performance_monitor is not None
            },
            "phase3_enabled": {
                "learning": self.enable_learning,
                "monitoring": self.enable_monitoring
            },
            "thresholds": {
                "vector_enhancement": self.vector_enhancement_threshold,
                "llm_enhancement": self.llm_enhancement_threshold,
                "learning_influence": self.learning_influence_threshold
            }
        }
        
        # Add learning summary if available
        if self.learning_integrator and self.enable_learning:
            stats["learning_summary"] = self.learning_integrator.get_learning_summary()
            
        # Add monitoring status if available  
        if self.performance_monitor and self.enable_monitoring:
            stats["monitoring_status"] = self.performance_monitor.get_monitoring_status()
            
        return stats
