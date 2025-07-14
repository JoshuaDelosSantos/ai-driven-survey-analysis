"""
Conversational Pattern Classifier

This module implements vector-based pattern classification for the conversational handler,
enhancing pattern recognition confidence through semantic similarity and leveraging
existing vector infrastructure.

Key Features:
- Vector-based pattern similarity scoring
- Integration with existing embedder and vector search
- Confidence boosting for template-based patterns
- Learning-driven pattern refinement
- Minimal overhead, maximum component reuse

Classes:
- ConversationalPatternClassifier: Main classifier using vector similarity
- PatternVector: Vector representation of patterns with metadata
- ClassificationResult: Enhanced result with vector confidence

Integration:
- Reuses existing Embedder from vector_search module
- Leverages proven vector infrastructure
- Seamless integration with ConversationalHandler
- Preserves all existing functionality while enhancing confidence

Example Usage:
    classifier = ConversationalPatternClassifier()
    await classifier.initialize()
    
    # Enhance pattern recognition
    result = await classifier.classify_with_vector_boost(
        query="Hello there!",
        template_confidence=0.8,
        pattern_type=ConversationalPattern.GREETING
    )
    
    # Use for edge case detection
    is_edge_case = await classifier.detect_edge_case(
        query="G'day mate, how's the data looking?",
        pattern_type=ConversationalPattern.GREETING
    )
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .handler import ConversationalPattern
from ..vector_search.embedder import Embedder, EmbeddingResult
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PatternVector:
    """Vector representation of a conversational pattern with metadata."""
    pattern_type: ConversationalPattern
    example_queries: List[str]
    embedding: Optional[List[float]] = None
    confidence_threshold: float = 0.7
    edge_case_indicators: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.edge_case_indicators is None:
            self.edge_case_indicators = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ClassificationResult:
    """Enhanced classification result with vector confidence data."""
    original_confidence: float
    vector_similarity: float
    boosted_confidence: float
    is_edge_case: bool
    similar_patterns: List[Tuple[ConversationalPattern, float]]
    processing_time: float


class ConversationalPatternClassifier:
    """
    Vector-based pattern classifier that enhances conversational pattern recognition
    using semantic similarity and existing vector infrastructure.
    """
    
    def __init__(self, embedder: Optional[Embedder] = None, vector_store = None):
        """Initialize the pattern classifier.
        
        Args:
            embedder: Optional existing embedder instance to reuse
            vector_store: Optional vector store (currently unused, for future compatibility)
        """
        self.embedder: Optional[Embedder] = embedder
        self.pattern_vectors: Dict[ConversationalPattern, PatternVector] = {}
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the classifier with embedder and pattern vectors."""
        try:
            # Initialize embedder using existing infrastructure or create new one
            if self.embedder is None:
                self.embedder = Embedder()
                await self.embedder.initialize()
                logger.info("Created new embedder for ConversationalPatternClassifier")
            else:
                # Verify existing embedder is initialized
                if not hasattr(self.embedder, '_model') or self.embedder._model is None:
                    await self.embedder.initialize()
                logger.info("Reusing existing embedder for ConversationalPatternClassifier")
            
            # Initialize pattern vectors with representative examples
            await self._initialize_pattern_vectors()
            
            self.is_initialized = True
            logger.info("ConversationalPatternClassifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationalPatternClassifier: {e}")
            raise
    
    async def _initialize_pattern_vectors(self) -> None:
        """Initialize vector representations for conversational patterns."""
        # Define representative examples for each pattern type
        pattern_examples = {
            ConversationalPattern.GREETING: [
                "Hello there",
                "Hi how are you",
                "Good morning",
                "G'day mate"
            ],
            ConversationalPattern.GREETING_FORMAL: [
                "Good day",
                "Good morning sir",
                "Pleased to meet you",
                "Thank you for your time"
            ],
            ConversationalPattern.GREETING_CASUAL: [
                "Hey there",
                "What's up",
                "Howdy",
                "Yo"
            ],
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: [
                "What can you do",
                "What are your capabilities",
                "What functions do you have",
                "What kind of help can you provide"
            ],
            ConversationalPattern.SYSTEM_QUESTION_DATA: [
                "What data do you have",
                "What information is available",
                "What datasets do you access",
                "Tell me about the data"
            ],
            ConversationalPattern.POLITENESS_THANKS: [
                "Thank you",
                "Thanks",
                "Much appreciated",
                "Cheers"
            ],
            ConversationalPattern.POLITENESS_GOODBYE: [
                "Goodbye",
                "See you later",
                "Take care",
                "Bye"
            ],
            ConversationalPattern.OFF_TOPIC: [
                "What's the weather like",
                "Tell me about politics",
                "How do I cook pasta",
                "What's happening in sports"
            ],
            ConversationalPattern.HELP_REQUEST: [
                "I need help",
                "Can you help me",
                "I'm stuck",
                "How do I start"
            ]
        }
        
        # Create embeddings for pattern examples
        for pattern_type, examples in pattern_examples.items():
            try:
                # Create combined text for embedding
                combined_text = " | ".join(examples)
                
                # Generate embedding
                result = await self.embedder.embed_text(combined_text)
                
                # Create pattern vector
                pattern_vector = PatternVector(
                    pattern_type=pattern_type,
                    example_queries=examples,
                    embedding=result.embedding,
                    confidence_threshold=self._get_pattern_threshold(pattern_type)
                )
                
                self.pattern_vectors[pattern_type] = pattern_vector
                
            except Exception as e:
                logger.warning(f"Failed to create embedding for pattern {pattern_type}: {e}")
                # Create pattern vector without embedding as fallback
                self.pattern_vectors[pattern_type] = PatternVector(
                    pattern_type=pattern_type,
                    example_queries=examples,
                    confidence_threshold=self._get_pattern_threshold(pattern_type)
                )
    
    def _get_pattern_threshold(self, pattern_type: ConversationalPattern) -> float:
        """Get confidence threshold for different pattern types."""
        # More specific patterns have higher thresholds
        specific_patterns = {
            ConversationalPattern.GREETING_FORMAL: 0.8,
            ConversationalPattern.GREETING_CASUAL: 0.8,
            ConversationalPattern.SYSTEM_QUESTION_CAPABILITIES: 0.85,
            ConversationalPattern.SYSTEM_QUESTION_DATA: 0.85,
            ConversationalPattern.POLITENESS_THANKS: 0.75,
            ConversationalPattern.POLITENESS_GOODBYE: 0.75,
        }
        
        return specific_patterns.get(pattern_type, 0.7)
    
    async def classify_with_vector_boost(
        self, 
        query: str, 
        template_confidence: float, 
        pattern_type: ConversationalPattern
    ) -> ClassificationResult:
        """
        Enhance pattern classification using vector similarity.
        
        Args:
            query: User query to classify
            template_confidence: Original template-based confidence
            pattern_type: Template-identified pattern type
            
        Returns:
            ClassificationResult with enhanced confidence and metadata
        """
        start_time = datetime.now()
        
        if not self.is_initialized:
            logger.warning("Classifier not initialized, using template confidence")
            return ClassificationResult(
                original_confidence=template_confidence,
                vector_similarity=0.0,
                boosted_confidence=template_confidence,
                is_edge_case=False,
                similar_patterns=[],
                processing_time=0.0
            )
        
        try:
            # Generate embedding for the query
            query_result = await self.embedder.embed_text(query)
            query_embedding = query_result.embedding
            
            # Calculate similarity with pattern vectors
            similarities = await self._calculate_pattern_similarities(query_embedding)
            
            # Get similarity for the identified pattern
            pattern_similarity = similarities.get(pattern_type, 0.0)
            
            # Boost confidence based on vector similarity
            vector_boost = self._calculate_confidence_boost(pattern_similarity, template_confidence)
            boosted_confidence = min(0.95, template_confidence + vector_boost)
            
            # Detect edge cases
            is_edge_case = await self._detect_edge_case(similarities, pattern_type, template_confidence)
            
            # Get top similar patterns
            sorted_similarities = sorted(
                similarities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ClassificationResult(
                original_confidence=template_confidence,
                vector_similarity=pattern_similarity,
                boosted_confidence=boosted_confidence,
                is_edge_case=is_edge_case,
                similar_patterns=sorted_similarities,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Vector classification failed: {e}")
            # Fallback to template confidence
            return ClassificationResult(
                original_confidence=template_confidence,
                vector_similarity=0.0,
                boosted_confidence=template_confidence,
                is_edge_case=False,
                similar_patterns=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _calculate_pattern_similarities(self, query_embedding: List[float]) -> Dict[ConversationalPattern, float]:
        """Calculate cosine similarity between query and all pattern vectors."""
        similarities = {}
        
        for pattern_type, pattern_vector in self.pattern_vectors.items():
            if pattern_vector.embedding:
                similarity = self._cosine_similarity(query_embedding, pattern_vector.embedding)
                similarities[pattern_type] = similarity
            else:
                similarities[pattern_type] = 0.0
        
        return similarities
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays for efficient computation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_confidence_boost(self, vector_similarity: float, template_confidence: float) -> float:
        """Calculate confidence boost based on vector similarity."""
        # Strong vector similarity boosts confidence
        if vector_similarity > 0.8:
            return 0.1
        elif vector_similarity > 0.6:
            return 0.05
        elif vector_similarity > 0.4:
            return 0.02
        
        # Low vector similarity might indicate edge case
        if vector_similarity < 0.3 and template_confidence > 0.7:
            return -0.05  # Slight penalty for potential misclassification
        
        return 0.0
    
    async def _detect_edge_case(
        self, 
        similarities: Dict[ConversationalPattern, float], 
        identified_pattern: ConversationalPattern, 
        template_confidence: float
    ) -> bool:
        """Detect if this query represents an edge case requiring LLM intervention."""
        # Edge case indicators:
        
        # 1. Low similarity to identified pattern
        pattern_similarity = similarities.get(identified_pattern, 0.0)
        if pattern_similarity < 0.4:
            return True
        
        # 2. High similarity to multiple different patterns (ambiguous)
        high_similarities = [sim for sim in similarities.values() if sim > 0.6]
        if len(high_similarities) > 2:
            return True
        
        # 3. Medium template confidence with conflicting vector evidence
        if 0.5 < template_confidence < 0.8:
            # Check if another pattern has significantly higher similarity
            max_similarity = max(similarities.values())
            if (max_similarity > pattern_similarity + 0.2 and 
                max_similarity > 0.7):
                return True
        
        # 4. Low overall similarity to any known pattern
        max_similarity = max(similarities.values()) if similarities else 0.0
        if max_similarity < 0.3:
            return True
        
        return False
    
    async def detect_edge_case(self, query: str, pattern_type: ConversationalPattern) -> bool:
        """Public method to detect edge cases for LLM routing decisions."""
        if not self.is_initialized:
            return False
        
        try:
            # Generate embedding for the query
            query_result = await self.embedder.embed_text(query)
            query_embedding = query_result.embedding
            
            # Calculate similarities
            similarities = await self._calculate_pattern_similarities(query_embedding)
            
            # Use edge case detection logic
            return await self._detect_edge_case(similarities, pattern_type, 0.7)
            
        except Exception as e:
            logger.error(f"Edge case detection failed: {e}")
            return False
    
    async def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about pattern classification performance."""
        insights = {
            "total_patterns": len(self.pattern_vectors),
            "initialized_patterns": sum(1 for pv in self.pattern_vectors.values() if pv.embedding),
            "pattern_details": {}
        }
        
        for pattern_type, pattern_vector in self.pattern_vectors.items():
            insights["pattern_details"][pattern_type.value] = {
                "has_embedding": pattern_vector.embedding is not None,
                "example_count": len(pattern_vector.example_queries),
                "confidence_threshold": pattern_vector.confidence_threshold,
                "created_at": pattern_vector.created_at.isoformat()
            }
        
        return insights
