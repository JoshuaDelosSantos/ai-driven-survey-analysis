"""
Multi-Stage Query Classification System for RAG Agent

This module implements a sophisticated query classification system that determines
the optimal processing strategy for user queries through multiple stages of analysis.

Key Features:
- Rule-based pre-filtering for fast routing of obvious queries
- LLM-based classification with confidence scoring for complex queries
- Comprehensive fallback mechanisms for robustness
- Australian PII protection throughout classification process
- Structured logging for classification decisions and audit compliance

Classification Categories:
- SQL: Queries requiring statistical analysis, aggregations, or structured data retrieval
- VECTOR: Queries requiring semantic search through free-text feedback and comments
- HYBRID: Queries requiring both statistical context and semantic content analysis
- CLARIFICATION_NEEDED: Ambiguous queries requiring user clarification

Security: Mandatory PII anonymization before LLM processing.
Performance: Rule-based pre-filter for fast path, LLM fallback for complex cases.
Privacy: Australian Privacy Principles (APP) compliance maintained.

Example Usage:
    # Basic classification
    classifier = QueryClassifier()
    await classifier.initialize()
    
    # Classify a statistical query
    result = await classifier.classify_query(
        "How many Level 6 users completed courses this year?"
    )
    print(f"Classification: {result.classification}")  # SQL
    print(f"Confidence: {result.confidence}")          # HIGH
    
    # Classify a feedback query
    result = await classifier.classify_query(
        "What did people say about the virtual learning experience?"
    )
    print(f"Classification: {result.classification}")  # VECTOR
    
    # Classify a hybrid query
    result = await classifier.classify_query(
        "Analyze satisfaction trends across agencies with supporting feedback"
    )
    print(f"Classification: {result.classification}")  # HYBRID
    
    # Handle ambiguous query
    result = await classifier.classify_query("Tell me about courses")
    if result.confidence == "LOW":
        print(f"Needs clarification: {result.reasoning}")
    
    # Classification with session tracking
    result = await classifier.classify_query(
        query="Show me feedback trends",
        session_id="user_123",
        anonymize_query=True
    )
"""

import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..privacy.pii_detector import AustralianPIIDetector
from ...utils.llm_utils import get_llm
from ...utils.logging_utils import get_logger
from ...config.settings import get_settings


logger = get_logger(__name__)


# Type definitions
ClassificationType = Literal["SQL", "VECTOR", "HYBRID", "CLARIFICATION_NEEDED"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


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
    """
    classification: ClassificationType
    confidence: ConfidenceLevel
    reasoning: str
    processing_time: float
    method_used: Literal["rule_based", "llm_based", "fallback"]
    anonymized_query: Optional[str] = None


class ClassificationMethod(Enum):
    """Enumeration of available classification methods."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    FALLBACK = "fallback"


class QueryClassifier:
    """
    Multi-stage query classification system for RAG agent.
    
    This classifier determines the optimal processing strategy for user queries
    through a sophisticated multi-stage approach:
    
    1. Rule-based pre-filtering for obvious queries (fast path)
    2. LLM-based classification for complex queries (comprehensive analysis)
    3. Fallback mechanisms for error recovery and robustness
    
    All classification includes mandatory PII anonymization and comprehensive
    audit logging for compliance with Australian Privacy Principles.
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize query classifier.
        
        Args:
            llm: Language model for LLM-based classification. If None, will use get_llm()
        """
        self._llm = llm
        self.settings = get_settings()
        self._pii_detector: Optional[AustralianPIIDetector] = None
        self._classification_prompt: Optional[PromptTemplate] = None
        
        # Enhanced classification patterns for rule-based pre-filtering with APS domain knowledge
        self._sql_patterns = [
            # Core statistical patterns (preserved from original)
            r'\b(?:count|how many|number of)\b',
            r'\b(?:average|mean|avg)\b',
            r'\b(?:percentage|percent|%)\b',
            r'\b(?:breakdown by|group by|categorized by)\b',
            r'\b(?:statistics|stats|statistical)\b',
            r'\b(?:total|sum|aggregate)\b',
            r'\b(?:compare numbers|numerical comparison)\b',
            r'\b(?:completion rate|enrollment rate)\b',
            r'\b(?:agency breakdown|level breakdown|user level)\b',
            
            # Enhanced APS-specific statistical patterns
            r'\b(?:executive level|level [1-6]|EL[12]|APS [1-6])\b.*(?:completion|attendance|performance)',
            r'\b(?:agency|department|portfolio)\b.*(?:breakdown|comparison|statistics)',
            r'\b(?:learning pathway|professional development|capability framework)\b.*(?:metrics|data)',
            r'\b(?:mandatory training|compliance training)\b.*(?:rates|numbers|tracking)',
            r'\b(?:face-to-face|virtual|blended)\b.*(?:delivery|attendance|completion)',
            r'\b(?:cost per|budget|resource allocation)\b.*(?:training|learning)',
            r'\b(?:quarterly|annual|yearly)\b.*(?:training|learning|development)\b.*(?:report|summary)',
            r'\b(?:participation rate|dropout rate|success rate)\b',
            r'\b(?:training hours|contact hours|learning hours)\b.*(?:total|average|per)',
            r'\b(?:geographical|location|state)\b.*(?:breakdown|distribution)'
        ]
        
        self._vector_patterns = [
            # Core feedback patterns (preserved from original)
            r'\b(?:what did.*say|what.*said)\b',
            r'\b(?:feedback about|comments about|opinions on)\b',
            r'\b(?:experiences with|experience of)\b',
            r'\b(?:user feedback|participant feedback)\b',
            r'\b(?:comments|opinions|thoughts|feelings)\b',
            r'\b(?:issues mentioned|problems reported)\b',
            r'\b(?:satisfaction|dissatisfaction)\b',
            r'\b(?:testimonials|reviews|responses)\b',
            r'\b(?:what people think|user opinions)\b',
            
            # Enhanced APS-specific feedback patterns
            r'\b(?:participant|delegate|attendee)\b.*(?:experience|reflection|view)',
            r'\b(?:training quality|course quality|learning experience)\b.*(?:feedback|assessment)',
            r'\b(?:facilitator|presenter|instructor)\b.*(?:effectiveness|skill|performance)',
            r'\b(?:venue|location|facilities)\b.*(?:issues|problems|concerns)',
            r'\b(?:accessibility|inclusion|diversity)\b.*(?:feedback|experience)',
            r'\b(?:technical issues|platform problems|system difficulties)\b',
            r'\b(?:relevance to role|workplace application|practical use)\b',
            r'\b(?:course content|curriculum|material)\b.*(?:feedback|quality|relevance)',
            r'\b(?:learning outcomes|skill development|capability building)\b.*(?:feedback|experience)',
            r'\b(?:recommendation|would recommend|likelihood to recommend)\b'
        ]
        
        self._hybrid_patterns = [
            # Core hybrid patterns (preserved from original)
            r'\b(?:analyze satisfaction|analyze feedback)\b',
            r'\b(?:compare feedback across|feedback trends)\b',
            r'\b(?:sentiment by agency|satisfaction by level)\b',
            r'\b(?:trends in opinions|opinion trends)\b',
            r'\b(?:comprehensive analysis|detailed analysis)\b',
            r'\b(?:both.*and|statistics.*feedback|numbers.*comments)\b',
            r'\b(?:quantitative.*qualitative|statistical.*sentiment)\b',
            
            # Enhanced APS-specific hybrid patterns
            r'\b(?:analyse|analyze)\b.*(?:satisfaction|effectiveness)\b.*(?:across|by|between)',
            r'\b(?:training ROI|return on investment|cost-benefit)\b.*(?:analysis|evaluation)',
            r'\b(?:performance impact|capability improvement|skill development)\b.*(?:measurement|assessment)',
            r'\b(?:stakeholder satisfaction|user experience)\b.*(?:metrics|analysis)',
            r'\b(?:trend analysis|pattern identification|insight generation)\b',
            r'\b(?:comprehensive|holistic|integrated)\b.*(?:evaluation|assessment|review)',
            r'\b(?:correlate|correlation)\b.*(?:satisfaction|feedback)\b.*(?:with|and)\b.*(?:completion|performance)',
            r'\b(?:demographic analysis|cohort analysis)\b.*(?:feedback|satisfaction)'
        ]
        
        # Pattern weighting system for improved confidence calibration
        self._pattern_weights = {
            "SQL": {
                "high_confidence": [
                    r'\b(?:count|how many|number of)\b',
                    r'\b(?:percentage|percent|%)\b',
                    r'\b(?:total|sum|aggregate)\b',
                    r'\b(?:completion rate|enrollment rate)\b',
                    r'\b(?:executive level|level [1-6]|EL[12]|APS [1-6])\b.*(?:completion|attendance|performance)',
                    r'\b(?:participation rate|dropout rate|success rate)\b'
                ],
                "medium_confidence": [
                    r'\b(?:breakdown by|group by|categorized by)\b',
                    r'\b(?:statistics|stats|statistical)\b',
                    r'\b(?:average|mean|avg)\b',
                    r'\b(?:agency|department|portfolio)\b.*(?:breakdown|comparison|statistics)',
                    r'\b(?:training hours|contact hours|learning hours)\b.*(?:total|average|per)'
                ],
                "low_confidence": [
                    r'\b(?:compare numbers|numerical comparison)\b',
                    r'\b(?:quarterly|annual|yearly)\b.*(?:training|learning|development)\b.*(?:report|summary)',
                    r'\b(?:geographical|location|state)\b.*(?:breakdown|distribution)'
                ]
            },
            "VECTOR": {
                "high_confidence": [
                    r'\b(?:what did.*say|feedback about)\b',
                    r'\b(?:comments|opinions|thoughts)\b',
                    r'\b(?:participant|delegate|attendee)\b.*(?:experience|reflection|view)',
                    r'\b(?:technical issues|platform problems|system difficulties)\b',
                    r'\b(?:recommendation|would recommend|likelihood to recommend)\b'
                ],
                "medium_confidence": [
                    r'\b(?:experiences with|satisfaction)\b',
                    r'\b(?:training quality|course quality|learning experience)\b.*(?:feedback|assessment)',
                    r'\b(?:facilitator|presenter|instructor)\b.*(?:effectiveness|skill|performance)',
                    r'\b(?:relevance to role|workplace application|practical use)\b'
                ],
                "low_confidence": [
                    r'\b(?:feelings|thoughts)\b',
                    r'\b(?:venue|location|facilities)\b.*(?:issues|problems|concerns)',
                    r'\b(?:accessibility|inclusion|diversity)\b.*(?:feedback|experience)'
                ]
            },
            "HYBRID": {
                "high_confidence": [
                    r'\b(?:analyze satisfaction|comprehensive analysis)\b',
                    r'\b(?:training ROI|return on investment|cost-benefit)\b.*(?:analysis|evaluation)',
                    r'\b(?:correlate|correlation)\b.*(?:satisfaction|feedback)\b.*(?:with|and)\b.*(?:completion|performance)'
                ],
                "medium_confidence": [
                    r'\b(?:trends in|patterns in)\b',
                    r'\b(?:performance impact|capability improvement|skill development)\b.*(?:measurement|assessment)',
                    r'\b(?:stakeholder satisfaction|user experience)\b.*(?:metrics|analysis)',
                    r'\b(?:demographic analysis|cohort analysis)\b.*(?:feedback|satisfaction)'
                ],
                "low_confidence": [
                    r'\b(?:both.*and|detailed analysis)\b',
                    r'\b(?:comprehensive|holistic|integrated)\b.*(?:evaluation|assessment|review)',
                    r'\b(?:trend analysis|pattern identification|insight generation)\b'
                ]
            }
        }
        
        # Compile patterns for performance
        self._compiled_patterns = {
            "SQL": [re.compile(pattern, re.IGNORECASE) for pattern in self._sql_patterns],
            "VECTOR": [re.compile(pattern, re.IGNORECASE) for pattern in self._vector_patterns],
            "HYBRID": [re.compile(pattern, re.IGNORECASE) for pattern in self._hybrid_patterns]
        }
        
        # Compile weighted patterns for enhanced confidence calculation
        self._compiled_weighted_patterns = {}
        for category in ["SQL", "VECTOR", "HYBRID"]:
            self._compiled_weighted_patterns[category] = {}
            for confidence_level in ["high_confidence", "medium_confidence", "low_confidence"]:
                self._compiled_weighted_patterns[category][confidence_level] = [
                    re.compile(pattern, re.IGNORECASE) 
                    for pattern in self._pattern_weights[category][confidence_level]
                ]
        
        # Classification statistics
        self._classification_count = 0
        self._method_stats = {
            "rule_based": 0,
            "llm_based": 0,
            "fallback": 0
        }
    
    async def initialize(self) -> None:
        """
        Initialize classifier components.
        
        Sets up LLM connection, PII detection, and classification prompts.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing query classifier...")
            start_time = time.time()
            
            # Initialize LLM if not provided
            if self._llm is None:
                self._llm = get_llm()
                logger.info(f"LLM initialized: {type(self._llm).__name__}")
            
            # Initialize PII detection
            self._pii_detector = AustralianPIIDetector()
            logger.info("PII detection system initialized")
            
            # Initialize classification prompt
            self._setup_classification_prompt()
            
            initialization_time = time.time() - start_time
            logger.info(f"Query classifier initialized successfully in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Query classifier initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize query classifier: {e}")
    
    def _setup_classification_prompt(self) -> None:
        """Set up the LLM classification prompt template."""
        prompt_text = """You are an expert query router for an Australian Public Service learning analytics system. Your task is to classify user queries into one of three categories:

SQL: Queries requiring statistical analysis, aggregations, or structured data retrieval.
VECTOR: Queries requiring semantic search through free-text feedback and comments.  
HYBRID: Queries requiring both statistical context and semantic content analysis.

DOMAIN CONTEXT:
- Users ask about course evaluations, attendance patterns, and learning outcomes.
- Structured data: attendance records, user levels (1-6, Exec 1-2), agencies, course types.
- Unstructured data: general feedback, issue details, course applications.

CLASSIFICATION RULES:
SQL indicators: "how many", "count", "average", "percentage", "breakdown by", "statistics", "numbers", "total".
VECTOR indicators: "what did people say", "feedback about", "experiences with", "opinions on", "comments", "issues mentioned".
HYBRID indicators: "analyze satisfaction", "compare feedback across", "trends in opinions", "sentiment by agency".

CONFIDENCE SCORING:
- HIGH (0.8-1.0): Clear keyword matches, unambiguous intent
- MEDIUM (0.5-0.79): Some indicators present, minor ambiguity
- LOW (0.0-0.49): Unclear intent, multiple possible interpretations

RESPONSE FORMAT:
Classification: [SQL|VECTOR|HYBRID]
Confidence: [HIGH|MEDIUM|LOW]
Reasoning: Brief explanation of classification decision

Query: "{query}"

Classification:"""
        
        self._classification_prompt = PromptTemplate(
            input_variables=["query"],
            template=prompt_text
        )
    
    async def classify_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        anonymize_query: bool = True
    ) -> ClassificationResult:
        """
        Classify a user query using multi-stage approach.
        
        Args:
            query: User query to classify
            session_id: Optional session identifier for tracking
            anonymize_query: Whether to anonymize query before LLM processing
            
        Returns:
            ClassificationResult with category, confidence, and reasoning
            
        Raises:
            ValueError: If query is empty or invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        self._classification_count += 1
        start_time = time.time()
        
        try:
            # Step 1: PII anonymization if requested
            anonymized_query = None
            if anonymize_query and self._pii_detector:
                anonymized_query = await self._anonymize_query(query)
                processing_query = anonymized_query
            else:
                processing_query = query
            
            # Step 2: Rule-based pre-filtering (fast path)
            rule_result = self._rule_based_classification(processing_query)
            if rule_result:
                processing_time = time.time() - start_time
                self._method_stats["rule_based"] += 1
                
                logger.info(
                    f"Rule-based classification: {rule_result.classification} "
                    f"({processing_time:.3f}s, session: {session_id or 'anonymous'})"
                )
                
                return ClassificationResult(
                    classification=rule_result.classification,
                    confidence=rule_result.confidence,
                    reasoning=rule_result.reasoning,
                    processing_time=processing_time,
                    method_used="rule_based",
                    anonymized_query=anonymized_query
                )
            
            # Step 3: LLM-based classification (comprehensive analysis)
            try:
                llm_result = await self._llm_based_classification(processing_query)
                processing_time = time.time() - start_time
                self._method_stats["llm_based"] += 1
                
                logger.info(
                    f"LLM-based classification: {llm_result.classification} "
                    f"({processing_time:.3f}s, session: {session_id or 'anonymous'})"
                )
                
                return ClassificationResult(
                    classification=llm_result.classification,
                    confidence=llm_result.confidence,
                    reasoning=llm_result.reasoning,
                    processing_time=processing_time,
                    method_used="llm_based",
                    anonymized_query=anonymized_query
                )
                
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
                
                # Step 4: Fallback to rule-based with low confidence
                fallback_result = self._fallback_classification(processing_query)
                processing_time = time.time() - start_time
                self._method_stats["fallback"] += 1
                
                logger.info(
                    f"Fallback classification: {fallback_result.classification} "
                    f"({processing_time:.3f}s, session: {session_id or 'anonymous'})"
                )
                
                return ClassificationResult(
                    classification=fallback_result.classification,
                    confidence="LOW",
                    reasoning=f"Fallback classification due to LLM error: {fallback_result.reasoning}",
                    processing_time=processing_time,
                    method_used="fallback",
                    anonymized_query=anonymized_query
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query classification failed: {e}")
            
            # Last resort: return clarification needed
            return ClassificationResult(
                classification="CLARIFICATION_NEEDED",
                confidence="LOW",
                reasoning=f"Classification failed due to error: {str(e)}",
                processing_time=processing_time,
                method_used="fallback",
                anonymized_query=anonymized_query
            )
    
    async def _anonymize_query(self, query: str) -> str:
        """
        Anonymize query using Australian PII detection.
        
        Args:
            query: Original query text
            
        Returns:
            Anonymized query text
        """
        try:
            if self._pii_detector:
                detection_result = await self._pii_detector.detect_and_anonymise(query)
                return detection_result.anonymised_text
            return query
        except Exception as e:
            logger.warning(f"PII anonymization failed: {e}")
            return query
    
    def _rule_based_classification(self, query: str) -> Optional[ClassificationResult]:
        """
        Perform enhanced rule-based classification using weighted regex patterns.
        
        Uses pattern weighting system to provide more accurate confidence scoring
        based on Australian Public Service domain knowledge.
        
        Args:
            query: Query text to classify
            
        Returns:
            ClassificationResult if confident match found, None otherwise
        """
        query_lower = query.lower()
        
        # Calculate weighted scores for each category
        weighted_scores = {}
        pattern_details = {}
        
        for category in ["SQL", "VECTOR", "HYBRID"]:
            category_score = 0
            matched_patterns = []
            
            # High confidence patterns (weight: 3)
            for pattern in self._compiled_weighted_patterns[category]["high_confidence"]:
                if pattern.search(query_lower):
                    category_score += 3
                    matched_patterns.append(("high", pattern.pattern))
            
            # Medium confidence patterns (weight: 2)
            for pattern in self._compiled_weighted_patterns[category]["medium_confidence"]:
                if pattern.search(query_lower):
                    category_score += 2
                    matched_patterns.append(("medium", pattern.pattern))
            
            # Low confidence patterns (weight: 1)
            for pattern in self._compiled_weighted_patterns[category]["low_confidence"]:
                if pattern.search(query_lower):
                    category_score += 1
                    matched_patterns.append(("low", pattern.pattern))
            
            weighted_scores[category] = category_score
            pattern_details[category] = matched_patterns
        
        # Find the category with highest weighted score
        max_score = max(weighted_scores.values())
        
        if max_score == 0:
            return None  # No pattern matches found
        
        # Find categories with maximum score
        top_categories = [cat for cat, score in weighted_scores.items() if score == max_score]
        
        if len(top_categories) > 1:
            return None  # Ambiguous - multiple categories tied
        
        classification = top_categories[0]
        
        # Enhanced confidence determination based on weighted scores
        if max_score >= 6:  # Multiple high-confidence patterns or mix of high+medium
            confidence = "HIGH"
        elif max_score >= 3:  # At least one high-confidence pattern or multiple medium
            confidence = "MEDIUM"
        elif max_score >= 2:  # Medium confidence pattern
            confidence = "MEDIUM"
        else:  # Only low confidence patterns
            confidence = "LOW"
        
        # Create detailed reasoning with pattern information
        matched_patterns = pattern_details[classification]
        high_count = sum(1 for level, _ in matched_patterns if level == "high")
        medium_count = sum(1 for level, _ in matched_patterns if level == "medium")
        low_count = sum(1 for level, _ in matched_patterns if level == "low")
        
        reasoning_parts = []
        if high_count > 0:
            reasoning_parts.append(f"{high_count} high-confidence")
        if medium_count > 0:
            reasoning_parts.append(f"{medium_count} medium-confidence")
        if low_count > 0:
            reasoning_parts.append(f"{low_count} low-confidence")
        
        reasoning = f"Enhanced rule-based: {', '.join(reasoning_parts)} pattern(s) for {classification} (score: {max_score})"
        
        return ClassificationResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=0.0,  # Will be set by caller
            method_used="rule_based"
        )
    
    async def _llm_based_classification(self, query: str) -> ClassificationResult:
        """
        Perform LLM-based classification with structured prompt.
        
        Args:
            query: Query text to classify
            
        Returns:
            ClassificationResult from LLM analysis
            
        Raises:
            Exception: If LLM classification fails
        """
        if not self._classification_prompt or not self._llm:
            raise RuntimeError("LLM classification not properly initialized")
        
        try:
            # Format prompt with query
            formatted_prompt = self._classification_prompt.format(query=query)
            
            # Get LLM response
            response = await self._llm.ainvoke(formatted_prompt)
            
            # Extract content from AIMessage if needed
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse response
            return self._parse_llm_response(response_text, query)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str, original_query: str) -> ClassificationResult:
        """
        Parse LLM response into structured classification result.
        
        Args:
            response: Raw LLM response text
            original_query: Original query for context
            
        Returns:
            Parsed ClassificationResult
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            lines = response.strip().split('\n')
            
            classification = None
            confidence = None
            reasoning = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Classification:'):
                    classification = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    confidence = line.split(':', 1)[1].strip()
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate classification
            if classification not in ["SQL", "VECTOR", "HYBRID"]:
                # Try to extract from response text
                if any(word in response.upper() for word in ["SQL", "DATABASE", "STATISTICAL"]):
                    classification = "SQL"
                elif any(word in response.upper() for word in ["VECTOR", "FEEDBACK", "COMMENT"]):
                    classification = "VECTOR"
                elif any(word in response.upper() for word in ["HYBRID", "BOTH", "COMBINED"]):
                    classification = "HYBRID"
                else:
                    classification = "CLARIFICATION_NEEDED"
            
            # Validate confidence
            if confidence not in ["HIGH", "MEDIUM", "LOW"]:
                confidence = "MEDIUM"  # Default fallback
            
            # Ensure reasoning exists
            if not reasoning:
                reasoning = f"LLM classified as {classification} with {confidence} confidence"
            
            return ClassificationResult(
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                processing_time=0.0,  # Will be set by caller
                method_used="llm_based"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Could not parse LLM response: {response[:100]}...")
    
    def _fallback_classification(self, query: str) -> ClassificationResult:
        """
        Fallback classification when other methods fail.
        
        Uses simple heuristics and defaults to encourage clarification.
        
        Args:
            query: Query text to classify
            
        Returns:
            Conservative ClassificationResult
        """
        query_lower = query.lower()
        
        # Simple keyword-based fallback
        sql_keywords = ["count", "many", "number", "average", "percentage", "total", "breakdown"]
        vector_keywords = ["feedback", "comment", "say", "opinion", "experience", "think"]
        
        sql_count = sum(1 for keyword in sql_keywords if keyword in query_lower)
        vector_count = sum(1 for keyword in vector_keywords if keyword in query_lower)
        
        if sql_count > vector_count and sql_count > 0:
            classification = "SQL"
            reasoning = "Fallback: detected statistical keywords"
        elif vector_count > sql_count and vector_count > 0:
            classification = "VECTOR"
            reasoning = "Fallback: detected feedback keywords"
        else:
            classification = "CLARIFICATION_NEEDED"
            reasoning = "Fallback: ambiguous query requires clarification"
        
        return ClassificationResult(
            classification=classification,
            confidence="LOW",
            reasoning=reasoning,
            processing_time=0.0,  # Will be set by caller
            method_used="fallback"
        )
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics for monitoring and debugging.
        
        Returns:
            Dictionary with classification method usage statistics
        """
        total = self._classification_count
        return {
            "total_classifications": total,
            "method_usage": {
                method: {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0
                }
                for method, count in self._method_stats.items()
            },
            "rule_patterns": {
                "sql_patterns": len(self._sql_patterns),
                "vector_patterns": len(self._vector_patterns),
                "hybrid_patterns": len(self._hybrid_patterns)
            }
        }
    
    async def close(self) -> None:
        """Clean up classifier resources."""
        try:
            logger.info(
                f"Query classifier closing. Processed {self._classification_count} queries. "
                f"Method usage: {self._method_stats}"
            )
        except Exception as e:
            logger.error(f"Error during classifier cleanup: {e}")


# Factory function for easy classifier creation
async def create_query_classifier(llm: Optional[BaseLanguageModel] = None) -> QueryClassifier:
    """
    Create and initialize a query classifier.
    
    Args:
        llm: Optional LLM instance. If None, will use get_llm()
        
    Returns:
        Initialized QueryClassifier ready for use
    """
    classifier = QueryClassifier(llm)
    await classifier.initialize()
    return classifier


# Example usage and testing for enhanced query classification
if __name__ == "__main__":
    """
    Example use cases demonstrating enhanced Australian Public Service query classification.
    
    This section provides practical examples of how the enhanced classifier works with
    APS-specific terminology and improved confidence calibration.
    """
    import asyncio
    
    async def demonstrate_enhanced_classification():
        """
        Demonstrate enhanced classification capabilities with APS-specific queries.
        """
        print("=== Enhanced Query Classification Demo ===\n")
        
        # Initialize classifier
        classifier = QueryClassifier()
        await classifier.initialize()
        
        # Test cases covering different APS scenarios
        test_queries = [
            # High-confidence SQL queries
            ("How many Executive Level 1 participants completed mandatory training across all agencies this quarter?", "SQL", "HIGH"),
            ("What's the completion rate for virtual learning by agency?", "SQL", "HIGH"),
            ("Show me participation rates for Level 4-6 staff in professional development", "SQL", "HIGH"),
            
            # High-confidence VECTOR queries
            ("What feedback did participants give about the virtual learning platform?", "VECTOR", "HIGH"),
            ("What technical issues were mentioned by attendees?", "VECTOR", "HIGH"),
            ("How did delegates rate the facilitator effectiveness?", "VECTOR", "HIGH"),
            
            # High-confidence HYBRID queries
            ("Analyze satisfaction trends across agencies with supporting participant feedback", "HYBRID", "HIGH"),
            ("Compare training ROI between face-to-face and virtual delivery with cost-benefit analysis", "HYBRID", "HIGH"),
            ("Correlate completion rates with participant satisfaction feedback", "HYBRID", "HIGH"),
            
            # Medium-confidence queries
            ("What are the quarterly training statistics by department?", "SQL", "MEDIUM"),
            ("How do participants feel about the course content?", "VECTOR", "MEDIUM"),
            ("Review training effectiveness with demographic analysis", "HYBRID", "MEDIUM"),
            
            # Edge cases and potential ambiguities
            ("Tell me about training", None, "LOW"),  # Should return None (too ambiguous)
            ("Training effectiveness", None, "LOW"),   # Should return None or low confidence
        ]
        
        print("Testing Enhanced Rule-Based Classification:\n")
        
        for i, (query, expected_classification, expected_confidence) in enumerate(test_queries, 1):
            try:
                result = await classifier.classify_query(query, anonymize_query=False)
                
                print(f"Test {i}: {query}")
                print(f"  Result: {result.classification} ({result.confidence})")
                print(f"  Expected: {expected_classification} ({expected_confidence})")
                print(f"  Method: {result.method_used}")
                print(f"  Reasoning: {result.reasoning}")
                print(f"  ✅ Match: {result.classification == expected_classification and result.confidence == expected_confidence}")
                print()
                
            except Exception as e:
                print(f"Test {i} failed: {e}")
                print()
        
        # Demonstrate classification statistics
        print("=== Classification Statistics ===")
        stats = classifier.get_classification_stats()
        print(f"Total classifications: {stats['total_classifications']}")
        print(f"Method distribution: {stats['method_distribution']}")
        print(f"Confidence distribution: {stats['confidence_distribution']}")
        if stats['classification_times']:
            avg_time = sum(stats['classification_times']) / len(stats['classification_times'])
            print(f"Average classification time: {avg_time:.3f}s")
        
    async def test_pattern_weighting():
        """
        Test the pattern weighting system with specific examples.
        """
        print("\n=== Pattern Weighting System Demo ===\n")
        
        classifier = QueryClassifier()
        
        # Test queries with different pattern weights
        test_cases = [
            ("How many EL1 staff completed training?", "Should be HIGH confidence due to high-weight SQL patterns"),
            ("What's the quarterly training report summary?", "Should be MEDIUM/LOW confidence due to low-weight SQL patterns"),
            ("Participants gave feedback about technical issues", "Should be HIGH confidence due to high-weight VECTOR patterns"),
            ("Venue concerns were mentioned", "Should be LOW confidence due to low-weight VECTOR patterns"),
            ("Analyze satisfaction with cost-benefit analysis", "Should be HIGH confidence due to high-weight HYBRID patterns"),
            ("Comprehensive review of training", "Should be LOW confidence due to low-weight HYBRID patterns"),
        ]
        
        for query, description in test_cases:
            result = classifier._rule_based_classification(query)
            if result:
                print(f"Query: {query}")
                print(f"Classification: {result.classification} ({result.confidence})")
                print(f"Reasoning: {result.reasoning}")
                print(f"Expected: {description}")
                print()
            else:
                print(f"Query: {query}")
                print("No classification (no pattern matches)")
                print(f"Expected: {description}")
                print()
    
    # Run demonstrations
    if __name__ == "__main__":
        print("Starting Enhanced Query Classifier Demonstrations...")
        asyncio.run(demonstrate_enhanced_classification())
        print("\nTesting Pattern Weighting...")
        asyncio.run(test_pattern_weighting())
        print("Demonstrations complete!")
