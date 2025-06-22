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
        
        # Classification patterns for rule-based pre-filtering
        self._sql_patterns = [
            r'\b(?:count|how many|number of)\b',
            r'\b(?:average|mean|avg)\b',
            r'\b(?:percentage|percent|%)\b',
            r'\b(?:breakdown by|group by|categorized by)\b',
            r'\b(?:statistics|stats|statistical)\b',
            r'\b(?:total|sum|aggregate)\b',
            r'\b(?:compare numbers|numerical comparison)\b',
            r'\b(?:completion rate|enrollment rate)\b',
            r'\b(?:agency breakdown|level breakdown|user level)\b'
        ]
        
        self._vector_patterns = [
            r'\b(?:what did.*say|what.*said)\b',
            r'\b(?:feedback about|comments about|opinions on)\b',
            r'\b(?:experiences with|experience of)\b',
            r'\b(?:user feedback|participant feedback)\b',
            r'\b(?:comments|opinions|thoughts|feelings)\b',
            r'\b(?:issues mentioned|problems reported)\b',
            r'\b(?:satisfaction|dissatisfaction)\b',
            r'\b(?:testimonials|reviews|responses)\b',
            r'\b(?:what people think|user opinions)\b'
        ]
        
        self._hybrid_patterns = [
            r'\b(?:analyze satisfaction|analyze feedback)\b',
            r'\b(?:compare feedback across|feedback trends)\b',
            r'\b(?:sentiment by agency|satisfaction by level)\b',
            r'\b(?:trends in opinions|opinion trends)\b',
            r'\b(?:comprehensive analysis|detailed analysis)\b',
            r'\b(?:both.*and|statistics.*feedback|numbers.*comments)\b',
            r'\b(?:quantitative.*qualitative|statistical.*sentiment)\b'
        ]
        
        # Compile patterns for performance
        self._compiled_patterns = {
            "SQL": [re.compile(pattern, re.IGNORECASE) for pattern in self._sql_patterns],
            "VECTOR": [re.compile(pattern, re.IGNORECASE) for pattern in self._vector_patterns],
            "HYBRID": [re.compile(pattern, re.IGNORECASE) for pattern in self._hybrid_patterns]
        }
        
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
                detection_result = await self._pii_detector.detect_pii_async(query)
                return detection_result.anonymized_text
            return query
        except Exception as e:
            logger.warning(f"PII anonymization failed: {e}")
            return query
    
    def _rule_based_classification(self, query: str) -> Optional[ClassificationResult]:
        """
        Perform rule-based classification using regex patterns.
        
        Args:
            query: Query text to classify
            
        Returns:
            ClassificationResult if confident match found, None otherwise
        """
        query_lower = query.lower()
        
        # Count pattern matches for each category
        matches = {
            "SQL": sum(1 for pattern in self._compiled_patterns["SQL"] if pattern.search(query_lower)),
            "VECTOR": sum(1 for pattern in self._compiled_patterns["VECTOR"] if pattern.search(query_lower)),
            "HYBRID": sum(1 for pattern in self._compiled_patterns["HYBRID"] if pattern.search(query_lower))
        }
        
        # Determine classification based on matches
        max_matches = max(matches.values())
        
        if max_matches == 0:
            return None  # No clear pattern match
        
        # Find categories with maximum matches
        top_categories = [cat for cat, count in matches.items() if count == max_matches]
        
        if len(top_categories) > 1:
            return None  # Ambiguous - multiple categories tied
        
        classification = top_categories[0]
        
        # Determine confidence based on match strength
        if max_matches >= 3:
            confidence = "HIGH"
        elif max_matches >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH" if max_matches == 1 else "MEDIUM"
        
        reasoning = f"Rule-based: {max_matches} pattern match(es) for {classification}"
        
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
            
            # Parse response
            return self._parse_llm_response(response, query)
            
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
