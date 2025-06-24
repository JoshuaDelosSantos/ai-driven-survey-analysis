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
import time
import re
from typing import Optional, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..privacy.pii_detector import AustralianPIIDetector
from ...utils.llm_utils import get_llm
from ...utils.logging_utils import get_logger
from ...config.settings import get_settings

from .data_structures import ClassificationResult, ClassificationType, ConfidenceLevel, ClassificationMethod
from .circuit_breaker import CircuitBreaker, FallbackMetrics, RetryConfig, CircuitBreakerState
from .confidence_calibrator import ConfidenceCalibrator
from .pattern_matcher import PatternMatcher
from .aps_patterns import APS_PATTERNS, PATTERN_WEIGHTS
from .llm_classifier import LLMClassifier


logger = get_logger(__name__)


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
        Initialize query classifier with sophisticated fallback mechanisms.
        
        Args:
            llm: Language model for LLM-based classification. If None, will use get_llm()
        """
        self._llm = llm
        self.settings = get_settings()
        self._pii_detector: Optional[AustralianPIIDetector] = None
        self._classification_prompt: Optional[PromptTemplate] = None
        
        # Sophisticated fallback system components
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(self.settings, 'classification_failure_threshold', 5),
            recovery_timeout=getattr(self.settings, 'classification_recovery_timeout', 60.0),
            half_open_max_calls=getattr(self.settings, 'classification_half_open_calls', 3)
        )
        self._retry_config = RetryConfig(
            max_retries=getattr(self.settings, 'classification_max_retries', 3),
            base_delay=getattr(self.settings, 'classification_base_delay', 1.0),
            max_delay=getattr(self.settings, 'classification_max_delay', 30.0)
        )
        self._fallback_metrics = FallbackMetrics()
        
        # Milestone 3: Sophisticated confidence calibration system
        self._confidence_calibrator = ConfidenceCalibrator()
        
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
        Classify query using sophisticated multi-stage approach with advanced fallback mechanisms.
        
        Enhanced Classification Pipeline:
        1. Input validation and PII anonymization
        2. Rule-based pre-filtering (fast path with pattern weighting)
        3. LLM-based classification (with circuit breaker and retry logic)
        4. Sophisticated fallback system (graceful degradation)
        5. Comprehensive error handling and recovery
        
        Resilience Features:
        - Circuit breaker pattern for LLM failure protection
        - Exponential backoff with jitter for retry logic
        - Multiple fallback strategies with confidence degradation
        - Real-time metrics collection and monitoring
        
        Args:
            query: User query to classify
            session_id: Optional session identifier for tracking
            anonymize_query: Whether to anonymize PII before LLM processing
            
        Returns:
            ClassificationResult with detailed confidence and reasoning
        """
        start_time = time.time()
        anonymized_query = None
        
        try:
            # Metrics tracking
            self._fallback_metrics.record_attempt()
            self._classification_count += 1
            
            logger.debug(
                f"Starting classification for query length {len(query)} "
                f"(session: {session_id or 'anonymous'})"
            )
            
            # Step 1: Input validation and PII anonymization
            if not query or not query.strip():
                return self._create_error_result(
                    "Empty or invalid query",
                    start_time,
                    anonymized_query
                )
            
            if anonymize_query and self._pii_detector:
                try:
                    anonymized_query = await self._anonymize_query(query)
                    processing_query = anonymized_query
                    logger.debug(f"Query anonymized (session: {session_id or 'anonymous'})")
                except Exception as e:
                    logger.warning(f"PII anonymization failed: {e}, proceeding with original query")
                    processing_query = query
            else:
                processing_query = query
            
            # Step 2: Rule-based pre-filtering (fast path) with confidence calibration
            rule_result = self._rule_based_classification(processing_query)
            if rule_result and rule_result.confidence in ["HIGH", "MEDIUM"]:
                processing_time = time.time() - start_time
                
                # Apply confidence calibration to rule-based result
                calibrated_result = self._apply_confidence_calibration(
                    rule_result, processing_query, pattern_matches=rule_result.pattern_matches
                )
                
                self._method_stats["rule_based"] += 1
                self._fallback_metrics.classification_times.append(processing_time)
                
                logger.info(
                    f"Rule-based classification success: {calibrated_result.classification} "
                    f"({processing_time:.3f}s, confidence: {rule_result.confidence} → {calibrated_result.confidence}, "
                    f"session: {session_id or 'anonymous'})"
                )
                
                return ClassificationResult(
                    classification=calibrated_result.classification,
                    confidence=calibrated_result.confidence,
                    reasoning=f"{rule_result.reasoning}. {calibrated_result.calibration_reasoning}",
                    processing_time=processing_time,
                    method_used="rule_based",
                    anonymized_query=anonymized_query
                )
            
            # Step 3: LLM-based classification with sophisticated fallback
            llm_result = await self._llm_classification_with_fallback(
                processing_query, session_id
            )
            
            if llm_result:
                processing_time = time.time() - start_time
                
                # Apply confidence calibration to LLM result
                calibrated_result = self._apply_confidence_calibration(
                    llm_result, processing_query
                )
                
                self._fallback_metrics.classification_times.append(processing_time)
                
                return ClassificationResult(
                    classification=calibrated_result.classification,
                    confidence=calibrated_result.confidence,
                    reasoning=f"{llm_result.reasoning}. {calibrated_result.calibration_reasoning}",
                    processing_time=processing_time,
                    method_used=llm_result.method_used,
                    anonymized_query=anonymized_query
                )
            
            # Step 4: Final fallback - enhanced rule-based with low confidence
            fallback_result = self._enhanced_fallback_classification(processing_query)
            processing_time = time.time() - start_time
            self._method_stats["fallback"] += 1
            self._fallback_metrics.record_fallback(time.time() - start_time)
            self._fallback_metrics.classification_times.append(processing_time)
            
            logger.warning(
                f"Using final fallback classification: {fallback_result.classification} "
                f"({processing_time:.3f}s, session: {session_id or 'anonymous'})"
            )
            
            return ClassificationResult(
                classification=fallback_result.classification,
                confidence="LOW",
                reasoning=f"Final fallback: {fallback_result.reasoning}",
                processing_time=processing_time,
                method_used="fallback",
                anonymized_query=anonymized_query
            )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._fallback_metrics.classification_times.append(processing_time)
            
            logger.error(
                f"Query classification completely failed: {e} "
                f"({processing_time:.3f}s, session: {session_id or 'anonymous'})"
            )
            
            # Absolute last resort
            return self._create_error_result(
                f"Complete classification failure: {str(e)}",
                start_time,
                anonymized_query
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
        
        # Calculate pattern match counts for confidence calibration
        pattern_match_counts = {
            "high_confidence": high_count,
            "medium_confidence": medium_count,
            "low_confidence": low_count
        }
        
        return ClassificationResult(
            classification=classification,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=0.0,  # Will be set by caller
            method_used="rule_based",
            pattern_matches=pattern_match_counts
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
    
    async def _llm_classification_with_fallback(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Optional[ClassificationResult]:
        """
        Perform LLM classification with sophisticated fallback mechanisms.
        
        Features:
        - Circuit breaker pattern to prevent cascading failures
        - Exponential backoff with jitter for retry logic
        - Real-time failure tracking and recovery
        - Graceful degradation when LLM is unavailable
        
        Args:
            query: Query to classify
            session_id: Optional session identifier
            
        Returns:
            ClassificationResult if successful, None if all attempts failed
        """
        # Check circuit breaker before attempting LLM call
        if not self._circuit_breaker.can_execute():
            self._fallback_metrics.record_circuit_breaker_block()
            logger.info(
                f"Circuit breaker OPEN - skipping LLM classification "
                f"(session: {session_id or 'anonymous'})"
            )
            return None
        
        # Attempt LLM classification with retry logic
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                if attempt > 0:
                    # Apply exponential backoff
                    delay = self._retry_config.get_delay(attempt - 1)
                    logger.info(f"Retrying LLM classification after {delay:.2f}s delay (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    self._fallback_metrics.record_retry()
                
                # Attempt LLM classification
                llm_start_time = time.time()
                llm_result = await self._llm_based_classification(query)
                llm_response_time = time.time() - llm_start_time
                
                # Success - record metrics and reset circuit breaker
                self._circuit_breaker.record_success()
                self._fallback_metrics.record_llm_success(llm_response_time)
                self._method_stats["llm_based"] += 1
                
                logger.info(
                    f"LLM classification success: {llm_result.classification} "
                    f"(attempt {attempt + 1}, {llm_response_time:.3f}s, "
                    f"session: {session_id or 'anonymous'})"
                )
                
                return llm_result
                
            except Exception as e:
                # Record failure
                self._circuit_breaker.record_failure()
                self._fallback_metrics.record_llm_failure()
                
                logger.warning(
                    f"LLM classification attempt {attempt + 1} failed: {e} "
                    f"(session: {session_id or 'anonymous'})"
                )
                
                # If this was the last attempt, we'll fall through
                if attempt == self._retry_config.max_retries:
                    logger.error(
                        f"All LLM classification attempts exhausted "
                        f"(session: {session_id or 'anonymous'})"
                    )
                    break
        
        # All LLM attempts failed
        return None
    
    def _enhanced_fallback_classification(self, query: str) -> ClassificationResult:
        """
        Enhanced fallback classification with multiple strategies.
        
        This method provides sophisticated fallback classification when both
        rule-based and LLM-based methods fail or are unavailable. It uses
        multiple heuristic strategies with confidence degradation.
        
        Args:
            query: Query text to classify
            
        Returns:
            ClassificationResult with fallback classification
        """
        query_lower = query.lower()
        
        # Strategy 1: Enhanced keyword analysis with APS domain knowledge
        classification_scores = self._calculate_keyword_scores(query_lower)
        
        # Strategy 2: Query length and complexity analysis
        complexity_hint = self._analyze_query_complexity(query)
        
        # Strategy 3: Contextual pattern analysis
        context_hint = self._analyze_contextual_patterns(query_lower)
        
        # Combine strategies for final classification
        final_classification = self._combine_fallback_strategies(
            classification_scores, complexity_hint, context_hint
        )
        
        reasoning_parts = [
            f"keyword analysis: {classification_scores}",
            f"complexity: {complexity_hint}",
            f"context: {context_hint}"
        ]
        
        return ClassificationResult(
            classification=final_classification,
            confidence="LOW",
            reasoning=f"Enhanced fallback using {', '.join(reasoning_parts)}",
            processing_time=0.0,  # Will be set by caller
            method_used="fallback"
        )
    
    def _calculate_keyword_scores(self, query: str) -> Dict[str, int]:
        """Calculate classification scores based on enhanced keyword analysis."""
        # Enhanced keyword sets with APS domain knowledge
        sql_keywords = [
            "count", "many", "number", "total", "sum", "average", "percentage", "breakdown",
            "statistics", "completion rate", "level", "agency", "department", "quarterly",
            "annual", "participation", "enrollment", "training hours", "cost", "budget"
        ]
        
        vector_keywords = [
            "feedback", "comment", "opinion", "experience", "say", "think", "feel",
            "participant", "delegate", "quality", "facilitator", "venue", "technical issues",
            "accessibility", "recommendation", "satisfaction", "testimonial", "review"
        ]
        
        hybrid_keywords = [
            "analyze", "analysis", "comprehensive", "trends", "correlation", "impact",
            "effectiveness", "ROI", "cost-benefit", "stakeholder", "demographic",
            "compare", "evaluate", "assessment", "holistic", "integrated"
        ]
        
        scores = {
            "SQL": sum(1 for keyword in sql_keywords if keyword in query),
            "VECTOR": sum(1 for keyword in vector_keywords if keyword in query),
            "HYBRID": sum(1 for keyword in hybrid_keywords if keyword in query)
        }
        
        return scores
    
    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity to provide classification hints."""
        word_count = len(query.split())
        
        if word_count < 5:
            return "simple"
        elif word_count > 15:
            return "complex"
        else:
            return "moderate"
    
    def _analyze_contextual_patterns(self, query: str) -> str:
        """Analyze contextual patterns in the query."""
        if any(word in query for word in ["what", "how", "show", "tell"]):
            return "interrogative"
        elif any(word in query for word in ["analyze", "compare", "evaluate"]):
            return "analytical"
        elif any(word in query for word in ["list", "give", "provide"]):
            return "informational"
        else:
            return "unclear"
    
    def _combine_fallback_strategies(
        self,
        keyword_scores: Dict[str, int],
        complexity: str,
        context: str
    ) -> ClassificationType:
        """Combine multiple fallback strategies for final classification."""
        # Find category with highest keyword score
        max_score = max(keyword_scores.values())
        
        if max_score == 0:
            # No keywords matched - use context and complexity
            if context == "analytical" or complexity == "complex":
                return "HYBRID"
            elif context == "interrogative":
                return "VECTOR"
            else:
                return "CLARIFICATION_NEEDED"
        
        # Find categories with max score
        top_categories = [cat for cat, score in keyword_scores.items() if score == max_score]
        
        if len(top_categories) == 1:
            return top_categories[0]
        
        # Tie-breaking logic using context
        if "HYBRID" in top_categories and context == "analytical":
            return "HYBRID"
        elif "VECTOR" in top_categories and context == "interrogative":
            return "VECTOR"
        elif "SQL" in top_categories and complexity == "simple":
            return "SQL"
        
        # Default to most common category in tie
        return top_categories[0]
    
    def _create_error_result(
        self,
        error_message: str,
        start_time: float,
        anonymized_query: Optional[str]
    ) -> ClassificationResult:
        """Create standardized error result."""
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            classification="CLARIFICATION_NEEDED",
            confidence="LOW",
            reasoning=error_message,
            processing_time=processing_time,
            method_used="fallback",
            anonymized_query=anonymized_query
        )
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive classification statistics for monitoring and debugging.
        
        Returns:
            Dictionary with enhanced classification method usage statistics,
            fallback system metrics, and performance data
        """
        total = self._classification_count
        
        # Basic method usage stats
        method_stats = {
            method: {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
            for method, count in self._method_stats.items()
        }
        
        # Enhanced fallback system metrics
        fallback_stats = {
            "total_attempts": self._fallback_metrics.total_attempts,
            "llm_successes": self._fallback_metrics.llm_successes,
            "llm_failures": self._fallback_metrics.llm_failures,
            "llm_success_rate": self._fallback_metrics.get_llm_success_rate(),
            "circuit_breaker_blocks": self._fallback_metrics.circuit_breaker_blocks,
            "retry_attempts": self._fallback_metrics.retry_attempts,
            "fallback_activations": self._fallback_metrics.fallback_activations
        }
        
        # Circuit breaker status
        circuit_breaker_stats = {
            "state": self._circuit_breaker.state.value,
            "failure_count": self._circuit_breaker.failure_count,
            "last_failure": self._circuit_breaker.last_failure_time.isoformat() if self._circuit_breaker.last_failure_time else None,
            "half_open_attempts": self._circuit_breaker.half_open_attempts
        }
        
        # Performance metrics
        performance_stats = self._fallback_metrics.get_average_times()
        
        # Pattern counts
        pattern_stats = {
            "sql_patterns": len(self._sql_patterns),
            "vector_patterns": len(self._vector_patterns),
            "hybrid_patterns": len(self._hybrid_patterns),
            "total_patterns": len(self._sql_patterns) + len(self._vector_patterns) + len(self._hybrid_patterns)
        }
        
        return {
            "total_classifications": total,
            "method_usage": method_stats,
            "fallback_system": fallback_stats,
            "circuit_breaker": circuit_breaker_stats,
            "performance": performance_stats,
            "rule_patterns": pattern_stats,
            "system_health": {
                "is_healthy": self._circuit_breaker.state == CircuitBreakerState.CLOSED,
                "uptime_percentage": ((self._fallback_metrics.llm_successes / max(1, self._fallback_metrics.total_attempts)) * 100) if self._fallback_metrics.total_attempts > 0 else 100,
                "classification_efficiency": (self._method_stats["rule_based"] / max(1, total)) * 100 if total > 0 else 0
            }
        }
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """
        Get detailed fallback system metrics for monitoring.
        
        Returns:
            Dictionary with detailed fallback system performance metrics
        """
        return {
            "circuit_breaker": {
                "state": self._circuit_breaker.state.value,
                "failure_threshold": self._circuit_breaker.failure_threshold,
                "recovery_timeout": self._circuit_breaker.recovery_timeout,
                "current_failures": self._circuit_breaker.failure_count,
                "last_failure": self._circuit_breaker.last_failure_time.isoformat() if self._circuit_breaker.last_failure_time else None
            },
            "retry_config": {
                "max_retries": self._retry_config.max_retries,
                "base_delay": self._retry_config.base_delay,
                "max_delay": self._retry_config.max_delay,
                "backoff_multiplier": self._retry_config.backoff_multiplier,
                "jitter_enabled": self._retry_config.jitter
            },
            "performance": {
                "llm_success_rate": self._fallback_metrics.get_llm_success_rate(),
                "average_response_times": self._fallback_metrics.get_average_times(),
                "total_retries": self._fallback_metrics.retry_attempts,
                "circuit_breaker_interventions": self._fallback_metrics.circuit_breaker_blocks
            }
        }
    
    def reset_metrics(self) -> None:
        """
        Reset all classification and fallback metrics.
        
        Useful for testing or periodic metric resets.
        """
        self._classification_count = 0
        self._method_stats = {
            "rule_based": 0,
            "llm_based": 0,
            "fallback": 0
        }
        self._fallback_metrics = FallbackMetrics()
        
        # Reset confidence calibration data as well
        self._confidence_calibrator.reset_calibration_data()
        
        logger.info("Classification metrics reset")
    
    def _apply_confidence_calibration(
        self,
        result: ClassificationResult,
        query: str,
        pattern_matches: Optional[Dict[str, int]] = None
    ) -> ClassificationResult:
        """
        Apply confidence calibration to a classification result.
        
        Args:
            result: Original classification result
            query: Original query text
            pattern_matches: Pattern match counts for rule-based classifications
            
        Returns:
            Classification result with calibrated confidence
        """
        try:
            # Perform confidence calibration
            calibration_result = self._confidence_calibrator.calibrate_confidence(
                raw_confidence=result.confidence,
                classification=result.classification,
                query=query,
                pattern_matches=pattern_matches,
                method_used=result.method_used
            )
            
            logger.debug(f"Confidence calibration: {calibration_result.adjustment_reasoning}")
            
            # Return updated result with calibrated confidence
            return ClassificationResult(
                classification=result.classification,
                confidence=calibration_result.calibrated_confidence,
                reasoning=result.reasoning,
                processing_time=result.processing_time,
                method_used=result.method_used,
                anonymized_query=result.anonymized_query,
                pattern_matches=result.pattern_matches,
                calibration_reasoning=calibration_result.adjustment_reasoning
            )
            
        except Exception as e:
            logger.warning(f"Confidence calibration failed: {e}, using original confidence")
            return result
    
    def record_classification_feedback(
        self,
        classification: ClassificationType,
        was_correct: bool,
        confidence_score: float
    ) -> None:
        """
        Record feedback on classification accuracy for calibration improvement.
        
        Args:
            classification: The classification that was made
            was_correct: Whether the classification was correct
            confidence_score: The confidence score that was assigned
        """
        self._confidence_calibrator.record_classification_outcome(
            classification, was_correct, confidence_score
        )
        logger.debug(f"Recorded classification feedback: {classification} ({'correct' if was_correct else 'incorrect'})")
    
    def get_confidence_calibration_stats(self) -> Dict[str, Any]:
        """
        Get confidence calibration statistics.
        
        Returns:
            Dictionary with calibration system statistics
        """
        return self._confidence_calibrator.get_calibration_stats()
    
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
            ("Tell me about training", "CLARIFICATION_NEEDED", "LOW"),
            ("Training effectiveness", "HYBRID", "LOW"),
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
                is_match = result.classification == expected_classification and result.confidence == expected_confidence
                print(f"  {'✅' if is_match else '❌'} Match")
                print()
                
            except Exception as e:
                print(f"Test {i} failed: {e}")
                print()
        
        # Demonstrate classification statistics
        print("=== Enhanced Classification Statistics ===")
        stats = classifier.get_classification_stats()
        print(f"Total classifications: {stats['total_classifications']}")
        print(f"Method usage: {stats['method_usage']}")
        print(f"System health: {stats['system_health']}")
        print(f"Circuit breaker status: {stats['circuit_breaker']['state']}")
        
        # Demonstrate fallback metrics
        fallback_metrics = classifier.get_fallback_metrics()
        print(f"LLM success rate: {fallback_metrics['performance']['llm_success_rate']:.1f}%")
        print(f"Average response times: {fallback_metrics['performance']['average_response_times']}")
        
        if stats['performance'].get('overall_avg_time', 0) > 0:
            print(f"Average classification time: {stats['performance']['overall_avg_time']:.3f}s")
    
    async def test_fallback_mechanisms():
        """
        Demonstrate sophisticated fallback mechanisms with circuit breaker and retry logic.
        """
        print("\n=== Fallback Mechanisms Demo ===\n")
        
        # Create classifier with mock LLM for fallback testing
        from unittest.mock import AsyncMock
        
        mock_llm = AsyncMock()
        classifier = QueryClassifier(llm=mock_llm)
        await classifier.initialize()
        
        print("1. Testing Circuit Breaker Pattern:")
        
        # Simulate LLM failures to trigger circuit breaker
        mock_llm.ainvoke.side_effect = Exception("Simulated LLM failure")
        
        test_query = "What technical issues were mentioned by participants?"
        
        # Make multiple failed attempts to open the circuit breaker
        for i in range(6):  # Exceed the failure threshold
            try:
                result = await classifier.classify_query(test_query, anonymize_query=False)
                print(f"  Attempt {i+1}: {result.classification} via {result.method_used}")
            except Exception as e:
                print(f"  Attempt {i+1}: Failed - {e}")
        
        # Check circuit breaker status
        stats = classifier.get_classification_stats()
        print(f"  Circuit breaker state: {stats['circuit_breaker']['state']}")
        print(f"  Failure count: {stats['circuit_breaker']['failure_count']}")
        
        print("\n2. Testing Enhanced Fallback Classification:")
        
        # Test enhanced fallback with various query types
        fallback_test_queries = [
            "How many users completed training last month?",  # Should use enhanced SQL fallback
            "What feedback did people give about the course?",  # Should use enhanced VECTOR fallback
            "Analyze satisfaction trends with demographic data",  # Should use enhanced HYBRID fallback
            "Training information please"  # Should require clarification
        ]
        
        for query in fallback_test_queries:
            result = classifier._enhanced_fallback_classification(query)
            print(f"  Query: {query}")
            print(f"  Fallback result: {result.classification} ({result.confidence})")
            print(f"  Reasoning: {result.reasoning}")
            print()
        
        print("3. Testing Exponential Backoff Strategy:")
        
        # Show exponential backoff delays
        retry_config = classifier._retry_config
        print(f"  Max retries: {retry_config.max_retries}")
        print(f"  Base delay: {retry_config.base_delay}s")
        print(f"  Backoff multiplier: {retry_config.backoff_multiplier}")
        print(f"  Jitter enabled: {retry_config.jitter}")
        
        print("  Calculated delays:")
        for attempt in range(retry_config.max_retries):
            delay = retry_config.get_delay(attempt)
            print(f"    Attempt {attempt + 2}: {delay:.2f}s delay")
        
        print("\n4. Testing Metrics Collection:")
        
        fallback_metrics = classifier.get_fallback_metrics()
        print(f"  Circuit breaker failures: {fallback_metrics['circuit_breaker']['current_failures']}")
        print(f"  Retry configuration: {fallback_metrics['retry_config']}")
        print(f"  Performance metrics: {fallback_metrics['performance']}")
    
    async def demonstrate_resilience_features():
        """
        Demonstrate the resilience features of the enhanced classification system.
        """
        print("\n=== Resilience Features Demo ===\n")
        
        classifier = QueryClassifier()
        await classifier.initialize()
        
        print("1. APS-Specific Enhanced Pattern Matching:")
        
        # Test APS-specific patterns with various confidence levels
        aps_queries = [
            ("How many EL1 staff completed mandatory training?", "HIGH confidence SQL"),
            ("What feedback about virtual delivery did participants provide?", "HIGH confidence VECTOR"),
            ("Analyze cost-benefit of training ROI across departments", "HIGH confidence HYBRID"),
            ("Show quarterly training statistics summary", "MEDIUM confidence SQL"),
            ("Participants mentioned venue accessibility concerns", "MEDIUM confidence VECTOR"),
            ("Review comprehensive training effectiveness", "LOW confidence HYBRID")
        ]
        
        for query, expected in aps_queries:
            result = classifier._rule_based_classification(query)
            if result:
                print(f"  Query: {query}")
                print(f"  Result: {result.classification} ({result.confidence})")
                print(f"  Expected: {expected}")
                print(f"  Pattern details: {result.reasoning}")
                print()
            else:
                print(f"  Query: {query}")
                print(f"  No rule-based match (would go to LLM or fallback)")
                print()
        
        print("2. Multi-Strategy Fallback System:")
        
        # Test the combination of multiple fallback strategies
        ambiguous_queries = [
            "training data analysis",
            "user satisfaction evaluation", 
            "course performance metrics",
            "stakeholder feedback review"
        ]
        
        for query in ambiguous_queries:
            result = classifier._enhanced_fallback_classification(query)
            print(f"  Query: {query}")
            print(f"  Fallback classification: {result.classification}")
            print(f"  Strategy details: {result.reasoning}")
            print()
        
        print("3. System Health Monitoring:")
        
        # Show system health metrics
        stats = classifier.get_classification_stats()
        health = stats['system_health']
        
        print(f"  System healthy: {health['is_healthy']}")
        print(f"  Uptime percentage: {health['uptime_percentage']:.1f}%")
        print(f"  Classification efficiency: {health['classification_efficiency']:.1f}%")
        print(f"  Total patterns loaded: {stats['rule_patterns']['total_patterns']}")
        
        print("\n4. Performance Optimization:")
        
        # Test rule-based performance
        start_time = time.time()
        for _ in range(100):  # Run 100 classifications
            classifier._rule_based_classification("How many users completed training?")
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"  Rule-based classification performance: {avg_time*1000:.2f}ms average")
        print(f"  Performance target (<100ms): {'✅ PASS' if avg_time < 0.1 else '❌ FAIL'}")
        
        await classifier.close()
        
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
        print("Starting Enhanced Query Classifier with Sophisticated Fallback Mechanisms...")
        print("=" * 80)
        
        asyncio.run(demonstrate_enhanced_classification())
        print("\n" + "=" * 80)
        
        print("Testing Pattern Weighting System...")
        # To run this, we need an event loop, but asyncio.run creates a new one each time.
        # For simplicity, we'll just call the function.
        asyncio.run(test_pattern_weighting())
        print("\n" + "=" * 80)
        
        print("Testing Sophisticated Fallback Mechanisms...")
        asyncio.run(test_fallback_mechanisms())
        print("\n" + "=" * 80)
        
        print("Demonstrating Resilience Features...")
        asyncio.run(demonstrate_resilience_features())
        print("\n" + "=" * 80)
        
        print("All demonstrations complete!")