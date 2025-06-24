"""
LLM-based query classification with structured prompts.

This module handles LLM-based classification for complex queries that
cannot be reliably handled by rule-based patterns. It provides sophisticated
prompt engineering and response parsing for accurate classification.

Example Usage:
    # Initialize LLM classifier
    llm_classifier = LLMClassifier()
    await llm_classifier.initialize()
    
    # Classify complex queries requiring semantic understanding
    result = await llm_classifier.classify_query(
        "I need insights on training effectiveness across different demographics"
    )
    print(f"Classification: {result.classification}")  # HYBRID
    print(f"Confidence: {result.confidence}")          # MEDIUM/HIGH
    print(f"Reasoning: {result.reasoning}")            # LLM explanation
    
    # Handle ambiguous queries
    result = await llm_classifier.classify_query(
        "Show me something about user engagement patterns"
    )
    print(f"Classification: {result.classification}")  # VECTOR or HYBRID
    print(f"Method: {result.method_used}")             # llm_based
    
    # Process queries with contextual nuances
    result = await llm_classifier.classify_query(
        "Compare participation rates while considering participant feedback quality"
    )
    print(f"Classification: {result.classification}")  # HYBRID
    print(f"Processing time: {result.processing_time:.3f}s")
"""

import time
from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from .data_structures import ClassificationResult, ClassificationType
from ...utils.llm_utils import get_llm
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMClassifier:
    """
    LLM-based query classifier with sophisticated prompt engineering.
    
    Handles complex queries that require semantic understanding and
    contextual analysis beyond simple pattern matching.
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize LLM classifier.
        
        Args:
            llm: Language model instance. If None, will use get_llm()
        """
        self._llm = llm
        self._classification_prompt: Optional[PromptTemplate] = None
        self._setup_classification_prompt()
    
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
    
    async def initialize(self) -> None:
        """
        Initialize LLM classifier components.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize LLM if not provided
            if self._llm is None:
                self._llm = get_llm()
                logger.info(f"LLM initialized for classification: {type(self._llm).__name__}")
            
            # Validate prompt setup
            if not self._classification_prompt:
                raise RuntimeError("Classification prompt not properly initialized")
            
            logger.info("LLM classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"LLM classifier initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize LLM classifier: {e}")
    
    async def classify_query(self, query: str) -> ClassificationResult:
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
        
        start_time = time.time()
        
        try:
            # Format prompt with query
            formatted_prompt = self._classification_prompt.format(query=query)
            
            logger.debug(f"Sending query to LLM for classification: {query[:100]}...")
            
            # Get LLM response
            response = await self._llm.ainvoke(formatted_prompt)
            
            # Extract content from AIMessage if needed
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.debug(f"LLM response received: {response_text[:200]}...")
            
            # Parse response
            result = self._parse_llm_response(response_text, query)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.debug(
                f"LLM classification completed: {result.classification} "
                f"({result.confidence}, {processing_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LLM classification failed after {processing_time:.3f}s: {e}")
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
            
            # Parse structured response
            for line in lines:
                line = line.strip()
                if line.startswith('Classification:'):
                    classification = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    confidence = line.split(':', 1)[1].strip()
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate and clean classification
            classification = self._validate_classification(classification, response)
            confidence = self._validate_confidence(confidence)
            reasoning = self._validate_reasoning(reasoning, classification, confidence)
            
            logger.debug(
                f"Parsed LLM response: {classification} ({confidence}) - {reasoning[:100]}..."
            )
            
            return ClassificationResult(
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                processing_time=0.0,  # Will be set by caller
                method_used="llm_based"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            raise ValueError(f"Could not parse LLM response: {response[:100]}...")
    
    def _validate_classification(self, classification: Optional[str], full_response: str) -> ClassificationType:
        """
        Validate and clean classification from LLM response.
        
        Args:
            classification: Extracted classification
            full_response: Full LLM response for fallback parsing
            
        Returns:
            Valid ClassificationType
        """
        # Clean up classification if present
        if classification:
            classification = classification.upper().strip()
            
            # Handle common variations
            if classification in ["SQL", "DATABASE", "STATISTICAL"]:
                return "SQL"
            elif classification in ["VECTOR", "SEMANTIC", "FEEDBACK", "COMMENTS"]:
                return "VECTOR"
            elif classification in ["HYBRID", "BOTH", "COMBINED", "MIXED"]:
                return "HYBRID"
        
        # Fallback: analyze full response for classification keywords
        response_upper = full_response.upper()
        
        # Count occurrences of classification indicators
        sql_indicators = ["SQL", "DATABASE", "STATISTICAL", "NUMBERS", "COUNT", "AGGREGATE"]
        vector_indicators = ["VECTOR", "SEMANTIC", "FEEDBACK", "COMMENTS", "OPINIONS"]
        hybrid_indicators = ["HYBRID", "BOTH", "COMBINED", "MIXED", "ANALYSIS"]
        
        sql_score = sum(1 for indicator in sql_indicators if indicator in response_upper)
        vector_score = sum(1 for indicator in vector_indicators if indicator in response_upper)
        hybrid_score = sum(1 for indicator in hybrid_indicators if indicator in response_upper)
        
        # Return the classification with the highest score
        if hybrid_score > max(sql_score, vector_score):
            return "HYBRID"
        elif sql_score > vector_score:
            return "SQL"
        elif vector_score > 0:
            return "VECTOR"
        else:
            logger.warning(f"Could not determine classification from response, defaulting to CLARIFICATION_NEEDED")
            return "CLARIFICATION_NEEDED"
    
    def _validate_confidence(self, confidence: Optional[str]) -> str:
        """
        Validate and clean confidence from LLM response.
        
        Args:
            confidence: Extracted confidence level
            
        Returns:
            Valid confidence level
        """
        if confidence:
            confidence = confidence.upper().strip()
            
            if confidence in ["HIGH", "H"]:
                return "HIGH"
            elif confidence in ["MEDIUM", "MED", "M"]:
                return "MEDIUM"
            elif confidence in ["LOW", "L"]:
                return "LOW"
        
        # Default to MEDIUM if unclear
        logger.debug("Could not determine confidence level, defaulting to MEDIUM")
        return "MEDIUM"
    
    def _validate_reasoning(
        self, 
        reasoning: Optional[str], 
        classification: ClassificationType, 
        confidence: str
    ) -> str:
        """
        Validate and enhance reasoning from LLM response.
        
        Args:
            reasoning: Extracted reasoning
            classification: Determined classification
            confidence: Determined confidence
            
        Returns:
            Valid reasoning string
        """
        if reasoning and reasoning.strip():
            return reasoning.strip()
        
        # Generate default reasoning if none provided
        default_reasoning = f"LLM classified as {classification} with {confidence} confidence"
        logger.debug("No reasoning provided by LLM, using default")
        return default_reasoning
    
    def test_llm_connectivity(self) -> dict:
        """
        Test LLM connectivity and response quality.
        
        Returns:
            Dictionary with connectivity test results
        """
        test_results = {
            "llm_available": False,
            "response_time": None,
            "response_quality": None,
            "error": None
        }
        
        try:
            if not self._llm:
                test_results["error"] = "LLM not initialized"
                return test_results
            
            # Test with a simple classification query
            test_query = "How many users completed training last quarter?"
            
            start_time = time.time()
            
            # Use synchronous invoke for testing
            if hasattr(self._llm, 'invoke'):
                response = self._llm.invoke(self._classification_prompt.format(query=test_query))
                response_time = time.time() - start_time
                
                test_results["llm_available"] = True
                test_results["response_time"] = response_time
                
                # Analyze response quality
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Check if response contains expected elements
                response_upper = response_text.upper()
                has_classification = any(word in response_upper for word in ["SQL", "VECTOR", "HYBRID"])
                has_confidence = any(word in response_upper for word in ["HIGH", "MEDIUM", "LOW"])
                has_reasoning = "REASONING" in response_upper or len(response_text) > 50
                
                quality_score = sum([has_classification, has_confidence, has_reasoning]) / 3
                test_results["response_quality"] = quality_score
                
                logger.info(f"LLM connectivity test passed: {response_time:.3f}s, quality: {quality_score:.2f}")
            else:
                test_results["error"] = "LLM does not support synchronous invoke"
                
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"LLM connectivity test failed: {e}")
        
        return test_results
    
    def get_prompt_template(self) -> str:
        """
        Get the current classification prompt template.
        
        Returns:
            Prompt template string
        """
        if self._classification_prompt:
            return self._classification_prompt.template
        return "No prompt template configured"
    
    def update_prompt_template(self, new_template: str) -> None:
        """
        Update the classification prompt template.
        
        Args:
            new_template: New prompt template string
        """
        try:
            self._classification_prompt = PromptTemplate(
                input_variables=["query"],
                template=new_template
            )
            logger.info("Classification prompt template updated")
        except Exception as e:
            logger.error(f"Failed to update prompt template: {e}")
            raise ValueError(f"Invalid prompt template: {e}")
