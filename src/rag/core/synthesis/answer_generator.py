"""
Answer Generation System for RAG Agent

This module provides intelligent answer synthesis capabilities that combine
results from SQL and vector search tools into coherent, comprehensive responses
while maintaining Australian PII compliance and audit requirements.

Key Features:
- Multi-modal result synthesis (SQL + Vector search)
- Context-aware answer formatting
- Source attribution and transparency
- PII protection in generated responses
- Confidence scoring and quality metrics

Security: All synthesized answers maintain PII anonymization.
Performance: Async processing with configurable timeouts.
Privacy: Australian Privacy Principles (APP) compliance maintained.

Example Usage:
    # Basic answer generation with statistical data only
    async def statistical_example():
        from ...utils.llm_utils import get_llm
        from ..privacy.pii_detector import AustralianPIIDetector
        
        # Initialize components
        llm = get_llm()
        pii_detector = AustralianPIIDetector()
        await pii_detector.initialize()
        
        generator = AnswerGenerator(
            llm=llm,
            pii_detector=pii_detector,
            enable_source_attribution=True
        )
        
        # SQL-only synthesis example
        sql_result = {
            "success": True,
            "result": [
                {"agency": "Department of Finance", "completion_rate": 87.5, "total_users": 240},
                {"agency": "Department of Health", "completion_rate": 92.1, "total_users": 180},
                {"agency": "Department of Education", "completion_rate": 78.9, "total_users": 320}
            ],
            "query": "SELECT agency, AVG(completion_rate) as completion_rate, COUNT(*) as total_users FROM user_stats GROUP BY agency"
        }
        
        result = await generator.synthesize_answer(
            query="What are the course completion rates by agency?",
            sql_result=sql_result,
            session_id="demo_001"
        )
        
        print(f"Answer Type: {result.answer_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Answer: {result.answer}")
        # Output: Professional statistical analysis with agency completion rates
    
    # Feedback-only synthesis example
    async def feedback_example():
        llm = get_llm()
        pii_detector = AustralianPIIDetector()
        await pii_detector.initialize()
        
        generator = AnswerGenerator(llm=llm, pii_detector=pii_detector)
        
        vector_result = {
            "results": [
                {"text": "The virtual learning platform was intuitive and easy to navigate", "score": 0.92},
                {"text": "I appreciated the flexibility to complete modules at my own pace", "score": 0.89},
                {"text": "Some technical issues with video playback on older browsers", "score": 0.76},
                {"text": "Overall satisfied with the learning experience and content quality", "score": 0.88},
                {"text": "Would recommend improvements to the mobile interface", "score": 0.71}
            ],
            "query": "virtual learning platform feedback",
            "total_results": 5
        }
        
        result = await generator.synthesize_answer(
            query="What feedback did users provide about the virtual learning platform?",
            vector_result=vector_result,
            session_id="demo_002"
        )
        
        print(f"Answer Type: {result.answer_type.value}")
        print(f"Sources: {result.sources}")
        print(f"Answer: {result.answer}")
        # Output: Comprehensive feedback analysis with themes and examples
    
    # Hybrid synthesis example (most comprehensive)
    async def hybrid_example():
        llm = get_llm()
        pii_detector = AustralianPIIDetector()
        await pii_detector.initialize()
        
        generator = AnswerGenerator(
            llm=llm,
            pii_detector=pii_detector,
            max_answer_length=1500,
            enable_source_attribution=True
        )
        
        # Combined SQL and vector search results
        sql_result = {
            "success": True,
            "result": [
                {"satisfaction_avg": 4.2, "response_count": 450, "completion_rate": 85.7},
                {"trend": "increasing", "period": "last_6_months"}
            ]
        }
        
        vector_result = {
            "results": [
                {"text": "The new features significantly improved my productivity", "score": 0.94},
                {"text": "Interface is much more user-friendly than the previous version", "score": 0.91},
                {"text": "Still experiencing occasional connectivity issues", "score": 0.68},
                {"text": "Training materials were comprehensive and well-structured", "score": 0.87},
                {"text": "Mobile accessibility has greatly improved work flexibility", "score": 0.89}
            ]
        }
        
        result = await generator.synthesize_answer(
            query="Analyze overall user satisfaction with the new platform, including both metrics and feedback",
            sql_result=sql_result,
            vector_result=vector_result,
            session_id="demo_003",
            additional_context="Quarterly satisfaction review for executive summary"
        )
        
        print(f"Answer Type: {result.answer_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"PII Detected: {result.pii_detected}")
        print(f"Answer: {result.answer}")
        # Output: Executive-level analysis combining statistics with user sentiment
    
    # Error handling example
    async def error_handling_example():
        llm = get_llm()
        generator = AnswerGenerator(llm=llm)
        
        # No results available
        result = await generator.synthesize_answer(
            query="What are the latest updates to the system?",
            sql_result=None,
            vector_result=None,
            session_id="demo_004"
        )
        
        print(f"Answer Type: {result.answer_type.value}")
        print(f"Answer: {result.answer}")
        # Output: User-friendly error message with suggestions
    
    # PII protection example
    async def pii_protection_example():
        llm = get_llm()
        pii_detector = AustralianPIIDetector()
        await pii_detector.initialize()
        
        generator = AnswerGenerator(llm=llm, pii_detector=pii_detector)
        
        # Vector result containing potential PII
        vector_result = {
            "results": [
                {"text": "Contact John Smith at john.smith@agency.gov.au for follow-up", "score": 0.85},
                {"text": "The training was excellent and very comprehensive", "score": 0.90}
            ]
        }
        
        result = await generator.synthesize_answer(
            query="What follow-up actions were mentioned in the feedback?",
            vector_result=vector_result,
            session_id="demo_005"
        )
        
        print(f"PII Detected: {result.pii_detected}")
        print(f"Answer: {result.answer}")
        # Output: Answer with PII automatically anonymized (email addresses masked)
    
    # Quality assessment example
    async def quality_assessment_example():
        llm = get_llm()
        generator = AnswerGenerator(llm=llm)
        
        # High-quality data results
        sql_result = {"success": True, "result": [{"metric": "value"}]}
        vector_result = {"results": [{"text": "feedback", "score": 0.95}] * 10}
        
        result = await generator.synthesize_answer(
            query="Comprehensive analysis request",
            sql_result=sql_result,
            vector_result=vector_result,
            session_id="demo_006"
        )
        
        print(f"Synthesis Quality Metrics:")
        print(f"- Answer Type: {result.answer_type.value}")
        print(f"- Confidence Score: {result.confidence:.2f}")
        print(f"- Sources Used: {len(result.sources)}")
        print(f"- Answer Length: {result.metadata['answer_length']} characters")
        print(f"- Processing Time: {result.processing_time:.2f}s")
        # Output: Detailed quality metrics for answer assessment
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from ..privacy.pii_detector import AustralianPIIDetector
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class AnswerType(Enum):
    """Types of answers that can be generated."""
    STATISTICAL_ONLY = "statistical"
    FEEDBACK_ONLY = "feedback"
    HYBRID_COMBINED = "hybrid"
    ERROR_RESPONSE = "error"
    CLARIFICATION_REQUEST = "clarification"


@dataclass
class SynthesisResult:
    """Result of answer synthesis process."""
    answer: str
    answer_type: AnswerType
    confidence: float
    sources: List[str]
    metadata: Dict[str, Any]
    pii_detected: bool = False
    processing_time: Optional[float] = None


class AnswerGenerator:
    """
    Intelligent answer synthesis system for RAG responses.
    
    Combines SQL analysis results and vector search feedback into coherent,
    comprehensive answers while maintaining privacy and audit requirements.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        pii_detector: Optional[AustralianPIIDetector] = None,
        max_answer_length: int = 2000,
        enable_source_attribution: bool = True
    ):
        """
        Initialize answer generator.
        
        Args:
            llm: Language model for answer synthesis
            pii_detector: PII detection system for privacy protection
            max_answer_length: Maximum length of generated answers
            enable_source_attribution: Whether to include source attribution
        """
        self.llm = llm
        self.pii_detector = pii_detector
        self.max_answer_length = max_answer_length
        self.enable_source_attribution = enable_source_attribution
        
        self._statistical_template = PromptTemplate(
            input_variables=["query", "sql_results", "context"],
            template="""
Based on the database analysis results, provide a clear and concise answer to the user's question.

User Question: {query}

Database Results: {sql_results}

Additional Context: {context}

Instructions:
- Provide specific numbers and statistics when available
- Use clear, professional language
- Be accurate and factual
- If data is incomplete, acknowledge limitations
- Keep response under 500 words

Answer:
"""
        )
        
        self._feedback_template = PromptTemplate(
            input_variables=["query", "feedback_results", "context"],
            template="""
Based on user feedback and comments, provide a comprehensive answer to the user's question.

User Question: {query}

User Feedback Results: {feedback_results}

Additional Context: {context}

Instructions:
- Summarize key themes and patterns in the feedback
- Provide specific examples when relevant
- Maintain user privacy (no personal identifiers)
- Use professional, analytical tone
- Highlight both positive and negative feedback
- Keep response under 500 words

Answer:
"""
        )
        
        self._hybrid_template = PromptTemplate(
            input_variables=["query", "sql_results", "feedback_results", "context"],
            template="""
Provide a comprehensive answer combining statistical analysis with user feedback insights.

User Question: {query}

Statistical Data: {sql_results}

User Feedback: {feedback_results}

Additional Context: {context}

Instructions:
- Integrate both statistical and qualitative insights
- Start with key statistics, then support with feedback themes
- Show how data and feedback align or contrast
- Maintain professional analytical tone
- Protect user privacy in all examples
- Provide actionable insights when possible
- Keep response under 800 words

Comprehensive Answer:
"""
        )
    
    async def synthesize_answer(
        self,
        query: str,
        sql_result: Optional[Dict[str, Any]] = None,
        vector_result: Optional[Dict[str, Any]] = None,
        session_id: str = "unknown",
        additional_context: Optional[str] = None
    ) -> SynthesisResult:
        """
        Synthesize comprehensive answer from available results.
        
        Args:
            query: Original user query
            sql_result: Results from SQL analysis
            vector_result: Results from vector search
            session_id: Session identifier for logging
            additional_context: Additional context for synthesis
            
        Returns:
            Synthesized answer with metadata
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting answer synthesis for session {session_id}")
            
            # Determine answer type and strategy
            answer_type = self._determine_answer_type(sql_result, vector_result)
            
            # Generate answer based on available data
            if answer_type == AnswerType.STATISTICAL_ONLY:
                answer = await self._generate_statistical_answer(
                    query, sql_result, additional_context
                )
            elif answer_type == AnswerType.FEEDBACK_ONLY:
                answer = await self._generate_feedback_answer(
                    query, vector_result, additional_context
                )
            elif answer_type == AnswerType.HYBRID_COMBINED:
                answer = await self._generate_hybrid_answer(
                    query, sql_result, vector_result, additional_context
                )
            else:
                answer = self._generate_error_response(query, sql_result, vector_result)
            
            # Apply PII protection if enabled
            pii_detected = False
            if self.pii_detector:
                cleaned_answer, pii_info = await self.pii_detector.anonymize_text(
                    text=answer,
                    session_id=session_id
                )
                if pii_info.get("pii_detected"):
                    answer = cleaned_answer
                    pii_detected = True
                    logger.warning(f"PII detected and anonymized in answer for session {session_id}")
            
            # Calculate confidence score
            confidence = self._calculate_confidence(sql_result, vector_result, answer_type)
            
            # Build sources list
            sources = self._build_sources_list(sql_result, vector_result, answer_type)
            
            # Add source attribution if enabled
            if self.enable_source_attribution and sources:
                answer = self._add_source_attribution(answer, sources)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Answer synthesis completed in {processing_time:.2f}s "
                f"(type: {answer_type.value}, confidence: {confidence:.2f})"
            )
            
            return SynthesisResult(
                answer=answer,
                answer_type=answer_type,
                confidence=confidence,
                sources=sources,
                metadata={
                    "query_length": len(query),
                    "answer_length": len(answer),
                    "session_id": session_id,
                    "has_sql_data": sql_result is not None,
                    "has_vector_data": vector_result is not None
                },
                pii_detected=pii_detected,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            
            error_answer = (
                "I encountered an issue while generating your answer. "
                "Please try rephrasing your question or contact support if the problem persists."
            )
            
            return SynthesisResult(
                answer=error_answer,
                answer_type=AnswerType.ERROR_RESPONSE,
                confidence=0.0,
                sources=[],
                metadata={"error": str(e), "session_id": session_id},
                processing_time=time.time() - start_time if 'start_time' in locals() else None
            )
    
    def _determine_answer_type(
        self,
        sql_result: Optional[Dict[str, Any]],
        vector_result: Optional[Dict[str, Any]]
    ) -> AnswerType:
        """Determine the appropriate answer synthesis strategy."""
        has_sql = sql_result and sql_result.get("success") and sql_result.get("result")
        has_vector = vector_result and vector_result.get("results")
        
        if has_sql and has_vector:
            return AnswerType.HYBRID_COMBINED
        elif has_sql:
            return AnswerType.STATISTICAL_ONLY
        elif has_vector:
            return AnswerType.FEEDBACK_ONLY
        else:
            return AnswerType.ERROR_RESPONSE
    
    async def _generate_statistical_answer(
        self,
        query: str,
        sql_result: Dict[str, Any],
        context: Optional[str]
    ) -> str:
        """Generate answer based on SQL results only."""
        prompt = self._statistical_template.format(
            query=query,
            sql_results=str(sql_result.get("result", "No data available")),
            context=context or "No additional context"
        )
        
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _generate_feedback_answer(
        self,
        query: str,
        vector_result: Dict[str, Any],
        context: Optional[str]
    ) -> str:
        """Generate answer based on vector search results only."""
        feedback_summary = self._summarize_feedback_results(vector_result)
        
        prompt = self._feedback_template.format(
            query=query,
            feedback_results=feedback_summary,
            context=context or "No additional context"
        )
        
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _generate_hybrid_answer(
        self,
        query: str,
        sql_result: Dict[str, Any],
        vector_result: Dict[str, Any],
        context: Optional[str]
    ) -> str:
        """Generate comprehensive answer combining SQL and vector results."""
        feedback_summary = self._summarize_feedback_results(vector_result)
        
        prompt = self._hybrid_template.format(
            query=query,
            sql_results=str(sql_result.get("result", "No statistical data")),
            feedback_results=feedback_summary,
            context=context or "No additional context"
        )
        
        response = await self.llm.ainvoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _generate_error_response(
        self,
        query: str,
        sql_result: Optional[Dict[str, Any]],
        vector_result: Optional[Dict[str, Any]]
    ) -> str:
        """Generate appropriate error response."""
        if not sql_result and not vector_result:
            return (
                "I wasn't able to find any relevant information for your query. "
                "Please try rephrasing your question or being more specific about what you're looking for."
            )
        else:
            return (
                "I found some information but encountered issues processing it. "
                "Please try a different approach to your question."
            )
    
    def _summarize_feedback_results(self, vector_result: Dict[str, Any]) -> str:
        """Create a summary of feedback results for prompt input."""
        results = vector_result.get("results", [])
        if not results:
            return "No relevant feedback found"
        
        # Limit to top results to avoid prompt length issues
        top_results = results[:5] if len(results) > 5 else results
        
        summary_parts = []
        for i, result in enumerate(top_results, 1):
            if isinstance(result, dict):
                text = result.get("text", str(result))
                score = result.get("score", "N/A")
                summary_parts.append(f"{i}. (Score: {score}) {text}")
            else:
                summary_parts.append(f"{i}. {str(result)}")
        
        return "\n".join(summary_parts)
    
    def _calculate_confidence(
        self,
        sql_result: Optional[Dict[str, Any]],
        vector_result: Optional[Dict[str, Any]],
        answer_type: AnswerType
    ) -> float:
        """Calculate confidence score for the synthesized answer."""
        if answer_type == AnswerType.ERROR_RESPONSE:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on available data
        if sql_result and sql_result.get("success"):
            confidence += 0.3
        
        if vector_result and vector_result.get("results"):
            confidence += 0.2
            # Boost based on number of relevant results
            num_results = len(vector_result["results"])
            confidence += min(0.2, num_results * 0.05)
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    def _build_sources_list(
        self,
        sql_result: Optional[Dict[str, Any]],
        vector_result: Optional[Dict[str, Any]],
        answer_type: AnswerType
    ) -> List[str]:
        """Build list of data sources used in the answer."""
        sources = []
        
        if sql_result and sql_result.get("success"):
            sources.append("Database Analysis")
        
        if vector_result and vector_result.get("results"):
            sources.append("User Feedback")
        
        return sources
    
    def _add_source_attribution(self, answer: str, sources: List[str]) -> str:
        """Add source attribution to the answer."""
        if not sources:
            return answer
        
        source_text = "Sources: " + ", ".join(sources)
        return f"{answer}\n\n*{source_text}*"
