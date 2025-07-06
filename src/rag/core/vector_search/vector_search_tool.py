"""
Vector Search Tool for RAG System

This module implements a privacy-compliant, async vector search tool that enables
semantic search over evaluation feedback while maintaining Australian data governance
standards.

Key Features:
- Async-first design for LangGraph integration
- Automatic query anonymization using existing PII detection
- Rich metadata filtering with privacy-safe operations
- Performance monitoring and audit logging
- Configurable similarity thresholds and result limits
- Clean LangChain tool interface for agent orchestration

Classes:
- VectorSearchTool: Main LangChain tool for semantic search
- SearchParameters: Configuration container for search operations

Usage Example:
    # Initialize and use vector search tool
    tool = VectorSearchTool()
    await tool.initialize()
    
    # Basic semantic search
    response = await tool.search(
        query="feedback about technical issues",
        max_results=10,
        similarity_threshold=0.65
    )
    
    # Search with metadata filtering
    response = await tool.search(
        query="course effectiveness feedback",
        filters={
            "user_level": ["Level 5", "Level 6"],
            "agency": "Department of Finance",
            "sentiment": {"type": "negative", "min_score": 0.6}
        }
    )
    
    # LangChain tool usage
    result = await tool.ainvoke({
        "query": "What did senior staff say about virtual learning?",
        "filters": {"user_level": ["Level 5", "Level 6", "Exec Level 1", "Exec Level 2"]},
        "max_results": 15
    })
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from .search_result import (
    VectorSearchResponse, VectorSearchResult, SearchMetadata,
    RelevanceCategory
)
from .embedder import Embedder
from ...data.embeddings_manager import EmbeddingsManager
from ...core.privacy.pii_detector import AustralianPIIDetector
from ...config.settings import get_settings
from ...utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class SearchParameters:
    """Configuration parameters for vector search operations."""
    query: str
    max_results: int = 10
    similarity_threshold: float = 0.65
    filters: Optional[Dict[str, Any]] = None
    field_names: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Validate search parameters."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        if self.max_results <= 0 or self.max_results > 100:
            raise ValueError("max_results must be between 1 and 100")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")


class VectorSearchInput(BaseModel):
    """Input schema for VectorSearchTool."""
    query: str = Field(description="Natural language query for semantic search")
    max_results: int = Field(default=10, description="Maximum number of results to return (1-100)")
    similarity_threshold: float = Field(default=0.65, description="Minimum similarity score (0.0-1.0)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for search refinement")


class VectorSearchTool(BaseTool):
    """
    Privacy-compliant async vector search tool for the RAG system.
    
    Provides semantic search capabilities over evaluation feedback with
    automatic PII protection and rich metadata filtering.
    """
    
    name: str = "vector_search"
    description: str = """
    Search through course evaluation feedback using semantic similarity.
    
    This tool finds relevant feedback and comments from course evaluations based on
    natural language queries. It automatically anonymizes queries for privacy and
    supports filtering by user level, agency, sentiment, and other metadata.
    
    Input should be a JSON object with:
    - query: Natural language search query (required)
    - max_results: Maximum results to return, 1-100 (default: 10)
    - similarity_threshold: Minimum similarity score, 0.0-1.0 (default: 0.75)
    - filters: Optional metadata filters as JSON object
    
    Example filters:
    {
        "user_level": ["Level 5", "Level 6"],
        "agency": "Department of Finance",
        "sentiment": {"type": "negative", "min_score": 0.6},
        "field_name": ["general_feedback", "did_experience_issue_detail"]
    }
    """
    args_schema: Type[VectorSearchInput] = VectorSearchInput
    
    # Pydantic v2 requires model configuration to allow extra fields
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def __init__(self):
        super().__init__()
        self._settings = get_settings()
        self._embedder: Optional[Embedder] = None
        self._embeddings_manager: Optional[EmbeddingsManager] = None
        self._pii_detector: Optional[AustralianPIIDetector] = None
        self._initialized = False
        
        # Default search configuration
        self.default_field_names = [
            "general_feedback",
            "did_experience_issue_detail", 
            "course_application_other"
        ]
        
        # Performance tracking
        self._search_count = 0
        self._total_search_time = 0.0
        
    async def initialize(self) -> None:
        """
        Initialize the vector search tool with required components.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return
            
        try:
            logger.info("Initializing VectorSearchTool...")
            
            # Initialize components
            self._embedder = Embedder()
            await self._embedder.initialize()
            
            self._embeddings_manager = EmbeddingsManager()
            await self._embeddings_manager.initialize()
            
            self._pii_detector = AustralianPIIDetector()
            
            self._initialized = True
            logger.info("VectorSearchTool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorSearchTool: {e}")
            raise RuntimeError(f"VectorSearchTool initialization failed: {e}")
    
    def _validate_initialized(self) -> None:
        """Ensure tool is properly initialized."""
        if not self._initialized:
            raise RuntimeError("VectorSearchTool not initialized. Call initialize() first.")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        similarity_threshold: float = 0.65,
        filters: Optional[Dict[str, Any]] = None,
        field_names: Optional[List[str]] = None
    ) -> VectorSearchResponse:
        """
        Perform semantic search over evaluation feedback.
        
        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
            filters: Optional metadata filters
            field_names: Specific fields to search (default: all feedback fields)
            
        Returns:
            VectorSearchResponse with results and metadata
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If search operation fails
        """
        self._validate_initialized()
        
        # Validate parameters
        params = SearchParameters(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            filters=filters or {},
            field_names=field_names or self.default_field_names
        )
        params.validate()
        
        start_time = time.time()
        
        try:
            # Step 1: Anonymize query for privacy compliance
            anonymized_query = await self._anonymize_query(query)
            
            # Step 2: Generate query embedding
            embedding_start = time.time()
            query_embedding = await self._get_query_embedding(anonymized_query)
            embedding_time = time.time() - embedding_start
            
            # Step 3: Perform vector search with filters
            search_start = time.time()
            raw_results = await self._perform_vector_search(
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                filters=filters,
                field_names=params.field_names
            )
            search_time = time.time() - search_start
            
            # Step 4: Process and format results
            filtering_start = time.time()
            processed_results = await self._process_search_results(raw_results)
            filtering_time = time.time() - filtering_start
            
            total_time = time.time() - start_time
            
            # Step 5: Create response with performance metrics
            response = VectorSearchResponse(
                query=query,
                results=processed_results,
                total_results=len(raw_results),
                processing_time=total_time,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                filters_applied=filters or {},
                embedding_time=embedding_time,
                search_time=search_time,
                filtering_time=filtering_time,
                query_anonymized=anonymized_query != query,
                privacy_controls_applied=["pii_detection", "query_anonymization"]
            )
            
            # Step 6: Log search for audit trail
            await self._log_search_operation(response)
            
            # Update performance metrics
            self._search_count += 1
            self._total_search_time += total_time
            
            return response
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    async def _anonymize_query(self, query: str) -> str:
        """Anonymize query using PII detection for privacy compliance."""
        try:
            result = await self._pii_detector.detect_and_anonymise(query)
            anonymized_query = result.anonymised_text if result.anonymised_text else query
            
            if result.entities_detected:
                logger.info(f"PII detected and anonymized in query: {[e.get('entity_type', 'unknown') for e in result.entities_detected]}")
            
            return anonymized_query
            
        except Exception as e:
            logger.warning(f"Query anonymization failed, using original query: {e}")
            return query
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the search query."""
        try:
            result = await self._embedder.embed_text(query)
            return result.embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    async def _perform_vector_search(
        self,
        query_embedding: List[float],
        similarity_threshold: float,
        max_results: int,
        filters: Optional[Dict[str, Any]],
        field_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Perform the actual vector similarity search."""
        try:
            # Build metadata filters
            metadata_filters = self._build_metadata_filters(filters, field_names)
            
            # Perform similarity search using embeddings manager
            results = await self._embeddings_manager.search_similar_with_metadata(
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                limit=max_results,
                metadata_filters=metadata_filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search execution failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}")
    
    def _build_metadata_filters(
        self,
        filters: Optional[Dict[str, Any]],
        field_names: List[str]
    ) -> Dict[str, Any]:
        """Build metadata filters for database query."""
        metadata_filters = {}
        
        # Always filter by field names
        if field_names:
            metadata_filters['field_name'] = field_names
        
        if not filters:
            return metadata_filters
        
        # User level filtering
        if 'user_level' in filters:
            metadata_filters['user_level'] = filters['user_level']
        
        # Agency filtering
        if 'agency' in filters:
            metadata_filters['agency'] = filters['agency']
        
        # Sentiment filtering
        if 'sentiment' in filters:
            sentiment_filter = filters['sentiment']
            if isinstance(sentiment_filter, dict):
                sentiment_type = sentiment_filter.get('type')
                min_score = sentiment_filter.get('min_score', 0.5)
                metadata_filters['sentiment'] = {
                    'type': sentiment_type,
                    'min_score': min_score
                }
        
        # Course delivery type filtering
        if 'course_delivery_type' in filters:
            metadata_filters['course_delivery_type'] = filters['course_delivery_type']
        
        # Knowledge level filtering
        if 'knowledge_level_prior' in filters:
            metadata_filters['knowledge_level_prior'] = filters['knowledge_level_prior']
        
        return metadata_filters
    
    async def _process_search_results(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> List[VectorSearchResult]:
        """Process raw search results into structured result objects."""
        processed_results = []
        
        for raw_result in raw_results:
            try:
                # Extract metadata
                metadata_dict = raw_result.get('metadata', {})
                
                metadata = SearchMetadata(
                    user_level=metadata_dict.get('user_level'),
                    agency=metadata_dict.get('agency'),
                    knowledge_level_prior=metadata_dict.get('knowledge_level_prior'),
                    course_delivery_type=metadata_dict.get('course_delivery_type'),
                    course_end_date=metadata_dict.get('course_end_date'),
                    field_name=raw_result.get('field_name', ''),
                    response_id=raw_result.get('response_id'),
                    chunk_index=raw_result.get('chunk_index'),
                    sentiment_scores=metadata_dict.get('sentiment_scores'),
                    facilitator_skills=metadata_dict.get('facilitator_skills'),
                    had_guest_speakers=metadata_dict.get('had_guest_speakers'),
                    model_version=raw_result.get('model_version', ''),
                    created_at=raw_result.get('created_at')
                )
                
                # Create result object
                result = VectorSearchResult(
                    chunk_text=raw_result.get('chunk_text', ''),
                    similarity_score=raw_result.get('similarity_score', 0.0),
                    metadata=metadata,
                    embedding_id=raw_result.get('embedding_id')
                )
                
                processed_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to process search result: {e}")
                continue
        
        return processed_results
    
    async def _log_search_operation(self, response: VectorSearchResponse) -> None:
        """Log search operation for audit trail."""
        try:
            audit_data = {
                "operation": "vector_search",
                "query_length": len(response.query),
                "result_count": response.result_count,
                "processing_time": response.processing_time,
                "similarity_threshold": response.similarity_threshold,
                "filters_applied": list(response.filters_applied.keys()),
                "query_anonymized": response.query_anonymized,
                "privacy_controls": response.privacy_controls_applied,
                "has_high_quality_results": response.has_high_quality_results
            }
            
            logger.info(f"Vector search completed: {audit_data}")
            
        except Exception as e:
            logger.warning(f"Failed to log search operation: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring and optimization."""
        if self._search_count == 0:
            return {"searches_performed": 0}
        
        return {
            "searches_performed": self._search_count,
            "total_search_time": self._total_search_time,
            "average_search_time": self._total_search_time / self._search_count,
            "initialization_status": "initialized" if self._initialized else "not_initialized"
        }
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Synchronous run method for LangChain compatibility."""
        # Convert to async and run
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, this shouldn't be called
            raise RuntimeError("Use ainvoke() for async execution")
        else:
            return loop.run_until_complete(self._arun(query, **kwargs))
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Async run method for LangChain tool interface."""
        try:
            # Ensure tool is initialized
            if not self._initialized:
                await self.initialize()
            
            # Extract parameters from kwargs
            max_results = kwargs.get('max_results', 10)
            similarity_threshold = kwargs.get('similarity_threshold', 0.75)
            filters = kwargs.get('filters')
            
            # Perform search
            response = await self.search(
                query=query,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            
            # Format response for LangChain
            if not response.results:
                return f"No relevant feedback found for query: '{query}'"
            
            # Create summary
            summary_parts = [
                f"Found {response.result_count} relevant feedback items:",
                f"Average relevance: {response.average_similarity:.2f}",
                f"Processing time: {response.processing_time:.2f}s",
                ""
            ]
            
            # Add top results
            for i, result in enumerate(response.results[:5], 1):
                summary_parts.extend([
                    f"{i}. [{result.relevance_category.value}] {result.user_context}",
                    f"   Sentiment: {result.sentiment_summary}",
                    f"   Feedback: {result.chunk_text[:200]}{'...' if len(result.chunk_text) > 200 else ''}",
                    ""
                ])
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Vector search tool execution failed: {e}")
            return f"Search failed: {str(e)}"
