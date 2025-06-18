"""
Vector Search Result Data Structures

This module provides data structures for vector search results, enabling
clean separation between search operations and result handling.

Key Features:
- Type-safe result containers with rich metadata
- Performance metrics tracking for optimization
- Similarity scoring with confidence indicators
- Extensible design for future enhancements

Classes:
- VectorSearchResult: Individual search result with metadata
- VectorSearchResponse: Complete search response with aggregated metrics
- SearchMetadata: Rich metadata container for filtering and analysis

Usage Example:
    # Basic search result handling
    result = VectorSearchResult(
        chunk_text="Course was very helpful for my role",
        similarity_score=0.87,
        metadata={
            "user_level": "Level 5",
            "agency": "Department of Finance",
            "sentiment_scores": {"positive": 0.85, "neutral": 0.15, "negative": 0.0}
        }
    )
    
    # Access result properties
    print(f"Relevance: {result.relevance_category}")  # "High"
    print(f"User context: {result.user_context}")
    
    # Complete search response
    response = VectorSearchResponse(
        query="feedback about course effectiveness",
        results=[result1, result2, result3],
        total_results=150,
        processing_time=0.234
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class RelevanceCategory(Enum):
    """Relevance categories based on similarity scores."""
    HIGH = "High"           # >= 0.80
    MEDIUM = "Medium"       # >= 0.65
    LOW = "Low"             # >= 0.50
    WEAK = "Weak"           # < 0.50


@dataclass
class SearchMetadata:
    """
    Rich metadata container for search results.
    
    Provides structured access to evaluation context and analytics data
    while maintaining privacy compliance.
    """
    # User context (anonymized)
    user_level: Optional[str] = None
    agency: Optional[str] = None
    knowledge_level_prior: Optional[str] = None
    
    # Course context
    course_delivery_type: Optional[str] = None
    course_end_date: Optional[str] = None
    
    # Evaluation context
    field_name: str = ""
    response_id: Optional[int] = None
    chunk_index: Optional[int] = None
    
    # Analytics data
    sentiment_scores: Optional[Dict[str, float]] = None
    facilitator_skills: Optional[List[str]] = None
    had_guest_speakers: Optional[str] = None
    
    # Technical metadata
    model_version: str = ""
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "user_context": {
                "user_level": self.user_level,
                "agency": self.agency,
                "knowledge_level_prior": self.knowledge_level_prior
            },
            "course_context": {
                "delivery_type": self.course_delivery_type,
                "end_date": self.course_end_date
            },
            "evaluation_context": {
                "field_name": self.field_name,
                "response_id": self.response_id,
                "chunk_index": self.chunk_index
            },
            "analytics": {
                "sentiment_scores": self.sentiment_scores,
                "facilitator_skills": self.facilitator_skills,
                "had_guest_speakers": self.had_guest_speakers
            },
            "technical": {
                "model_version": self.model_version,
                "created_at": self.created_at.isoformat() if self.created_at else None
            }
        }


@dataclass
class VectorSearchResult:
    """
    Individual vector search result with similarity scoring and metadata.
    
    Represents a single text chunk that matched the search query with
    associated metadata for context and analysis.
    """
    chunk_text: str
    similarity_score: float
    metadata: SearchMetadata
    embedding_id: Optional[int] = None
    
    @property
    def relevance_category(self) -> RelevanceCategory:
        """Categorize relevance based on similarity score."""
        if self.similarity_score >= 0.80:
            return RelevanceCategory.HIGH
        elif self.similarity_score >= 0.65:
            return RelevanceCategory.MEDIUM
        elif self.similarity_score >= 0.50:
            return RelevanceCategory.LOW
        else:
            return RelevanceCategory.WEAK
    
    @property
    def user_context(self) -> str:
        """Human-readable user context summary."""
        level = self.metadata.user_level or "Unknown Level"
        agency = self.metadata.agency or "Unknown Agency"
        return f"{level} staff from {agency}"
    
    @property
    def sentiment_summary(self) -> str:
        """Human-readable sentiment summary."""
        if not self.metadata.sentiment_scores:
            return "No sentiment data"
        
        scores = self.metadata.sentiment_scores
        dominant = max(scores.items(), key=lambda x: x[1])
        return f"{dominant[0].title()} ({dominant[1]:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "chunk_text": self.chunk_text,
            "similarity_score": self.similarity_score,
            "relevance_category": self.relevance_category.value,
            "embedding_id": self.embedding_id,
            "user_context": self.user_context,
            "sentiment_summary": self.sentiment_summary,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class VectorSearchResponse:
    """
    Complete vector search response with results and performance metrics.
    
    Aggregates search results with query information and performance
    data for monitoring and optimization.
    """
    query: str
    results: List[VectorSearchResult]
    total_results: int
    processing_time: float
    
    # Search parameters
    similarity_threshold: float = 0.75
    max_results: int = 10
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    embedding_time: float = 0.0
    search_time: float = 0.0
    filtering_time: float = 0.0
    
    # Query metadata
    query_anonymized: bool = False
    privacy_controls_applied: List[str] = field(default_factory=list)
    
    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)
    
    @property
    def relevance_distribution(self) -> Dict[str, int]:
        """Distribution of results by relevance category."""
        distribution = {category.value: 0 for category in RelevanceCategory}
        for result in self.results:
            distribution[result.relevance_category.value] += 1
        return distribution
    
    @property
    def average_similarity(self) -> float:
        """Average similarity score across all results."""
        if not self.results:
            return 0.0
        return sum(result.similarity_score for result in self.results) / len(self.results)
    
    @property
    def has_high_quality_results(self) -> bool:
        """Whether response contains high-quality results."""
        return any(result.relevance_category == RelevanceCategory.HIGH for result in self.results)
    
    def get_results_by_agency(self, agency: str) -> List[VectorSearchResult]:
        """Filter results by specific agency."""
        return [
            result for result in self.results 
            if result.metadata.agency == agency
        ]
    
    def get_results_by_sentiment(self, sentiment_type: str, min_score: float = 0.5) -> List[VectorSearchResult]:
        """Filter results by sentiment type and minimum confidence."""
        filtered_results = []
        for result in self.results:
            if (result.metadata.sentiment_scores and 
                sentiment_type in result.metadata.sentiment_scores and
                result.metadata.sentiment_scores[sentiment_type] >= min_score):
                filtered_results.append(result)
        return filtered_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "query": self.query,
            "result_count": self.result_count,
            "total_results": self.total_results,
            "processing_time": self.processing_time,
            "average_similarity": self.average_similarity,
            "relevance_distribution": self.relevance_distribution,
            "has_high_quality_results": self.has_high_quality_results,
            "search_parameters": {
                "similarity_threshold": self.similarity_threshold,
                "max_results": self.max_results,
                "filters_applied": self.filters_applied
            },
            "performance_metrics": {
                "embedding_time": self.embedding_time,
                "search_time": self.search_time,
                "filtering_time": self.filtering_time
            },
            "privacy_info": {
                "query_anonymized": self.query_anonymized,
                "privacy_controls_applied": self.privacy_controls_applied
            },
            "results": [result.to_dict() for result in self.results]
        }


# Type aliases for better code readability
SearchFilters = Dict[str, Union[str, List[str], Dict[str, Any]]]
SimilarityScore = float
MetadataFilter = Dict[str, Any]
