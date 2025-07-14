"""
Vector Search Threshold Management

This module provides intelligent threshold management for vector search operations
based on query context and search requirements.
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

from ...config.settings import get_settings


class SearchIntent(Enum):
    """Different search intent types requiring different threshold strategies."""
    STRICT = "strict"          # High precision, exact matches
    BALANCED = "balanced"      # Default behavior, good precision/recall balance
    BROAD = "broad"           # High recall, exploratory search
    FUZZY = "fuzzy"           # Very relaxed, catch everything potentially relevant


@dataclass
class ThresholdStrategy:
    """Strategy for determining optimal similarity threshold."""
    similarity_threshold: float
    max_results: int
    description: str
    use_cases: list[str]


class VectorSearchThresholdManager:
    """
    Manages similarity thresholds for vector search based on query context.
    
    This class centralizes threshold logic and provides intelligent defaults
    based on search intent, content type, and system configuration.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._strategies = self._build_strategies()
    
    def _build_strategies(self) -> Dict[SearchIntent, ThresholdStrategy]:
        """Build threshold strategies based on system configuration."""
        return {
            SearchIntent.STRICT: ThresholdStrategy(
                similarity_threshold=self.settings.vector_strict_threshold,
                max_results=5,
                description="High precision for specific, well-defined queries",
                use_cases=["Exact feedback matching", "Specific issue investigation"]
            ),
            SearchIntent.BALANCED: ThresholdStrategy(
                similarity_threshold=self.settings.vector_similarity_threshold,
                max_results=self.settings.vector_max_results,
                description="Default balanced precision/recall",
                use_cases=["General feedback search", "Theme exploration"]
            ),
            SearchIntent.BROAD: ThresholdStrategy(
                similarity_threshold=self.settings.vector_relaxed_threshold,
                max_results=15,
                description="High recall for exploratory search",
                use_cases=["Trend discovery", "Pattern identification", "Brainstorming"]
            ),
            SearchIntent.FUZZY: ThresholdStrategy(
                similarity_threshold=0.15,  # Very relaxed
                max_results=20,
                description="Maximum recall, catch potential matches",
                use_cases=["Content discovery", "Correlation hunting", "Debugging"]
            )
        }
    
    def get_threshold_config(
        self, 
        intent: SearchIntent = SearchIntent.BALANCED,
        custom_threshold: Optional[float] = None,
        custom_max_results: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Get optimal threshold configuration for a search operation.
        
        Args:
            intent: Search intent determining the strategy
            custom_threshold: Override threshold if provided
            custom_max_results: Override max_results if provided
            
        Returns:
            Configuration dict with similarity_threshold and max_results
        """
        strategy = self._strategies[intent]
        
        return {
            "similarity_threshold": custom_threshold or strategy.similarity_threshold,
            "max_results": custom_max_results or strategy.max_results,
            "strategy_description": strategy.description
        }
    
    def analyze_query_intent(self, query: str) -> SearchIntent:
        """
        Analyze query to determine optimal search intent.
        
        Args:
            query: Natural language search query
            
        Returns:
            Recommended SearchIntent based on query analysis
        """
        query_lower = query.lower()
        
        # Strict intent indicators
        strict_keywords = [
            "specific", "exact", "precisely", "exactly", "only about",
            "find the comment about", "locate feedback on"
        ]
        if any(keyword in query_lower for keyword in strict_keywords):
            return SearchIntent.STRICT
        
        # Broad intent indicators  
        broad_keywords = [
            "trends", "patterns", "themes", "general", "overall", "any",
            "explore", "discover", "what do people", "how do users",
            "sentiment", "opinions", "thoughts", "experiences"
        ]
        if any(keyword in query_lower for keyword in broad_keywords):
            return SearchIntent.BROAD
        
        # Fuzzy intent indicators
        fuzzy_keywords = [
            "anything related", "similar to", "like", "around", "about",
            "might be relevant", "could relate", "correlation", "connection"
        ]
        if any(keyword in query_lower for keyword in fuzzy_keywords):
            return SearchIntent.FUZZY
        
        # Default to balanced
        return SearchIntent.BALANCED
    
    def get_smart_threshold_config(
        self, 
        query: str,
        override_intent: Optional[SearchIntent] = None,
        custom_threshold: Optional[float] = None,
        custom_max_results: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Get intelligent threshold configuration based on query analysis.
        
        Args:
            query: Natural language search query
            override_intent: Override automatic intent detection
            custom_threshold: Override threshold if provided
            custom_max_results: Override max_results if provided
            
        Returns:
            Configuration dict with optimal settings for the query
        """
        intent = override_intent or self.analyze_query_intent(query)
        config = self.get_threshold_config(intent, custom_threshold, custom_max_results)
        
        # Add analysis metadata
        config.update({
            "detected_intent": intent.value,
            "analysis_reason": self._get_intent_reasoning(query, intent)
        })
        
        return config
    
    def _get_intent_reasoning(self, query: str, intent: SearchIntent) -> str:
        """Generate reasoning for intent detection."""
        if intent == SearchIntent.STRICT:
            return "Query suggests need for precise, specific matches"
        elif intent == SearchIntent.BROAD:
            return "Query indicates exploratory or thematic search"
        elif intent == SearchIntent.FUZZY:
            return "Query suggests need for maximum recall and discovery"
        else:
            return "Using balanced approach for general search"
    
    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about current threshold configuration."""
        return {
            "system_defaults": {
                "default_threshold": self.settings.vector_similarity_threshold,
                "strict_threshold": self.settings.vector_strict_threshold,
                "relaxed_threshold": self.settings.vector_relaxed_threshold,
                "max_results": self.settings.vector_max_results
            },
            "available_strategies": {
                intent.value: {
                    "threshold": strategy.similarity_threshold,
                    "max_results": strategy.max_results,
                    "description": strategy.description,
                    "use_cases": strategy.use_cases
                }
                for intent, strategy in self._strategies.items()
            }
        }


# Convenience functions for quick access
def get_default_threshold_config() -> Dict[str, any]:
    """Get default threshold configuration from system settings."""
    manager = VectorSearchThresholdManager()
    return manager.get_threshold_config()


def get_smart_config_for_query(query: str, **kwargs) -> Dict[str, any]:
    """Get intelligent threshold configuration for a specific query."""
    manager = VectorSearchThresholdManager()
    return manager.get_smart_threshold_config(query, **kwargs)


def analyze_query_for_threshold(query: str) -> SearchIntent:
    """Analyze query and return recommended search intent."""
    manager = VectorSearchThresholdManager()
    return manager.analyze_query_intent(query)
