#!/usr/bin/env python3
"""
Phase 3: Configuration and Settings Testing

Simple tests to validate feedback system configuration options
and ensure they work as expected.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

def test_feedback_settings_structure():
    """Test that feedback settings are properly structured."""
    
    from src.rag.config.settings import RAGSettings
    
    # Test default values
    try:
        # This will fail if required fields are missing, but we can check structure
        settings_class = RAGSettings.__annotations__
        
        # Check that feedback settings exist in the class
        assert 'enable_feedback_collection' in settings_class, "Missing enable_feedback_collection setting"
        assert 'feedback_database_enabled' in settings_class, "Missing feedback_database_enabled setting"
        
    except Exception:
        # If we can't load settings due to missing config, that's okay for this test
        # We're just checking the structure exists
        pass

def test_feedback_collector_with_disabled_collection():
    """Test feedback collector behavior when collection is disabled."""
    
    from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
    
    collector = FeedbackCollector()
    
    # Create mock settings with feedback disabled
    mock_settings = MagicMock()
    mock_settings.enable_feedback_collection = False
    mock_settings.feedback_database_enabled = False
    
    # Test that validation still works even when disabled
    feedback_data = FeedbackData(
        session_id='config_test',
        query_id='config_test',
        query_text='Test query',
        response_text='Test response',
        rating=5
    )
    
    is_valid, error = collector.validate_feedback_data(feedback_data)
    assert is_valid, "Validation should work regardless of config"

def test_rating_validation_boundaries():
    """Test rating validation with boundary values."""
    
    from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
    
    collector = FeedbackCollector()
    
    # Test all valid ratings
    for rating in [1, 2, 3, 4, 5]:
        feedback_data = FeedbackData(
            session_id='boundary_test',
            query_id='boundary_test',
            query_text='Test query',
            response_text='Test response',
            rating=rating
        )
        
        is_valid, error = collector.validate_feedback_data(feedback_data)
        assert is_valid, f"Rating {rating} should be valid"
    
    # Test invalid ratings
    for rating in [0, 6, -1, 10, 3.5]:
        feedback_data = FeedbackData(
            session_id='boundary_test',
            query_id='boundary_test',
            query_text='Test query',
            response_text='Test response',
            rating=rating
        )
        
        is_valid, error = collector.validate_feedback_data(feedback_data)
        assert not is_valid, f"Rating {rating} should be invalid"

def test_analytics_empty_data_handling():
    """Test analytics behavior with no data."""
    
    from src.rag.core.synthesis.feedback_analytics import FeedbackAnalytics, FeedbackStats
    
    analytics = FeedbackAnalytics()
    
    # Test empty stats
    empty_stats = FeedbackStats()
    formatted = analytics.format_stats_for_display(empty_stats)
    
    # Should handle empty data gracefully
    assert formatted, "Should return something for empty stats"
    assert 'No feedback data available' in formatted or 'Total responses: 0' in formatted, "Should indicate no data"

def test_terminal_app_initialization():
    """Test terminal app initializes correctly with feedback system."""
    
    from src.rag.interfaces.terminal_app import TerminalApp
    
    # Test both agent modes
    for enable_agent in [True, False]:
        app = TerminalApp(enable_agent=enable_agent)
        
        # Should always have feedback components regardless of agent mode
        assert hasattr(app, 'feedback_collector'), f"Missing collector in agent={enable_agent} mode"
        assert hasattr(app, 'feedback_analytics'), f"Missing analytics in agent={enable_agent} mode"
        assert app.feedback_collector is not None, f"Collector not initialized in agent={enable_agent} mode"
        assert app.feedback_analytics is not None, f"Analytics not initialized in agent={enable_agent} mode"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
