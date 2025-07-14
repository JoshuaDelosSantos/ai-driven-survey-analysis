#!/usr/bin/env python3
"""
Phase 3: End-to-End Feedback System Testing

Simple tests to validate core functionality of the complete feedback system
without requiring full database setup or complex configurations.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

# Mock the db module before any imports that depend on it
sys.modules['db'] = MagicMock()
sys.modules['db.db_connector'] = MagicMock()

@pytest.mark.asyncio
async def test_feedback_collection_workflow():
    """Test the complete feedback collection workflow."""
    
    from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
    
    collector = FeedbackCollector()
    
    # Test complete workflow
    feedback_data = FeedbackData(
        session_id='e2e_test_session',
        query_id='e2e_test_query',
        query_text='What are the completion rates?',
        response_text='The completion rate is 85% based on database analysis.',
        rating=4,
        comment='Very helpful response with my email john@test.com',
        response_sources=['Database Query']
    )
    
    # Test validation passes
    is_valid, error = collector.validate_feedback_data(feedback_data)
    assert is_valid, f"End-to-end validation failed: {error}"
    
    # Test PII anonymisation works
    anonymised = collector._anonymise_text(feedback_data.comment)
    assert '[EMAIL]' in anonymised, "Email not anonymised in workflow"
    assert 'john@test.com' not in anonymised, "Original email still present"

@pytest.mark.asyncio 
async def test_analytics_with_mock_data():
    """Test analytics functionality with realistic mock data."""
    
    from src.rag.core.synthesis.feedback_analytics import FeedbackAnalytics, FeedbackStats
    
    analytics = FeedbackAnalytics()
    
    # Test with realistic feedback stats
    stats = FeedbackStats(
        total_count=12,
        average_rating=4.1,
        rating_counts={1: 0, 2: 1, 3: 2, 4: 6, 5: 3},
        recent_comments=[
            'Great analysis, very accurate',
            'Could be faster but good results',
            'Excellent breakdown of the data'
        ],
        days_analyzed=30
    )
    
    formatted = analytics.format_stats_for_display(stats)
    
    # Validate key components
    assert 'Total responses: 12' in formatted, "Total count missing"
    assert '4.1/5.0' in formatted, "Average rating missing"
    assert 'Great analysis' in formatted, "Comments missing"
    assert '30 days' in formatted, "Time period missing"

def test_terminal_app_commands():
    """Test that terminal app has the correct feedback commands."""
    
    from src.rag.interfaces.terminal_app import TerminalApp
    
    app = TerminalApp(enable_agent=True)
    
    # Test that feedback methods exist and are callable
    assert hasattr(app, '_collect_feedback'), "Missing feedback collection method"
    assert hasattr(app, '_show_feedback_stats'), "Missing feedback stats method"
    assert callable(app._collect_feedback), "Feedback collection not callable"
    assert callable(app._show_feedback_stats), "Feedback stats not callable"
    
    # Test feedback system components are initialized
    assert app.feedback_collector is not None, "FeedbackCollector not initialized"
    assert app.feedback_analytics is not None, "FeedbackAnalytics not initialized"

def test_error_handling():
    """Test that feedback system handles errors gracefully."""
    
    from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
    
    collector = FeedbackCollector()
    
    # Test invalid rating
    invalid_feedback = FeedbackData(
        session_id='error_test',
        query_id='error_test',
        query_text='Test query',
        response_text='Test response',
        rating=10  # Invalid rating
    )
    
    is_valid, error = collector.validate_feedback_data(invalid_feedback)
    assert not is_valid, "Should reject invalid rating"
    assert 'Rating must be an integer between 1 and 5' in error, "Wrong error message"
    
    # Test missing required fields
    incomplete_feedback = FeedbackData(
        session_id='',  # Missing session ID
        query_id='error_test',
        query_text='Test query',
        response_text='Test response',
        rating=4
    )
    
    is_valid, error = collector.validate_feedback_data(incomplete_feedback)
    assert not is_valid, "Should reject missing session ID"
    assert 'Session ID is required' in error, "Wrong error message"

def test_pii_edge_cases():
    """Test PII anonymisation with edge cases."""
    
    from src.rag.core.synthesis.feedback_collector import FeedbackCollector
    
    collector = FeedbackCollector()
    
    # Test edge cases
    test_cases = [
        ('', ''),  # Empty string
        (None, None),  # None input
        ('No PII here', 'No PII here'),  # No PII to anonymise
        ('Multiple emails: test@example.com and user@domain.org', '[EMAIL]'),  # Multiple emails
        ('Phone: 0412345678 and email: test@example.com', '[PHONE]')  # Mixed PII
    ]
    
    for input_text, expected_pattern in test_cases:
        result = collector._anonymise_text(input_text)
        if expected_pattern:
            assert expected_pattern in result, f"Expected pattern '{expected_pattern}' not found in result for input: {input_text}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
