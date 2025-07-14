#!/usr/bin/env python3
"""
Test Enhanced Feedback Integration in Terminal App

Simple test to verify the enhanced feedback system integration
without requiring full database setup or LLM configuration.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

# Mock the db module before any imports that depend on it
sys.modules['db'] = MagicMock()
sys.modules['db.db_connector'] = MagicMock()

@pytest.mark.asyncio
async def test_terminal_app_integration():
    """Test the enhanced terminal app with feedback integration."""
    
    try:
        # Mock database operations for feedback collector
        with patch('src.rag.core.synthesis.feedback_collector.db_connector') as mock_db:
            mock_db.get_db_connection.return_value = MagicMock()
            mock_db.execute_query.return_value = None
            mock_db.close_db_connection.return_value = None
            
            # Test imports
            from src.rag.interfaces.terminal_app import TerminalApp
            
            # Test instantiation with feedback components
            app = TerminalApp(enable_agent=True)
            
            # Test feedback collector initialization
            assert hasattr(app, 'feedback_collector'), "FeedbackCollector missing"
            assert app.feedback_collector is not None, "FeedbackCollector not initialized"
            
            # Test feedback analytics initialization
            assert hasattr(app, 'feedback_analytics'), "FeedbackAnalytics missing"
            assert app.feedback_analytics is not None, "FeedbackAnalytics not initialized"
            
            # Test feedback collection method exists
            assert hasattr(app, '_collect_feedback'), "Feedback collection method missing"
            assert callable(getattr(app, '_collect_feedback')), "Feedback collection method not callable"
            
            # Test feedback stats method exists
            assert hasattr(app, '_show_feedback_stats'), "Feedback statistics method missing"
            assert callable(getattr(app, '_show_feedback_stats')), "Feedback statistics method not callable"
            
            # Test session initialization
            assert app.session_id, "Session ID not generated"
            assert len(app.session_id) == 8, "Session ID incorrect length"
            
            # Test initial state
            assert app.query_count == 0, "Initial query count should be 0"
            assert app.feedback_collected == {}, "Initial feedback collection should be empty"
        
    except Exception as e:
        pytest.fail(f"Terminal app integration test failed: {e}")

@pytest.mark.asyncio
async def test_feedback_collector_integration():
    """Test feedback collector component integration."""
    
    try:
        # Mock database operations
        with patch('src.rag.core.synthesis.feedback_collector.db_connector') as mock_db:
            mock_db.get_db_connection.return_value = MagicMock()
            mock_db.execute_query.return_value = None
            mock_db.close_db_connection.return_value = None
            
            from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
            
            # Test instantiation
            collector = FeedbackCollector()
            assert collector is not None, "FeedbackCollector not instantiated"
            
            # Test validation method
            feedback_data = FeedbackData(
                session_id='test_session_001',
                query_id='test_query_001',
                query_text='Test query',
                response_text='Test response',
                rating=4
            )
            
            is_valid, error = collector.validate_feedback_data(feedback_data)
            assert is_valid, f"Validation failed: {error}"
            assert error == "", "Should have no validation errors"
            
            # Test PII anonymisation
            original = 'Contact john.smith@example.com or call 0412345678 for more info'
            anonymised = collector._anonymise_text(original)
            assert '[EMAIL]' in anonymised, "Email not anonymised"
            assert '[PHONE]' in anonymised or '[MOBILE]' in anonymised, "Phone not anonymised"
            
    except Exception as e:
        pytest.fail(f"Feedback collector integration test failed: {e}")

@pytest.mark.asyncio
async def test_feedback_analytics_integration():
    """Test feedback analytics component integration."""
    
    try:
        # Mock database operations
        with patch('src.rag.core.synthesis.feedback_analytics.db_connector') as mock_db:
            mock_db.get_db_connection.return_value = MagicMock()
            mock_db.execute_query.return_value = []
            mock_db.close_db_connection.return_value = None
            
            from src.rag.core.synthesis.feedback_analytics import FeedbackAnalytics, FeedbackStats
            
            # Test instantiation
            analytics = FeedbackAnalytics()
            assert analytics is not None, "FeedbackAnalytics not instantiated"
            
            # Test stats formatting with sample data
            sample_stats = FeedbackStats(
                total_count=25,
                average_rating=4.2,
                rating_counts={1: 1, 2: 2, 3: 5, 4: 12, 5: 5},
                recent_comments=['Great system', 'Very helpful', 'Could be faster'],
                days_analyzed=7
            )
            
            formatted = analytics.format_stats_for_display(sample_stats)
            assert formatted, "Formatted stats should not be empty"
            assert 'Total responses: 25' in formatted, "Total count not in formatted output"
            assert '4.2/5.0' in formatted, "Average rating not in formatted output"
            assert 'Great system' in formatted, "Comments not in formatted output"
            
    except Exception as e:
        pytest.fail(f"Feedback analytics integration test failed: {e}")

def test_imports():
    """Test that all feedback modules can be imported successfully."""
    
    try:
        from src.rag.interfaces.terminal_app import TerminalApp
        from src.rag.core.synthesis.feedback_collector import FeedbackCollector, FeedbackData
        from src.rag.core.synthesis.feedback_analytics import FeedbackAnalytics, FeedbackStats
        
        # All imports successful
        assert True
        
    except ImportError as e:
        pytest.fail(f"Import test failed: {e}")

def test_pii_anonymisation():
    """Test PII anonymisation functionality."""
    
    # Mock database operations
    with patch('src.rag.core.synthesis.feedback_collector.db_connector'):
        from src.rag.core.synthesis.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        # Test cases for different PII patterns
        test_cases = [
            {
                'input': 'Contact john.smith@example.com for more info',
                'should_contain': '[EMAIL]',
                'should_not_contain': 'john.smith@example.com'
            },
            {
                'input': 'Call me at 0412345678 or 0398765432',
                'should_contain': '[PHONE]',
                'should_not_contain': '0412345678'
            },
            {
                'input': 'My name is John Smith and I work here',
                'should_contain': '[NAME]',
                'should_not_contain': 'John Smith'
            }
        ]
        
        for case in test_cases:
            anonymised = collector._anonymise_text(case['input'])
            assert case['should_contain'] in anonymised, f"Expected {case['should_contain']} in anonymised text"
            assert case['should_not_contain'] not in anonymised, f"Original PII still present in anonymised text"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
