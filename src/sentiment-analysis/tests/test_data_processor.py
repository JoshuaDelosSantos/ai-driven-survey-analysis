"""
Tests for DataProcessor module.

Tests initialization, data processing workflows, error handling,
and integration between components.
"""
import sys
import os
# Add both sentiment-analysis directory and parent src directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
sentiment_dir = os.path.dirname(test_dir)
src_dir = os.path.dirname(sentiment_dir)
sys.path.insert(0, sentiment_dir)
sys.path.insert(0, src_dir)

import pytest
from unittest.mock import Mock, patch, call
from data_processor import DataProcessor


class TestDataProcessorInitialization:
    """Test DataProcessor initialization and dependency injection."""
    
    def test_init_with_valid_dependencies(self):
        """Test DataProcessor initializes correctly with valid dependencies."""
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        
        assert processor.db_ops is mock_db_ops
        assert processor.analyser is mock_analyser
    
    def test_init_with_none_dependencies(self):
        """Test DataProcessor handles None dependencies."""
        processor = DataProcessor(None, None)
        
        assert processor.db_ops is None
        assert processor.analyser is None
    
    def test_init_dependency_types(self):
        """Test DataProcessor accepts different dependency types."""
        # Test with mock objects
        mock_db = Mock()
        mock_analyser = Mock()
        processor = DataProcessor(mock_db, mock_analyser)
        
        assert hasattr(processor, 'db_ops')
        assert hasattr(processor, 'analyser')


class TestDataProcessorProcessAll:
    """Test DataProcessor.process_all() method functionality."""
    
    @patch('data_processor.fetch_data')
    def test_process_all_normal_data(self, mock_fetch, sample_evaluation_data):
        """Test processing normal evaluation data."""
        mock_fetch.return_value = sample_evaluation_data['normal_data']
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}
        mock_db_ops.write_sentiment.return_value = 1
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify fetch_data called with correct query
        expected_query = "SELECT response_id, did_experience_issue_detail, course_application_other, general_feedback FROM evaluation"
        mock_fetch.assert_called_once_with(expected_query)
        
        # Verify analyser.analyse called for each non-empty text
        assert mock_analyser.analyse.call_count == 9  # 3 rows × 3 text columns
        
        # Verify write_sentiment called for each analysis
        assert mock_db_ops.write_sentiment.call_count == 9
    
    @patch('data_processor.fetch_data')
    def test_process_all_mixed_data(self, mock_fetch, sample_evaluation_data):
        """Test processing data with empty strings, None values, and whitespace."""
        mock_fetch.return_value = sample_evaluation_data['mixed_data']
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.3, 'neu': 0.4, 'pos': 0.3}
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Should only process non-empty, non-whitespace text
        # Row 1: 'Good course', 'Recommended' (2 texts)
        # Row 2: 'No issues' (1 text)
        # Row 3: 'Great!' (1 text)
        expected_calls = 4
        assert mock_analyser.analyse.call_count == expected_calls
        assert mock_db_ops.write_sentiment.call_count == expected_calls
    
    @patch('data_processor.fetch_data')
    def test_process_all_empty_dataset(self, mock_fetch):
        """Test processing empty dataset."""
        mock_fetch.return_value = []
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # No analysis or database writes should occur
        mock_analyser.analyse.assert_not_called()
        mock_db_ops.write_sentiment.assert_not_called()
    
    @patch('data_processor.fetch_data')
    def test_process_all_single_row(self, mock_fetch, sample_evaluation_data):
        """Test processing single evaluation row."""
        mock_fetch.return_value = sample_evaluation_data['single_row']
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.0, 'neu': 0.1, 'pos': 0.9}
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Should process all 3 text columns for single row
        assert mock_analyser.analyse.call_count == 3
        assert mock_db_ops.write_sentiment.call_count == 3
        
        # Verify correct response_id used
        calls = mock_db_ops.write_sentiment.call_args_list
        for call_args in calls:
            assert call_args[0][0] == 42  # response_id from single_row


class TestDataProcessorQueryGeneration:
    """Test SQL query generation and data fetching."""
    
    @patch('data_processor.fetch_data')
    def test_query_structure(self, mock_fetch, expected_sql_queries):
        """Test that correct SQL query is generated."""
        mock_fetch.return_value = []
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify exact query structure
        expected_query = expected_sql_queries['base_select']
        mock_fetch.assert_called_once_with(expected_query)
    
    @patch('data_processor.fetch_data')
    def test_column_order_consistency(self, mock_fetch):
        """Test that columns are selected in correct order."""
        mock_fetch.return_value = [(1, 'text1', 'text2', 'text3')]
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.2, 'neu': 0.3, 'pos': 0.5}
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify write_sentiment called with correct column names
        calls = mock_db_ops.write_sentiment.call_args_list
        column_names = [call[0][1] for call in calls]
        
        expected_columns = ['did_experience_issue_detail', 'course_application_other', 'general_feedback']
        assert all(col in expected_columns for col in column_names)


class TestDataProcessorErrorHandling:
    """Test error handling during data processing."""
    
    @patch('data_processor.fetch_data')
    def test_analyser_error_handling(self, mock_fetch, capsys):
        """Test handling of analyser errors."""
        mock_fetch.return_value = [(1, 'Test text', 'Another text', 'Third text')]
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.side_effect = [
            {'neg': 0.1, 'neu': 0.2, 'pos': 0.7},  # Success
            ValueError("Invalid input"),  # Error
            {'neg': 0.3, 'neu': 0.3, 'pos': 0.4}   # Success
        ]
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Should continue processing despite error
        assert mock_analyser.analyse.call_count == 3
        assert mock_db_ops.write_sentiment.call_count == 2  # Only successful analyses
        
        # Check error message printed
        captured = capsys.readouterr()
        assert "Error processing response 1" in captured.out
        assert "course_application_other" in captured.out
    
    @patch('data_processor.fetch_data')
    def test_database_write_error_handling(self, mock_fetch, capsys):
        """Test handling of database write errors."""
        mock_fetch.return_value = [(2, 'Test feedback', 'Issue details', 'General comment')]
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.2, 'neu': 0.3, 'pos': 0.5}
        mock_db_ops.write_sentiment.side_effect = [
            1,  # Success
            Exception("Database connection failed"),  # Error
            1   # Success
        ]
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Should continue processing despite database error
        assert mock_analyser.analyse.call_count == 3
        assert mock_db_ops.write_sentiment.call_count == 3
        
        # Check error message printed
        captured = capsys.readouterr()
        assert "Error processing response 2" in captured.out
        assert "course_application_other" in captured.out
    
    @patch('data_processor.fetch_data')
    def test_fetch_data_error_propagation(self, mock_fetch):
        """Test that fetch_data errors are properly propagated."""
        mock_fetch.side_effect = Exception("Database connection failed")
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        
        with pytest.raises(Exception, match="Database connection failed"):
            processor.process_all()
    
    @patch('data_processor.fetch_data')
    def test_multiple_errors_in_row(self, mock_fetch, capsys):
        """Test handling multiple errors in same row."""
        mock_fetch.return_value = [(3, 'Text 1', 'Text 2', 'Text 3')]
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.side_effect = [
            ValueError("Error 1"),
            RuntimeError("Error 2"), 
            Exception("Error 3")
        ]
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # All three should fail
        assert mock_analyser.analyse.call_count == 3
        mock_db_ops.write_sentiment.assert_not_called()
        
        # Should print error for each failed analysis
        captured = capsys.readouterr()
        assert captured.out.count("Error processing response 3") == 3


class TestDataProcessorIntegration:
    """Test integration scenarios and workflows."""
    
    @patch('data_processor.fetch_data')
    def test_end_to_end_workflow(self, mock_fetch):
        """Test complete end-to-end processing workflow."""
        # Setup test data
        test_data = [
            (1, 'Excellent course!', 'No issues encountered', 'Highly recommended'),
            (2, 'Poor quality content', 'System crashed frequently', 'Not worth the time'),
        ]
        mock_fetch.return_value = test_data
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        # Mock different sentiment scores for different texts
        sentiment_responses = [
            {'neg': 0.1, 'neu': 0.2, 'pos': 0.7},  # Positive
            {'neg': 0.0, 'neu': 0.1, 'pos': 0.9},  # Very positive
            {'neg': 0.0, 'neu': 0.2, 'pos': 0.8},  # Positive
            {'neg': 0.8, 'neu': 0.1, 'pos': 0.1},  # Negative
            {'neg': 0.9, 'neu': 0.1, 'pos': 0.0},  # Very negative
            {'neg': 0.7, 'neu': 0.2, 'pos': 0.1},  # Negative
        ]
        mock_analyser.analyse.side_effect = sentiment_responses
        mock_db_ops.write_sentiment.return_value = 1
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify complete workflow
        assert mock_analyser.analyse.call_count == 6  # 2 rows × 3 columns
        assert mock_db_ops.write_sentiment.call_count == 6
        
        # Verify correct data passed to write_sentiment
        write_calls = mock_db_ops.write_sentiment.call_args_list
        response_ids = [call[0][0] for call in write_calls]
        assert response_ids.count(1) == 3  # First row
        assert response_ids.count(2) == 3  # Second row
    
    @patch('data_processor.fetch_data')
    def test_large_dataset_processing(self, mock_fetch, sample_evaluation_data):
        """Test processing large dataset efficiently."""
        mock_fetch.return_value = sample_evaluation_data['large_dataset']
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        mock_analyser.analyse.return_value = {'neg': 0.33, 'neu': 0.33, 'pos': 0.34}
        mock_db_ops.write_sentiment.return_value = 1
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Should process all 100 rows × 3 columns = 300 analyses
        assert mock_analyser.analyse.call_count == 300
        assert mock_db_ops.write_sentiment.call_count == 300
    
    @patch('data_processor.fetch_data')
    def test_configuration_integration(self, mock_fetch):
        """Test that DataProcessor uses correct configuration values."""
        mock_fetch.return_value = []
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify that the query uses the correct table and columns from config
        call_args = mock_fetch.call_args[0][0]
        assert "evaluation" in call_args  # EVALUATION_TABLE
        assert "did_experience_issue_detail" in call_args  # From FREE_TEXT_COLUMNS
        assert "course_application_other" in call_args     # From FREE_TEXT_COLUMNS
        assert "general_feedback" in call_args            # From FREE_TEXT_COLUMNS
    
    @patch('data_processor.fetch_data')
    def test_dependency_interaction(self, mock_fetch):
        """Test interaction between DataProcessor dependencies."""
        mock_fetch.return_value = [(1, 'Test text', 'Another text', 'Third text')]
        
        mock_db_ops = Mock()
        mock_analyser = Mock()
        
        # Setup analyser to return specific scores
        test_scores = {'neg': 0.25, 'neu': 0.25, 'pos': 0.5}
        mock_analyser.analyse.return_value = test_scores
        mock_db_ops.write_sentiment.return_value = 1
        
        processor = DataProcessor(mock_db_ops, mock_analyser)
        processor.process_all()
        
        # Verify analyser called with correct texts
        analyse_calls = mock_analyser.analyse.call_args_list
        texts_analysed = [call[0][0] for call in analyse_calls]
        assert 'Test text' in texts_analysed
        assert 'Another text' in texts_analysed
        assert 'Third text' in texts_analysed
        
        # Verify write_sentiment called with analyser results
        write_calls = mock_db_ops.write_sentiment.call_args_list
        for call in write_calls:
            assert call[0][2] == test_scores  # scores parameter
