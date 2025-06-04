"""
Test suite for DBOperations class.

Tests cover:
- Database operations initialization
- Sentiment score writing with various inputs
- SQL query generation and parameter binding
- Upsert functionality with conflict resolution
- Error handling for database failures
- Return value validation
"""
import pytest
import re
from unittest.mock import Mock, patch, MagicMock

# Import the class under test
import sys
import os
# Add both the sentiment-analysis directory and the parent src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sentiment_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(sentiment_dir)
sys.path.append(sentiment_dir)
sys.path.append(src_dir)

from db_operations import DBOperations


class TestDBOperationsInitialization:
    """Test DBOperations initialization."""
    
    def test_init_creates_instance(self):
        """Test that DBOperations can be instantiated."""
        # Act
        db_ops = DBOperations()
        
        # Assert
        assert isinstance(db_ops, DBOperations)
    
    def test_init_no_persistent_connection(self):
        """Test that initialization doesn't create persistent connections."""
        # Act
        db_ops = DBOperations()
        
        # Assert - should not have connection attributes
        assert not hasattr(db_ops, 'connection')
        assert not hasattr(db_ops, 'cursor')


class TestDBOperationsWriteSentiment:
    """Test the write_sentiment method with various inputs."""
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_normal_scores(self, mock_execute_query):
        """Test writing normal sentiment scores."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 42
        column = 'general_feedback'
        scores = {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        mock_execute_query.assert_called_once()
        
        # Check the query structure
        call_args = mock_execute_query.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        
        assert 'INSERT INTO' in query
        assert 'evaluation_sentiment' in query
        assert 'ON CONFLICT' in query
        assert 'DO UPDATE' in query and 'SET' in query
        
        # Check parameters
        assert params[0] == response_id
        assert params[1] == column
        assert list(params[2:]) == [0.1, 0.2, 0.7]
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_edge_scores(self, mock_execute_query):
        """Test writing edge case sentiment scores (0.0 and 1.0)."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 1
        column = 'course_feedback'
        scores = {'neg': 0.0, 'neu': 0.0, 'pos': 1.0}
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        assert list(params[2:]) == [0.0, 0.0, 1.0]
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_missing_score_key(self, mock_execute_query):
        """Test writing scores with missing keys (should use None)."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 123
        column = 'feedback'
        scores = {'neg': 0.3, 'neu': 0.4}  # missing 'pos'
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        assert list(params[2:]) == [0.3, 0.4, None]  # None for missing 'pos'
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_extra_score_keys(self, mock_execute_query):
        """Test writing scores with extra keys (should be ignored)."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 456
        column = 'comments'
        scores = {'neg': 0.2, 'neu': 0.3, 'pos': 0.5, 'extra': 0.1}
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        assert list(params[2:]) == [0.2, 0.3, 0.5]  # 'extra' ignored
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_large_response_id(self, mock_execute_query):
        """Test writing with large response ID."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 999999
        column = 'feedback'
        scores = {'neg': 0.1, 'neu': 0.8, 'pos': 0.1}
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        assert params[0] == response_id
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_long_column_name(self, mock_execute_query):
        """Test writing with long column name."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 789
        column = 'very_long_column_name_for_detailed_feedback_responses'
        scores = {'neg': 0.5, 'neu': 0.3, 'pos': 0.2}
        
        # Act
        result = db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        assert result == 1
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        assert params[1] == column


class TestDBOperationsQueryGeneration:
    """Test SQL query generation and structure."""
    
    @patch('db_operations.execute_query')
    def test_query_structure_validation(self, mock_execute_query):
        """Test that generated SQL query has correct structure."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        
        # Act
        db_ops.write_sentiment(1, 'test_column', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        
        # Assert
        call_args = mock_execute_query.call_args
        query = call_args[0][0]
        
        # Check query structure components individually (more reliable than complex regex)
        assert 'INSERT INTO' in query
        assert 'VALUES' in query
        assert 'ON CONFLICT' in query
        assert 'DO UPDATE' in query
        assert 'SET' in query
        
        # Check specific components
        assert 'evaluation_sentiment' in query
        assert 'response_id' in query
        assert 'column_name' in query
        assert 'neg' in query
        assert 'neu' in query
        assert 'pos' in query
        assert 'EXCLUDED' in query
    
    @patch('db_operations.execute_query')
    def test_parameter_binding_validation(self, mock_execute_query):
        """Test that parameters are correctly bound to query."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 42
        column = 'test_col'
        scores = {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}
        
        # Act
        db_ops.write_sentiment(response_id, column, scores)
        
        # Assert
        call_args = mock_execute_query.call_args
        params = call_args[0][1]
        
        assert isinstance(params, tuple)
        assert len(params) == 5  # response_id, column, neg, neu, pos
        assert params[0] == response_id
        assert params[1] == column
        assert params[2] == scores['neg']
        assert params[3] == scores['neu']
        assert params[4] == scores['pos']
    
    @patch('db_operations.execute_query')
    def test_query_uses_config_values(self, mock_execute_query):
        """Test that query uses values from config module."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        
        # Act
        db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        
        # Assert
        call_args = mock_execute_query.call_args
        query = call_args[0][0]
        
        # Should use SENTIMENT_TABLE from config
        from config import SENTIMENT_TABLE, SCORE_COLUMNS
        assert SENTIMENT_TABLE in query
        
        # Should use SCORE_COLUMNS from config
        for col in SCORE_COLUMNS:
            assert col in query


class TestDBOperationsReturnValues:
    """Test return value handling and validation."""
    
    @patch('db_operations.execute_query')
    def test_returns_execute_query_result(self, mock_execute_query):
        """Test that write_sentiment returns execute_query result."""
        # Arrange
        mock_execute_query.return_value = 5
        db_ops = DBOperations()
        
        # Act
        result = db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        
        # Assert
        assert result == 5
    
    @patch('db_operations.execute_query')
    def test_returns_zero_when_no_rows_affected(self, mock_execute_query):
        """Test return value when no rows are affected."""
        # Arrange
        mock_execute_query.return_value = 0
        db_ops = DBOperations()
        
        # Act
        result = db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        
        # Assert
        assert result == 0
    
    @patch('db_operations.execute_query')
    def test_returns_multiple_rows_affected(self, mock_execute_query):
        """Test return value when multiple rows are affected."""
        # Arrange
        mock_execute_query.return_value = 2
        db_ops = DBOperations()
        
        # Act
        result = db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        
        # Assert
        assert result == 2


class TestDBOperationsErrorHandling:
    """Test error handling for various failure scenarios."""
    
    @patch('db_operations.execute_query')
    def test_handles_database_connection_error(self, mock_execute_query):
        """Test handling of database connection errors."""
        # Arrange
        mock_execute_query.side_effect = Exception("Database connection failed")
        db_ops = DBOperations()
        
        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
    
    @patch('db_operations.execute_query')
    def test_handles_sql_syntax_error(self, mock_execute_query):
        """Test handling of SQL syntax errors."""
        # Arrange
        mock_execute_query.side_effect = Exception("SQL syntax error")
        db_ops = DBOperations()
        
        # Act & Assert
        with pytest.raises(Exception, match="SQL syntax error"):
            db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
    
    @patch('db_operations.execute_query')
    def test_handles_constraint_violation(self, mock_execute_query):
        """Test handling of database constraint violations."""
        # Arrange
        mock_execute_query.side_effect = Exception("Foreign key constraint violation")
        db_ops = DBOperations()
        
        # Act & Assert
        with pytest.raises(Exception, match="Foreign key constraint violation"):
            db_ops.write_sentiment(1, 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
    
    def test_handles_invalid_response_id_type(self):
        """Test handling of invalid response_id types."""
        # Arrange
        db_ops = DBOperations()
        
        # Act & Assert - Should let the database handle type validation
        # The method itself doesn't validate types, relying on DB constraints
        try:
            with patch('db_operations.execute_query') as mock_execute:
                mock_execute.side_effect = Exception("Invalid data type")
                db_ops.write_sentiment("invalid_id", 'col', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
        except Exception as e:
            assert "Invalid data type" in str(e)
    
    def test_handles_none_column_name(self):
        """Test handling of None column name."""
        # Arrange
        db_ops = DBOperations()
        
        # Act & Assert
        with patch('db_operations.execute_query') as mock_execute:
            mock_execute.side_effect = Exception("NULL value in column")
            with pytest.raises(Exception, match="NULL value in column"):
                db_ops.write_sentiment(1, None, {'neg': 0.1, 'neu': 0.2, 'pos': 0.7})
    
    def test_handles_empty_scores_dict(self):
        """Test handling of empty scores dictionary."""
        # Arrange
        db_ops = DBOperations()
        
        # Act
        with patch('db_operations.execute_query') as mock_execute:
            mock_execute.return_value = 1
            result = db_ops.write_sentiment(1, 'col', {})
        
        # Assert - should pass None values for all scores
        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert list(params[2:]) == [None, None, None]
        assert result == 1


class TestDBOperationsIntegration:
    """Integration tests for database operations."""
    
    @patch('db_operations.execute_query')
    def test_multiple_writes_same_response_column(self, mock_execute_query):
        """Test multiple writes to same response_id and column (upsert behavior)."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        response_id = 1
        column = 'feedback'
        
        # Act - write twice with different scores
        scores1 = {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}
        scores2 = {'neg': 0.3, 'neu': 0.4, 'pos': 0.3}
        
        result1 = db_ops.write_sentiment(response_id, column, scores1)
        result2 = db_ops.write_sentiment(response_id, column, scores2)
        
        # Assert
        assert result1 == 1
        assert result2 == 1
        assert mock_execute_query.call_count == 2
        
        # Check that both calls used upsert query
        for call in mock_execute_query.call_args_list:
            query = call[0][0]
            assert 'ON CONFLICT' in query
            assert 'DO UPDATE' in query and 'SET' in query
    
    @patch('db_operations.execute_query')
    def test_write_sentiment_realistic_workflow(self, mock_execute_query):
        """Test realistic workflow with multiple sentiment writes."""
        # Arrange
        mock_execute_query.return_value = 1
        db_ops = DBOperations()
        
        # Simulate processing multiple evaluation responses
        test_data = [
            (1, 'general_feedback', {'neg': 0.1, 'neu': 0.2, 'pos': 0.7}),
            (1, 'course_application_other', {'neg': 0.2, 'neu': 0.3, 'pos': 0.5}),
            (2, 'general_feedback', {'neg': 0.8, 'neu': 0.1, 'pos': 0.1}),
            (3, 'did_experience_issue_detail', {'neg': 0.0, 'neu': 1.0, 'pos': 0.0}),
        ]
        
        # Act
        results = []
        for response_id, column, scores in test_data:
            result = db_ops.write_sentiment(response_id, column, scores)
            results.append(result)
        
        # Assert
        assert all(result == 1 for result in results)
        assert mock_execute_query.call_count == len(test_data)
        
        # Verify each call had correct parameters
        for i, (response_id, column, scores) in enumerate(test_data):
            call_args = mock_execute_query.call_args_list[i]
            params = call_args[0][1]
            assert params[0] == response_id
            assert params[1] == column
            assert params[2] == scores.get('neg')
            assert params[3] == scores.get('neu')
            assert params[4] == scores.get('pos')
