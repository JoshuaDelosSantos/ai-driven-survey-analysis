"""
Shared fixtures and configuration for sentiment analysis tests.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch


@pytest.fixture
def mock_tokenizer():
    """Mock AutoTokenizer for testing without loading actual model."""
    tokenizer = Mock()
    # Mock tokenizer output - typical structure for transformers
    tokenizer.return_value = {
        'input_ids': torch.tensor([[101, 2023, 2003, 102]]),  # Example token IDs
        'attention_mask': torch.tensor([[1, 1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock AutoModelForSequenceClassification for testing."""
    model = Mock()
    model.eval = Mock()
    
    # Mock model output with logits
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.1, 0.3, 0.6]])  # [neg, neu, pos] example logits
    model.return_value = mock_output
    
    return model


@pytest.fixture
def mock_sentiment_analyser():
    """Create a SentimentAnalyser with mocked dependencies."""
    with patch('analyser.AutoTokenizer.from_pretrained') as mock_tokenizer_cls:
        with patch('analyser.AutoModelForSequenceClassification.from_pretrained') as mock_model_cls:
            mock_tokenizer_cls.return_value = Mock()
            mock_model_cls.return_value = Mock()
            mock_model_cls.return_value.eval = Mock()
            
            from analyser import SentimentAnalyser
            return SentimentAnalyser()


@pytest.fixture
def sample_texts():
    """Sample texts for testing different scenarios."""
    return {
        'positive': "This course was excellent and very informative!",
        'negative': "Terrible experience, waste of time.",
        'neutral': "The content was okay but could be improved.",
        'empty': "",
        'whitespace': "   \n\t   ",
        'long': "A" * 1000,  # Very long text (>512 tokens)
        'special_chars': "Mixed feelings: good content 😊 but poor delivery 😞",
        'mixed_language': "Great course! Très bien! 素晴らしい!",
        'numbers_only': "123 456 789",
        'symbols_only': "!@#$%^&*()",
    }


@pytest.fixture
def expected_score_structure():
    """Expected structure for sentiment scores."""
    return ['neg', 'neu', 'pos']


@pytest.fixture
def mock_config():
    """Mock configuration values."""
    return {
        'MODEL_NAME': 'cardiffnlp/twitter-roberta-base-sentiment',
        'SCORE_COLUMNS': ['neg', 'neu', 'pos']
    }


@pytest.fixture
def mock_execute_query():
    """Mock execute_query function for database operations testing."""
    return Mock(return_value=1)  # Default: 1 row affected


@pytest.fixture
def sample_sentiment_scores():
    """Sample sentiment scores for testing database operations."""
    return {
        'valid_scores': {'neg': 0.1, 'neu': 0.2, 'pos': 0.7},
        'edge_scores': {'neg': 0.0, 'neu': 0.0, 'pos': 1.0},
        'missing_key': {'neg': 0.3, 'neu': 0.4},  # missing 'pos'
        'extra_key': {'neg': 0.2, 'neu': 0.3, 'pos': 0.5, 'extra': 0.1},
        'none_values': {'neg': None, 'neu': 0.5, 'pos': 0.5}
    }


@pytest.fixture
def sample_db_data():
    """Sample database data for testing."""
    return {
        'response_ids': [1, 42, 999, 123456],
        'columns': ['general_feedback', 'course_application_other', 'did_experience_issue_detail'],
        'invalid_columns': ['', 'invalid_col', None],
        'invalid_response_ids': [0, -1, None, 'string_id']
    }


@pytest.fixture
def expected_upsert_query():
    """Expected SQL query structure for upsert operations."""
    return {
        'base_pattern': r'INSERT INTO .+ \(.+\) VALUES \(.+\) ON CONFLICT \(.+\) DO UPDATE SET .+',
        'table_name': 'evaluation_sentiment',
        'columns': ['response_id', 'column_name', 'neg', 'neu', 'pos'],
        'conflict_columns': ['response_id', 'column_name']
    }
