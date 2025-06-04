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
