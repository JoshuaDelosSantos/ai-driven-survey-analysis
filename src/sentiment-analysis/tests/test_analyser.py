"""
Test suite for SentimentAnalyser class.

Tests cover:
- Model and tokenizer initialization
- Text analysis with various inputs
- Output format validation
- Error handling for invalid inputs
- Performance characteristics
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from scipy.special import softmax

# Import the class under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyser import SentimentAnalyser


class TestSentimentAnalyserInitialization:
    """Test SentimentAnalyser initialization and setup."""
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_init_loads_model_and_tokenizer(self, mock_model_cls, mock_tokenizer_cls):
        """Test that initialization properly loads model and tokenizer."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        # Act
        analyser = SentimentAnalyser()
        
        # Assert
        mock_tokenizer_cls.assert_called_once()
        mock_model_cls.assert_called_once()
        mock_model.eval.assert_called_once()
        assert analyser.tokenizer == mock_tokenizer
        assert analyser.model == mock_model
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_init_uses_correct_model_name(self, mock_model_cls, mock_tokenizer_cls):
        """Test that initialization uses the configured model name."""
        # Arrange
        mock_tokenizer_cls.return_value = Mock()
        mock_model_cls.return_value = Mock()
        mock_model_cls.return_value.eval = Mock()
        
        # Act
        SentimentAnalyser()
        
        # Assert
        from config import MODEL_NAME
        mock_tokenizer_cls.assert_called_with(MODEL_NAME)
        mock_model_cls.assert_called_with(MODEL_NAME)
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_init_handles_model_loading_error(self, mock_model_cls, mock_tokenizer_cls):
        """Test that initialization handles model loading errors gracefully."""
        # Arrange
        mock_tokenizer_cls.side_effect = Exception("Failed to load tokenizer")
        
        # Act & Assert
        with pytest.raises(Exception, match="Failed to load tokenizer"):
            SentimentAnalyser()


class TestSentimentAnalyserAnalyse:
    """Test the analyse method with various inputs."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        self.mock_model.eval = Mock()
        
        # Mock tokenizer output
        self.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        # Mock model output with realistic logits
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, -0.5, 1.2]])  # [neg, neu, pos]
        self.mock_model.return_value = mock_output
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_normal_text(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of normal positive text."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = "This course was excellent and very informative!"
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert 'neg' in result
        assert 'neu' in result
        assert 'pos' in result
        
        # Check that tokenizer was called with the text
        self.mock_tokenizer.assert_called_with(text, return_tensors='pt')
        
        # Check that model was called with tokenized input
        self.mock_model.assert_called_once()
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_empty_string(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of empty string."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = ""
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        self.mock_tokenizer.assert_called_with(text, return_tensors='pt')
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_whitespace_only(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of whitespace-only string."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = "   \n\t   "
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        self.mock_tokenizer.assert_called_with(text, return_tensors='pt')
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_very_long_text(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of very long text (>512 tokens)."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = "A" * 1000  # Very long text
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        self.mock_tokenizer.assert_called_with(text, return_tensors='pt')
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_special_characters(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of text with special characters and emojis."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = "Mixed feelings: good content 😊 but poor delivery 😞"
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3
        self.mock_tokenizer.assert_called_with(text, return_tensors='pt')
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_mixed_language(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis of mixed language text."""
        # Arrange
        mock_tokenizer_cls.return_value = self.mock_tokenizer
        mock_model_cls.return_value = self.mock_model
        
        analyser = SentimentAnalyser()
        text = "Great course! Très bien! 素晴らしい!"
        
        # Act
        result = analyser.analyse(text)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 3


class TestSentimentAnalyserOutputValidation:
    """Test output format and value validation."""
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_output_structure_validation(self, mock_model_cls, mock_tokenizer_cls):
        """Test that output has correct structure."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, -0.5, 1.2]])
        mock_model.return_value = mock_output
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act
        result = analyser.analyse("Test text")
        
        # Assert
        assert isinstance(result, dict)
        assert set(result.keys()) == {'neg', 'neu', 'pos'}
        
        # Check that all values are floats
        for key, value in result.items():
            assert isinstance(value, float)
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_score_value_ranges(self, mock_model_cls, mock_tokenizer_cls):
        """Test that sentiment scores are in valid range [0, 1]."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, -0.5, 1.2]])
        mock_model.return_value = mock_output
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act
        result = analyser.analyse("Test text")
        
        # Assert
        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"Score {key}={value} is outside range [0, 1]"
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_scores_sum_to_one(self, mock_model_cls, mock_tokenizer_cls):
        """Test that sentiment scores sum to approximately 1.0."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, -0.5, 1.2]])
        mock_model.return_value = mock_output
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act
        result = analyser.analyse("Test text")
        
        # Assert
        total = sum(result.values())
        assert abs(total - 1.0) < 0.001, f"Scores sum to {total}, expected ~1.0"


class TestSentimentAnalyserErrorHandling:
    """Test error handling for various edge cases."""
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_with_none_input(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis with None input."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act & Assert
        with pytest.raises((TypeError, AttributeError)):
            analyser.analyse(None)
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_with_non_string_input(self, mock_model_cls, mock_tokenizer_cls):
        """Test analysis with non-string input."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act & Assert
        with pytest.raises((TypeError, AttributeError)):
            analyser.analyse(123)
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_handles_tokenizer_error(self, mock_model_cls, mock_tokenizer_cls):
        """Test that analyse handles tokenizer errors gracefully."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.side_effect = Exception("Tokenizer error")
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act & Assert
        with pytest.raises(Exception, match="Tokenizer error"):
            analyser.analyse("Test text")
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_handles_model_error(self, mock_model_cls, mock_tokenizer_cls):
        """Test that analyse handles model inference errors gracefully."""
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_model.side_effect = Exception("Model inference error")
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        
        # Act & Assert
        with pytest.raises(Exception, match="Model inference error"):
            analyser.analyse("Test text")


class TestSentimentAnalyserPerformance:
    """Test performance characteristics."""
    
    @patch('analyser.AutoTokenizer.from_pretrained')
    @patch('analyser.AutoModelForSequenceClassification.from_pretrained')
    def test_analyse_performance_benchmark(self, mock_model_cls, mock_tokenizer_cls):
        """Test that analysis completes within performance requirements."""
        import time
        
        # Arrange
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, -0.5, 1.2]])
        mock_model.return_value = mock_output
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        analyser = SentimentAnalyser()
        text = "This is a test sentence for performance benchmarking."
        
        # Act
        start_time = time.time()
        result = analyser.analyse(text)
        end_time = time.time()
        
        # Assert
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Analysis took {execution_time:.3f}s, should be <1s"
        assert isinstance(result, dict)


class TestSentimentAnalyserIntegration:
    """Integration tests with real components (when available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_analyse_with_gpu_if_available(self):
        """Test analysis works with GPU if available."""
        # This test would run actual model inference if GPU is available
        # For now, we'll skip it as it requires actual model weights
        pass
    
    def test_analyse_consistency(self):
        """Test that the same input produces consistent output."""
        # This would test actual model consistency
        # For now, we mock it to ensure our test structure works
        with patch('analyser.AutoTokenizer.from_pretrained'):
            with patch('analyser.AutoModelForSequenceClassification.from_pretrained'):
                # Mock setup would go here
                pass
