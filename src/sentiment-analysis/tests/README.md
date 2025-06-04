# Sentiment Analysis Tests

This directory contains all automated tests for the sentiment analysis module.

## Structure
- `test_analyser.py`: Unit and integration tests for the SentimentAnalyser class
- `test_db_operations.py`: Tests for database operations (to be implemented)
- `test_data_processor.py`: Tests for data processing workflow (to be implemented)
- `conftest.py`: Shared fixtures and mocks for all tests
- `fixtures/`: Sample data and mock responses for use in tests

## Running Tests
From the `sentiment-analysis` directory, run:
```bash
pytest
```

Test discovery and configuration are managed by the `pytest.ini` file in this [directory](../tests/).
