# Sentiment Analysis Module

This directory contains all code and utilities related to performing sentiment analysis on free-text survey data and managing user metadata for the AI-Driven Analysis project.

## Overview

This module uses the Hugging Face RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) to perform sentiment analysis on the "general_feedback" column and other free-text fields in the evaluation table. The sentiment analysis results are stored back in the database for further analysis and reporting.

## Architecture 
The sentiment-analysis module adopts a class-based structure to ensure maintainability and testability:

1. **config.py**
   - Centralises configuration constants (model name, database URI, table names, and free-text columns).

2. **analyser.py**
   - Defines `SentimentAnalyser`:
     - `__init__`: loads the Hugging Face tokenizer and model.
     - `analyse(text: str) -> dict`: returns a sentiment scores dictionary `{neg, neu, pos}`.

3. **db_operations.py**
   - Defines `DBOperations`:
     - `__init__`: establishes a database connection using credentials from `config.py`.
     - `write_sentiment(response_id: int, column: str, scores: dict)`: upserts sentiment scores into a dedicated table.

4. **data_processor.py**
   - Defines `DataProcessor`:
     - `__init__`: accepts instances of `DBOperations` and `SentimentAnalyser`.
     - `process_all()`: fetches evaluation rows, iterates over configured free-text fields, calls `analyse`, and persists results via `DBOperations`.

5. **runner.py**
   - Script entry point:
     - Ensures sentiment table exists by invoking `src/db/create_sentiment_table.py`.
     - Parses any CLI arguments (optional).
     - Instantiates `SentimentAnalyser`, `DBOperations`, and `DataProcessor`.
     - Calls `DataProcessor.process_all()` to execute the pipeline.

### Dependency: Table Creation Script
Before running the sentiment pipeline, ensure the sentiment table is created:
```bash
python src/db/create_sentiment_table.py
```

### File Structure
```
src/sentiment-analysis/
├── analyser.py
├── config.py
├── data_processor.py
├── db_operations.py
├── runner.py
└── tests/
    ├── __init__.py
    ├── test_analyser.py              # SentimentAnalyser tests
    ├── test_db_operations.py         # DBOperations tests
    ├── test_data_processor.py        # DataProcessor tests
    └── fixtures/
        ├── sample_data.py            # Test data fixtures
        └── mock_responses.py         # Mock API responses
```

## Testing

### Testing Philosophy
- **Comprehensive Coverage**: Test all components, methods, and edge cases
- **Isolated Testing**: Each test should be independent and not rely on external state
- **Fast Feedback**: Tests should run quickly to enable rapid development cycles
- **Documentation**: Tests serve as living documentation of expected behavior

### Testing Framework
**Primary Framework**: `pytest`
- Easy to write and read tests
- Excellent fixture support for setup/teardown
- Rich assertion introspection

**Additional Libraries**:
- `pytest-mock`: For mocking dependencies
- `pytest-cov`: For test coverage reporting
- `pytest-xdist`: For parallel test execution

### Test Categories

#### Core Component Tests
**Purpose**: Test the three main components in isolation and integration

- **SentimentAnalyser Tests** (`test_analyser.py`):
  - Model and tokenizer initialisation
  - Text analysis with various inputs (normal text, empty strings, long text, special characters)
  - Output format validation (score structure and ranges 0-1)
  - Error handling for invalid inputs

- **DBOperations Tests** (`test_db_operations.py`):
  - SQL query generation and parameter binding
  - Upsert functionality with conflict resolution
  - Error handling for database failures
  - Return value validation

- **DataProcessor Tests** (`test_data_processor.py`):
  - Data fetching and processing workflow
  - Integration with SentimentAnalyser and DBOperations
  - Error handling for individual records
  - Empty dataset and edge case handling

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock pytest-cov

# Run all tests
pytest

# Run specific test files
pytest tests/test_analyser.py         # SentimentAnalyser tests only
pytest tests/test_db_operations.py   # DBOperations tests only
pytest tests/test_data_processor.py  # DataProcessor tests only

# Run with coverage reporting
pytest --cov=. --cov-report=html

# Run tests with verbose output
pytest -v
```

### Test Configuration
Create `pytest.ini` in the sentiment-analysis directory:
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]  
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--cov=.",
    "--cov-report=term-missing",
    "--cov-fail-under=80"
]
```

### Quality Gates
- **Minimum Coverage**: 80% overall test coverage
- **Target Coverage**: 90% for core components (analyser, db_operations, data_processor)
- **Performance**: Analysis speed <1 second per evaluation record
- **All tests must pass before deployment**
