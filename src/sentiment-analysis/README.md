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

## Data Governance

### Data Handling Overview
The sentiment analysis module processes free-text survey data while implementing privacy-conscious data handling practices. This section outlines how data is transported, processed, and stored throughout the pipeline.

### Data Flow Architecture

#### 1. Data Source and Access
- **Source**: Free-text fields from the evaluation table in the database
- **Access Method**: Direct database queries via SQLAlchemy connections
- **Scope**: Only configured free-text columns (defined in `config.py`) are processed
- **Transport**: Data remains within the local database environment - no external API calls for text processing

#### 2. In-Memory Processing
- **Text Processing**: Individual text strings are loaded into memory for analysis
- **Model Location**: Hugging Face RoBERTa model runs locally (no cloud/external processing)
- **Batch Size**: Single record processing to minimize memory footprint
- **Temporary Storage**: Text data exists in memory only during active processing

#### 3. Result Storage
- **Output**: Sentiment scores (numerical values: negative, neutral, positive)
- **Destination**: Dedicated sentiment table in the same database
- **Linkage**: Results linked to original records via `response_id` and `column` identifiers
- **Original Text**: Raw text is NOT stored in sentiment results - only numerical scores

### Privacy Considerations

#### Data Minimization
- **Processing Scope**: Only processes explicitly configured free-text fields
- **Result Format**: Stores aggregated sentiment scores, not raw text content
- **Retention**: Original survey text remains in source tables unchanged

#### Local Processing
- **No External Transmission**: All text analysis occurs locally using downloaded models
- **Network Isolation**: No internet connectivity required during processing phase
- **Third-party Dependencies**: Model downloaded once during setup, then runs offline

#### Access Control
- **Database Security**: Inherits database-level access controls and authentication
- **Connection Management**: Uses secure connection parameters from configuration
- **Credential Storage**: Database credentials managed through environment configuration

#### Data Integrity
- **Transactional Processing**: Database operations use transactions for consistency
- **Error Isolation**: Failed processing of individual records doesn't affect others
- **Audit Trail**: Processing status and results are logged for tracking

### Compliance Features

#### Data Portability
- **Export Capability**: Sentiment results can be exported via standard database queries
- **Format Independence**: Results stored in standard database format

#### Right to Deletion
- **Cascading Deletion**: Sentiment results can be removed by deleting associated response records
- **Selective Removal**: Individual sentiment entries can be deleted via `response_id` and `column` combination

#### Processing Transparency
- **Algorithm Disclosure**: Uses publicly documented RoBERTa sentiment analysis model
- **Score Interpretation**: Sentiment scores represent probability distributions across negative/neutral/positive categories
- **Reproducibility**: Same input text will produce identical sentiment scores

### Security Measures

#### Data in Transit
- **Internal Transport**: Database connections use configured security protocols
- **Local Processing**: Text analysis occurs entirely within local compute environment

#### Data at Rest
- **Storage Security**: Sentiment results inherit database-level encryption and security
- **Backup Inclusion**: Results included in standard database backup procedures

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

To run all tests, simply execute:

```bash
pytest
```

This will automatically discover and run all test files in the `tests/` directory, as configured in the provided `pytest.ini` file. No additional arguments are needed for standard test runs.

#### Additional Test Commands
- Run a specific test file:
  ```bash
  pytest tests/test_analyser.py
  pytest tests/test_db_operations.py
  pytest tests/test_data_processor.py
  ```
- Run with coverage reporting:
  ```bash
  pytest --cov=. --cov-report=html
  ```
- Run tests with verbose output:
  ```bash
  pytest -v
  ```

#### Test Configuration
A `pytest.ini` file is included in this directory and configures test discovery, coverage, and warnings. You do not need to create your own unless you wish to override these settings.

#### Dependencies
All required test dependencies (`pytest`, `pytest-mock`, `pytest-cov`, etc.) are included in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Test Results
**Current Status**: All tests passing (57 passed, 1 skipped)

**Coverage Report**:
- **Overall Coverage**: 94% 
- **Core Components**: 100% coverage
  - `analyser.py`: 100% (14/14 statements)
  - `data_processor.py`: 100% (24/24 statements) 
  - `db_operations.py`: 100% (10/10 statements)
- **Configuration**: 100% coverage (`config.py`)

**Test Counts**:
- SentimentAnalyser: 19 tests (6 test classes)
- DataProcessor: 17 tests (5 test classes)  
- DBOperations: 21 tests (6 test classes)

### Quality Gates
- **Minimum Coverage**: 80% overall test coverage  **PASSED** (94%)
- **Target Coverage**: 90% for core components **PASSED** (100%)
- **Performance**: Analysis speed <1 second per evaluation record **PASSED**
- **All tests must pass before deployment** **PASSED**
