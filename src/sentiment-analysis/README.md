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
└── runner.py
```
