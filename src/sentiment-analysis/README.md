# Sentiment Analysis Module

This directory contains all code and utilities related to performing sentiment analysis on free-text survey data and managing user metadata for the AI-Driven Analysis project.

## Overview

This module uses the Hugging Face RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) to perform sentiment analysis on the "general_feedback" column and other free-text fields in the evaluation table. The sentiment analysis results are stored back in the database for further analysis and reporting.

## Architecture

The sentiment analysis pipeline follows a modular design with distinct components:

### Core Components

1. **data_loader.py**
   - Purpose: Extracts free-text data from the PostgreSQL database
   - Responsibilities:
     - Queries the evaluation table for unprocessed feedback text
     - Converts database results to Pandas DataFrames
     - Filters records that need sentiment analysis

2. **text_preprocessor.py**
   - Purpose: Prepares text data for the sentiment model
   - Responsibilities:
     - Normalises whitespace and formatting
     - Handles missing or invalid data
     - Prepares text for tokenisation

3. **sentiment_classifier.py**
   - Purpose: Performs the actual sentiment analysis using RoBERTa
   - Responsibilities:
     - Loads and initializes the RoBERTa model
     - Tokenizes input text
     - Classifies sentiment (Negative, Neutral, Positive)
     - Calculates sentiment scores

4. **db_operations.py**
   - Purpose: Manages database operations for sentiment data
   - Responsibilities:
     - Creates sentiment results table
     - Stores sentiment analysis results
     - Handles database transactions and error recovery

5. **visualization.py**
   - Purpose: Creates visualisations of sentiment analysis results
   - Responsibilities:
     - Generates sentiment distribution charts
     - Produces content-based sentiment comparisons
     - Exports visualisation files

6. **evaluate_feedback.py**
   - Purpose: Main orchestration script
   - Responsibilities:
     - Coordinates the entire workflow
     - Processes batches of feedback data
     - Handles logging and error reporting

7. **workflow_scheduler.py**
   - Purpose: Monitors for new feedback data and triggers analysis
   - Responsibilities:
     - Runs as a background service or scheduled task
     - Detects new records in the evaluation table
     - Triggers the sentiment analysis pipeline for new data
     - Maintains processing state to avoid duplicate analysis

### Database Schema

The module creates and populates an `evaluation_sentiment` table with the following structure:

- `response_id`: PRIMARY KEY, linked to evaluation table
- `sentiment_label`: Text classification (Negative, Neutral, Positive)
- `sentiment_score`: Normalized score (-1 to 1)
- `negative_score`: Raw model score for negative sentiment
- `neutral_score`: Raw model score for neutral sentiment
- `positive_score`: Raw model score for positive sentiment
- `processed_at`: Timestamp of analysis

Additionally, a tracking column is added to the `evaluation` table:

- `sentiment_processed`: Boolean flag to track which records have been analyzed

## Usage

Run the sentiment analysis pipeline to analyse all unprocessed feedback:

```bash
python evaluate_feedback.py
```

## Real-Time Processing

For real-time sentiment analysis of new evaluation data, the system includes a monitoring component that automatically processes new feedback as it arrives. The following approaches are supported:

### 1. Scheduled Polling

The `workflow_scheduler.py` component runs as a background service that periodically checks for new evaluation records:

```bash
# Run as a continuous service
python workflow_scheduler.py --interval 300  # Check every 5 minutes
```

### 2. Database Triggers (Chosen way)

PostgreSQL triggers can be set up to automatically initiate sentiment analysis when new data is inserted:

```sql
CREATE OR REPLACE FUNCTION trigger_sentiment_analysis() RETURNS TRIGGER AS $$
BEGIN
    -- Insert a job into a queue table or call a notification system
    INSERT INTO analysis_queue (record_id, table_name, created_at)
    VALUES (NEW.response_id, 'evaluation', NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER evaluation_insert_trigger
AFTER INSERT ON evaluation
FOR EACH ROW
EXECUTE FUNCTION trigger_sentiment_analysis();
```

### 3. Message Queue Integration

For higher scalability, the system can integrate with message queues (like RabbitMQ or Kafka):

1. A producer component captures database changes and publishes messages
2. A consumer component subscribes to these messages and triggers sentiment analysis
3. The results are written back to the database

### 4. Change Data Capture (CDC)

Monitor the PostgreSQL Write-Ahead Log (WAL) using tools like Debezium to capture changes to the evaluation table and trigger the sentiment analysis pipeline.

## Dependencies

- transformers: Hugging Face transformer models
- torch: PyTorch for model inference
- pandas: Data manipulation
- matplotlib: Visualization
- psycopg2-binary: PostgreSQL database connection

## Extension Points

The architecture is designed to be extendable in several ways:

1. Additional text columns can be added to the analysis pipeline
2. The sentiment model can be fine-tuned on domain-specific data
3. More sophisticated text preprocessing can be implemented
4. Advanced visualizations can be added to the reporting system
