# Sentiment Analysis Module

This directory contains all code and utilities related to performing sentiment analysis on free-text survey data and managing user metadata for the AI-Driven Analysis project.

## 1. Overview

The core objective of this module is to:

  1. Manage connection to a PostgreSQL database
  2. Create and load user metadata from CSV files
  3. Load survey/evaluation text data for analysis
  4. (Optional) Preprocess text if needed
  5. Perform sentiment analysis on free-text responses using RoBERTa
  6. Save results back into the database

All operations are implemented as standalone Python scripts that can be invoked independently or orchestrated via a workflow script.

## 2. Architecture & Data Schema

Refer to `../../documentations/architecture.md` for full project context. Key database tables for MVP 1:

- **users**: Stores user_id, user_level, agency, created_at
- **learning_content**: Course/video/live learning details with surrogate keys
- **attendance**: Tracks enrolment, status, timestamps
- **evaluation**: Free-text responses and structured fields

Each table lives in the `public` schema of a PostgreSQL database (containerised via Docker Compose).

## 3. Prerequisites & Setup

1. Install system dependencies:
    ```bash
    # macOS / Linux
    brew install docker docker-compose
    ```
2. Start the database container:
    ```bash
    cd <project-root>
    docker-compose up -d db
    ```
3. Create a Python virtual environment and install requirements:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4. Add your database credentials to a `.env` file at project root:
    ```dotenv
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=your_password
    POSTGRES_DB=csi-db
    ```
5. Ensure VS Code’s PostgreSQL extension is configured (host: `localhost`, port: `5432`) **developement only**.

## 4. Environment Variables

Loaded automatically via [`python-dotenv`](https://github.com/theskumar/python-dotenv):

- `POSTGRES_DB` – database name
- `POSTGRES_USER` – user name
- `POSTGRES_PASSWORD` – password

## 5. Key Scripts

### 5.1 `create_users_table.py`
Creates the `users` table if it doesn’t exist, or logs its structure if it does.

Usage:
```bash
python create_users_table.py
```

### 5.2 `load_user_data.py`
Reads `user.csv` under `src/csv/` and batch-inserts into the `users` table.

Usage:
```bash
python load_user_data.py
```

### 5.3 `db_connector.py`
Reusable database utilities:
- `get_db_connection()`
- `close_db_connection()`
- `fetch_data()`
- `execute_query()`
- `batch_insert_data()`
- `test_database_operations()` (self-test)

### 5.4 (Future) Analysis Pipeline
- `data_loader.py` – Fetch evaluation text for analysis
- `text_preprocessor.py` – Optional cleaning before tokenization
- `sentiment_analyzer.py` – Run RoBERTa sentiment classification
- `results_saver.py` – Insert sentiment results into `evaluation_sentiments`
- `main_workflow.py` – Orchestrate end-to-end pipeline

## 6. Usage Examples

**Test database utilities**:
```bash
python -c "import db_connector; db_connector.test_database_operations()"
```

**Load user CSV**:
```bash
python load_user_data.py
```

**Run end-to-end analysis** (future):
```bash
python main_workflow.py
```

## 7. Testing & Validation

- Each script logs detailed timestamps and error messages.
- Built-in test functions validate schema, inserts, selects, and cleanups.
