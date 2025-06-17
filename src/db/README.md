# Database Utilities Module

This directory contains standalone Python scripts for managing and manipulating the PostgreSQL database used by the AI-Driven Analysis project.

## Overview

- Centralises all database-related operations in one place.
- Provides connection utilities, table creation, and data loading functions.
- Designed to be invoked independently or as part of larger workflows.


## Files

- **db_connector.py**  
  Reusable functions for connecting to and interacting with PostgreSQL:
  - `get_db_connection()` — establish a connection using `.env` credentials.
  - `close_db_connection(connection, cursor=None)` — safely close resources.
  - `fetch_data(query, params=None, connection=None)` — run `SELECT` statements.
  - `execute_query(query, params=None, connection=None)` — run `INSERT`/`UPDATE`/`DELETE` with transaction support.
  - `batch_insert_data(query, data_list, connection=None)` — efficiently insert multiple rows.
  - `table_exists(table_name, connection=None)` — check if a table exists in the database.
  - `test_database_operations()` — self-test: create, insert, fetch, and drop a sample table.

- **create_users_table.py**  
  Checks for and creates the `users` table:
  - If the table exists, logs its schema.
  - Otherwise, defines and creates the table with columns (`user_id`, `user_level`, `agency`, `created_at`).

- **load_user_data.py**  
  Loads user data from `src/csv/user.csv` into the `users` table:
  1. Reads CSV via Pandas.
  2. Validates required columns (`user_id`, `user_level`, `agency`).
  3. Converts and batches rows into the database.

- **create_learning_content_table.py**  
  Checks for and creates the `learning_content` table:
  - If the table exists, logs its schema.
  - Otherwise, defines and creates the table with columns (`surrogate_key`, `name`, `content_id`, `content_type`, `target_level`, `governing_bodies`, `created_at`).

- **load_learning_content_data.py**  
  Loads learning content data from `src/csv/learning_content.csv` into the `learning_content` table:
  1. Reads CSV via Pandas.
  2. Validates required columns and checks for duplicate surrogate_keys.
  3. Converts and batches rows into the database with comprehensive logging.

- **create_attendance_table.py**  
  Checks for and creates the `attendance` table:
  - If the table exists, logs its schema.
  - Otherwise, defines and creates the table with columns (`attendance_id`, `user_id`, `learning_content_surrogate_key`, `date_effective`, `status`).
  - Includes foreign key constraints to `users` and `learning_content` tables.

- **load_attendance_data.py**  
  Loads attendance data from `src/csv/attendance.csv` into the `attendance` table:
  1. Reads CSV via Pandas.
  2. Validates required columns and checks for duplicate attendance_ids.
  3. Converts and batches rows into the database with joins to display related data.

- **create_evaluation_table.py**  
  Checks for and creates the `evaluation` table:
  - If the table exists, logs its schema.
  - Otherwise, defines and creates the table with the columns defined in `data-dictionary.json`.
  - Includes foreign key constraints to `users` and `learning_content` tables.

- **load_evaluation_data.py**  
  Loads evaluation data from `src/csv/evaluation.csv` into the `evaluation` table:
  1. Reads CSV via Pandas.
  2. Validates required columns and checks for duplicate response_ids.
  3. Converts and batches rows into the database with joins to display related data.

- **create_sentiment_table.py**  
  Checks for and creates the `evaluation_sentiment` table:
  - If the table exists, logs status and skips creation.
  - Otherwise, creates `evaluation_sentiment` with columns (`response_id`, `column_name`, `neg`, `neu`, `pos`, `created_at`) and foreign key to `evaluation(response_id)`.

- **create_rag_readonly_role.py**  
  Creates a dedicated read-only PostgreSQL role for the RAG (Retrieval-Augmented Generation) module:
  - Creates or updates `rag_user_readonly` role with minimal required permissions.
  - Grants SELECT-only access to `attendance`, `users`, `learning_content`, and `evaluation` tables.
  - Explicitly denies INSERT, UPDATE, DELETE, TRUNCATE, and CREATE permissions.
  - Validates security constraints and documents compliance with Australian Privacy Principles (APP).
  - Implements defence-in-depth security for the Text-to-SQL functionality.

- **create_rag_embeddings_table.py**  
  Creates the `rag_embeddings` table for vector embeddings storage in the RAG system:
  - Enables the pgvector extension if not already active.
  - Creates the table with proper schema for storing text embeddings, metadata, and chunk information.
  - Includes foreign key constraint to `evaluation(response_id)` for data integrity.
  - Creates optimized indexes for vector similarity search (ivfflat) and metadata filtering (GIN).
  - Supports configurable vector dimensions and embedding model versioning.

- **tests/test_rag_connection.py**  
  Validates the RAG read-only database connection and security constraints:
  - Tests successful connection with the `rag_user_readonly` role.
  - Verifies SELECT operations work on all required tables.
  - Confirms write operations (INSERT, UPDATE, DELETE, CREATE) are properly blocked.
  - Tests complex JOIN queries to ensure analytical capabilities.
  - Uses pytest framework for automated testing and validation.

## Prerequisites

1. **Environment Variables** — Create a `.env` file at project root with:
   ```dotenv
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=<your_password>
   POSTGRES_DB=csi-db
   
   # RAG Module Database Access (read-only)
   RAG_DB_USER=rag_user_readonly
   RAG_DB_PASSWORD=rag_secure_readonly_password
   RAG_DB_HOST=IP
   RAG_DB_PORT=port#
   RAG_DB_NAME=db-name
   ```
2. **Dependencies** — Ensure Python packages are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **Database Container** — Start Postgres with pgvector extension:
   ```bash
   docker-compose up -d db
   ```

## Usage Examples

### Test Connection and Utilities
```bash
python db_connector.py           # runs self-test suite
```

### Create Users Table
```bash
python create_users_table.py      # idempotent: logs if table exists
```

### Load User CSV Data
```bash
python load_user_data.py          # reads src/csv/user.csv and batch-inserts into users
```

### Create Learning Content Table
```bash
python create_learning_content_table.py    # idempotent: logs if table exists
```

### Load Learning Content CSV Data
```bash
python load_learning_content_data.py       # reads src/csv/learning_content.csv and batch-inserts into learning_content
```

### Create Attendance Table
```bash
python create_attendance_table.py          # idempotent: logs if table exists
```

### Load Attendance CSV Data
```bash
python load_attendance_data.py             # reads src/csv/attendance.csv and batch-inserts into attendance
```

### Create Evaluation Table
```bash
python create_evaluation_table.py          # idempotent: logs if table exists
```

### Load Evaluation CSV Data
```bash
python load_evaluation_data.py             # reads src/csv/evaluation.csv and batch-inserts into evaluation
```

### Create Sentiment Table
```bash
python create_sentiment_table.py           # idempotent: logs if table exists
```

### Create RAG Read-Only Role
```bash
python create_rag_readonly_role.py          # creates secure read-only role for RAG module
```

### Create RAG Embeddings Table
```bash
python create_rag_embeddings_table.py       # creates table for vector embeddings in RAG
```

### Test RAG Database Connection
```bash
python -m pytest tests/test_rag_connection.py -v    # validates RAG security constraints
```

## Setup

Run the following scripts in sequence to initialise tables and load CSV data:
```bash
python create_users_table.py      # create 'users' table
python load_user_data.py          # load user data from CSV
python create_learning_content_table.py    # create 'learning_content' table
python load_learning_content_data.py       # load learning content data from CSV
python create_attendance_table.py          # create 'attendance' table
python load_attendance_data.py             # load attendance data from CSV
python create_evaluation_table.py          # create 'evaluation' table
python load_evaluation_data.py             # load evaluation data from CSV
python create_sentiment_table.py           # create 'evaluation_sentiment' table
python create_rag_readonly_role.py          # create secure read-only role for RAG module
python create_rag_embeddings_table.py       # creates table for vector embeddings in RAG
```

## Security and Testing

The database module includes comprehensive security measures for the RAG (Retrieval-Augmented Generation) integration:

- **Read-Only Access Control**: The `rag_user_readonly` role provides minimal privileges following the principle of least privilege.
- **Security Validation**: Automated tests verify that write operations are blocked and only authorised read operations are permitted.
- **Compliance Ready**: Security setup supports Australian Privacy Principles (APP) compliance and audit logging requirements.
- **Defence-in-Depth**: Multiple layers of security including explicit permission denial and connection testing.

### Running Security Tests
```bash
# Test RAG database security constraints
python -m pytest tests/test_rag_connection.py -v

# Test all database operations
python db_connector.py
```

## Best Practices

- Keep your `.env` file out of version control.  
- Use `test_database_operations()` to validate DB connectivity after any changes.  
- Add new table scripts here, following the same pattern: check existence → create schema → log status.
- **Security**: Always use the dedicated `rag_user_readonly` role for RAG module database access.
- **Testing**: Run security validation tests after any changes to database roles or permissions.
- **Compliance**: Maintain audit logs and document any changes to database access controls.

