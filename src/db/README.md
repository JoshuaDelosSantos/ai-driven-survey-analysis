# Database Utilities Module

This directory contains standalone Python scripts for managing and manipulating the PostgreSQL database used by the AI-Driven Analysis project.

## Overview

- Centralizes all database-related operations in one place.
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
  - `test_database_operations()` — self-test: create, insert, fetch, and drop a sample table.

- **create_users_table.py**  
  Checks for and creates the `users` table:  
  - If the table exists, logs its schema.  
  - Otherwise, defines and creates the table with columns (`user_id`, `user_level`, `agency`, `created_at`).

- **load_user_data.py**  
  Loads user metadata from `src/csv/user.csv` into the `users` table:  
  1. Reads CSV via Pandas.  
  2. Validates required columns (`user_id`, `user_level`, `agency`).  
  3. Converts and batches rows into the database.

## Prerequisites

1. **Environment Variables** — Create a `.env` file at project root with:
   ```dotenv
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=<your_password>
   POSTGRES_DB=csi-db
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

## Best Practices

- Keep your `.env` file out of version control.  
- Use `test_database_operations()` to validate DB connectivity after any changes.  
- Add new table scripts here, following the same pattern: check existence → create schema → log status.

