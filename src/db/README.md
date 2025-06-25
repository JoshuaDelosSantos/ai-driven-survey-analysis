# Database Utilities Module

This directory contains standalone Python scripts for managing the PostgreSQL database used by the AI-Driven Analysis project, including comprehensive table creation, data loading, and privacy-compliant user feedback collection.

## Overview

- Centralises all database-related operations with privacy-first design and Australian governance compliance
- Provides connection utilities, table creation, data loading, and feedback analytics functions
- Includes comprehensive RAG embeddings infrastructure with pgVector support
- User feedback system with 1-5 scale ratings and anonymous comment collection
- Designed for independent invocation or integration with larger workflows
- Implements read-only access controls for secure RAG system integration


## Core Database Files

### Connection and Security Management
- **db_connector.py**  
  Enhanced reusable functions for PostgreSQL connection management:
  - `get_db_connection()` — establish connections using environment credentials with security validation
  - `close_db_connection(connection, cursor=None)` — safely close resources with cleanup verification
  - `fetch_data(query, params=None, connection=None)` — run SELECT statements with result sanitisation
  - `execute_query(query, params=None, connection=None)` — run INSERT/UPDATE/DELETE with transaction support
  - `batch_insert_data(query, data_list, connection=None)` — efficiently insert multiple rows with validation
  - `table_exists(table_name, connection=None)` — check table existence with schema validation
  - `test_database_operations()` — comprehensive self-test with security validation

- **create_rag_readonly_role.py**  
  Creates dedicated read-only PostgreSQL role for secure RAG system access:
  - Creates or updates `rag_user_readonly` role with minimal required permissions
  - Grants SELECT-only access to core tables plus feedback table support
  - Explicitly denies INSERT, UPDATE, DELETE, TRUNCATE, and CREATE permissions
  - Validates security constraints and documents Australian Privacy Principles (APP) compliance
  - Implements defence-in-depth security for hybrid Text-to-SQL and vector search functionality

### Core Data Tables
- **create_users_table.py** — User profiles with agency and level information
- **load_user_data.py** — User data loading from CSV with validation
- **create_learning_content_table.py** — Learning content metadata and classification
- **load_learning_content_data.py** — Content data loading with duplicate detection
- **create_attendance_table.py** — Attendance tracking with foreign key constraints
- **load_attendance_data.py** — Attendance data loading with relationship validation
- **create_evaluation_table.py** — Course evaluation responses with comprehensive feedback fields
- **load_evaluation_data.py** — Evaluation data loading with data integrity checks
- **create_sentiment_table.py** — Sentiment analysis results storage (legacy support)

### RAG System Infrastructure
- **create_rag_embeddings_table.py**  
  Creates the `rag_embeddings` table for vector embeddings storage:
  - Enables pgvector extension with configurable vector dimensions (384 for local, 1536 for OpenAI)
  - Includes foreign key constraints to evaluation table for data integrity
  - Creates optimised indexes for vector similarity search (ivfflat) and metadata filtering (GIN)
  - Supports multiple embedding model versions with comprehensive metadata storage
  - Privacy-compliant design with anonymised text chunk storage

- **create_rag_user_feedback_table.py** ✅ NEW (Phase 3)
  Creates the `rag_user_feedback` table for user satisfaction monitoring:
  - Stores 1-5 scale ratings with optional anonymous text comments
  - Includes query and response context for analytics and system improvement
  - Implements PII anonymisation fields for privacy compliance and Australian governance
  - Creates optimised indexes for session tracking, rating analysis, and temporal queries
  - Supports real-time feedback analytics and system quality monitoring
  - Follows project patterns for table creation, logging, and security validation

### Testing and Validation
- **tests/test_rag_connection.py**  
  Validates RAG read-only database security:
  - Tests connection success with `rag_user_readonly` role
  - Verifies SELECT operations work on all required tables including feedback table
  - Confirms write operations are properly blocked for security compliance
  - Tests complex JOIN queries for analytical capabilities
  - Pytest framework integration for automated validation

- **tests/test_create_rag_embeddings_table.py**  
  Comprehensive RAG embeddings infrastructure testing:
  - Tests pgvector extension and vector column configuration
  - Validates table structure, constraints, and index creation
  - Verifies configurable vector dimensions and model compatibility
  - Tests privacy-compliant design and data integrity features

## Prerequisites

1. **Environment Variables** — Create a `.env` file at project root with:
   ```dotenv
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=<your_password>
   POSTGRES_DB=csi-db
   
   # RAG Module Database Access (read-only)
   RAG_DB_USER=rag_user_readonly
   RAG_DB_PASSWORD=rag_secure_readonly_2025
   RAG_DB_HOST=localhost
   RAG_DB_PORT=5432
   RAG_DB_NAME=csi-db
   
   # Embedding Configuration (for RAG table creation)
   EMBEDDING_DIMENSION=384  # 384 for local models, 1536 for OpenAI
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

### Create RAG User Feedback Table
```bash
python create_rag_user_feedback_table.py    # creates table for RAG system user feedback
```

### Test RAG Database Connection and Security
```bash
cd tests
pytest test_rag_connection.py -v              # validates RAG security constraints
pytest test_create_rag_embeddings_table.py -v # tests embeddings table creation
pytest -v                                     # run all database tests
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
python create_rag_user_feedback_table.py    # creates table for RAG system user feedback
```

## Security and Testing

The database module includes comprehensive security measures and testing infrastructure for the RAG (Retrieval-Augmented Generation) integration:

- **Read-Only Access Control**: The `rag_user_readonly` role provides minimal privileges following the principle of least privilege.
- **Security Validation**: Automated tests verify that write operations are blocked and only authorised read operations are permitted.
- **Comprehensive Testing**: Full test suite covering both security constraints and table creation functionality.
- **Compliance Ready**: Security setup supports Australian Privacy Principles (APP) compliance and audit logging requirements.
- **Defence-in-Depth**: Multiple layers of security including explicit permission denial and connection testing.

### Current Test Status
- **✅ All Tests Passing**: Both RAG connection and embeddings table tests are fully functional
- **Configurable Dimensions**: Vector table supports both local (384) and cloud (1536) embedding models
- **Production Ready**: Security constraints validated and embeddings infrastructure tested

### Running Security Tests
```bash
# Test RAG database security constraints
cd tests && pytest test_rag_connection.py -v

# Test embeddings table creation and configuration
cd tests && pytest test_create_rag_embeddings_table.py -v

# Run all database tests
cd tests && pytest -v

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
- **Vector Dimensions**: Configure `EMBEDDING_DIMENSION` appropriately for your embedding provider before creating the RAG embeddings table.

---
**Last Updated**: 25 June 2025

