# Database Tests

This directory contains tests for database table creation scripts and connection validation.

## Available Tests

### `test_rag_connection.py`
Tests the RAG read-only database connection and validates security constraints:
- **Connection Test**: Verifies RAG user can connect to database
- **SELECT Operations**: Confirms read access to required tables
- **Write Protection**: Validates that write operations are blocked
- **Complex Queries**: Tests JOIN operations for analytical queries

**Usage:**
```bash
cd src/db/tests
python test_rag_connection.py
```

### `test_create_rag_embeddings_table.py`
Tests the RAG embeddings table creation and schema validation:
- **pgVector Extension**: Verifies pgvector extension is enabled
- **Table Creation**: Tests table creation script execution
- **Schema Structure**: Validates all required columns exist
- **Foreign Key Constraint**: Checks reference to evaluation table
- **Indexes**: Verifies vector and metadata indexes are created
- **Vector Column**: Validates vector column type and configuration

**Usage with pytest (recommended):**
```bash
cd src/db/tests
pytest test_create_rag_embeddings_table.py -v
```

**Usage standalone:**
```bash
cd src/db/tests
python test_create_rag_embeddings_table.py
```

## Prerequisites

### Environment Variables
Ensure your `.env` file contains:
```bash
RAG_DB_HOST=
RAG_DB_PORT=
RAG_DB_NAME=
RAG_DB_USER=
RAG_DB_PASSWORD=
```

### Database Setup
1. **PostgreSQL with pgvector**: Ensure database is running
2. **RAG Read-Only Role**: Run `create_rag_readonly_role.py` first
3. **Evaluation Table**: Ensure evaluation table exists for foreign key constraints
4. **RAG Embeddings Table**: Run `create_rag_embeddings_table.py` before testing

## Running All Tests

## Running All Tests

### Using pytest (recommended):
```bash
cd src/db/tests
pytest -v
```

### Using individual scripts:
```bash
cd src/db/tests
python test_rag_connection.py
python test_create_rag_embeddings_table.py
```

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check database is running and credentials are correct
2. **Missing Tables**: Ensure prerequisite tables (evaluation) exist
3. **Permission Errors**: Verify RAG read-only role has been created
4. **pgVector Extension**: Ensure PostgreSQL has pgvector extension installed

### Test Dependencies

Tests must be run in order:
1. Create base tables (users, learning_content, evaluation)
2. Create RAG read-only role
3. Create RAG embeddings table
4. Run tests

## Security Notes

- Tests use read-only credentials where possible
- Write operation tests verify security constraints
- No test data is permanently stored in database
- All operations are logged for audit purposes
