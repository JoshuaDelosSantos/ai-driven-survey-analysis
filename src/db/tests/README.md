# Database Tests

This directory contains tests for database table creation scripts and connection validation for the AI-driven analysis project.

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
- **Vector Column**: Validates configurable vector dimensions (384/1536)

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
RAG_DB_HOST=localhost
RAG_DB_PORT=5432
RAG_DB_NAME=csi-db
RAG_DB_USER=rag_user_readonly
RAG_DB_PASSWORD=
EMBEDDING_DIMENSION=384  # For local models, 1536 for OpenAI
```
```

### Database Setup
1. **PostgreSQL with pgvector**: Ensure database is running with pgvector extension
2. **Base Tables**: Ensure users, learning_content, and evaluation tables exist
3. **RAG Read-Only Role**: Run `create_rag_readonly_role.py` to create test user
4. **RAG Embeddings Table**: Run `create_rag_embeddings_table.py` with correct `EMBEDDING_DIMENSION`
5. **Permissions**: Grant necessary permissions for testing (handled automatically)

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

## Current Status

### ✅ All Tests Passing
- **Connection Tests**: RAG read-only user properly configured
- **Table Creation Tests**: Configurable vector dimensions working (384 for local models)
- **Schema Validation**: All required columns and constraints verified
- **Index Creation**: Vector similarity and metadata indexes functioning
- **Permission Testing**: Read/write access controls validated

## Configuration Notes

### Flexible Vector Dimensions
The RAG embeddings table now supports configurable vector dimensions:
- **384 dimensions**: For local sentence transformer models (all-MiniLM-L6-v2)
- **1536 dimensions**: For OpenAI embedding models
- **Auto-configuration**: Table creation reads `EMBEDDING_DIMENSION` from environment

### Current Configuration (Local Model)
```bash
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check database is running and credentials are correct
2. **Permission Errors**: Ensure RAG read-only role exists and has proper permissions
3. **Table Missing**: Run table creation scripts in correct order
4. **Vector Dimension Mismatch**: Ensure `EMBEDDING_DIMENSION` matches your embedding provider

### Quick Resolution Steps
```bash
# 1. Recreate RAG user with proper permissions
python src/db/create_rag_readonly_role.py

# 2. Recreate embeddings table with current dimension setting
python src/db/create_rag_embeddings_table.py

# 3. Run tests
cd src/db/tests && pytest -v
```

## Integration with RAG Module

These database tests validate the foundation for the RAG embedding functionality:
- **Vector Storage**: Compatible with `rag/data/embeddings_manager.py`
- **Search Operations**: Supports semantic similarity queries
- **Australian Context**: Designed for Australian Public Service evaluation data
- **Security**: Read-only access patterns for production safety


**Last Updated**: 17 June 2025 