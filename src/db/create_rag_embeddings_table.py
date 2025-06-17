"""
RAG Embeddings Table Creator for AI-Driven Analysis Project

This module creates the 'rag_embeddings' table in the PostgreSQL database if it doesn't exist.
This table stores vector embeddings for free-text fields from the evaluation table,
supporting semantic search capabilities in the RAG system.

The table is designed to support:
- Multiple embedding models with versioning
- Text chunking for long content
- Rich metadata for filtering and analysis
- Efficient vector similarity search with pgvector

Functions:
- create_rag_embeddings_table(): Creates the rag_embeddings table with proper schema
- enable_pgvector_extension(): Ensures pgvector extension is enabled

Usage:
    python create_rag_embeddings_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def enable_pgvector_extension():
    """
    Enable the pgvector extension if not already enabled.
    This is required for vector operations.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Checking pgvector extension status")
        connection = db_connector.get_db_connection()
        
        # Check if pgvector extension exists
        check_extension_query = """
        SELECT EXISTS(
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        )
        """
        
        result = db_connector.fetch_data(check_extension_query, connection=connection)
        extension_exists = result[0][0] if result else False
        
        if extension_exists:
            print(f"[{datetime.now()}] pgvector extension is already enabled")
        else:
            print(f"[{datetime.now()}] Enabling pgvector extension")
            enable_extension_query = "CREATE EXTENSION IF NOT EXISTS vector"
            db_connector.execute_query(enable_extension_query, connection=connection)
            print(f"[{datetime.now()}] pgvector extension enabled successfully")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error managing pgvector extension: {e}")
        raise
    finally:
        if connection:
            db_connector.close_db_connection(connection)

def create_rag_embeddings_table():
    """
    Create the rag_embeddings table with appropriate schema for vector storage.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting rag_embeddings table creation process")
        
        # First ensure pgvector extension is enabled
        enable_pgvector_extension()
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if rag_embeddings table already exists
        if db_connector.table_exists('rag_embeddings', connection):
            print(f"[{datetime.now()}] rag_embeddings table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'rag_embeddings' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing rag_embeddings table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            # Show indexes
            index_query = """
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'rag_embeddings'
            """
            indexes = db_connector.fetch_data(index_query, connection=connection)
            if indexes:
                print(f"[{datetime.now()}] Existing indexes:")
                for idx in indexes:
                    print(f"  - {idx[0]}: {idx[1]}")
            
            return
        
        # Create rag_embeddings table if it doesn't exist
        print(f"[{datetime.now()}] rag_embeddings table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE rag_embeddings (
            embedding_id SERIAL PRIMARY KEY,
            response_id INTEGER NOT NULL,
            field_name VARCHAR(50) NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding VECTOR(1536) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_chunk UNIQUE(response_id, field_name, chunk_index),
            CONSTRAINT fk_response_id FOREIGN KEY (response_id) REFERENCES evaluation(response_id) ON DELETE CASCADE
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] rag_embeddings table created successfully")
        
        # Create indexes for efficient vector operations and filtering
        print(f"[{datetime.now()}] Creating indexes for optimal query performance")
        
        indexes = [
            "CREATE INDEX ON rag_embeddings USING ivfflat (embedding vector_cosine_ops)",
            "CREATE INDEX ON rag_embeddings (response_id)",
            "CREATE INDEX ON rag_embeddings (field_name)",
            "CREATE INDEX ON rag_embeddings (model_version)",
            "CREATE INDEX ON rag_embeddings USING GIN (metadata)"
        ]
        
        for index_query in indexes:
            try:
                db_connector.execute_query(index_query, connection=connection)
                print(f"[{datetime.now()}] Index created: {index_query.split('(')[0].strip()}")
            except Exception as e:
                print(f"[{datetime.now()}] Warning: Could not create index - {e}")
        
        # Verify table creation and show structure
        if db_connector.table_exists('rag_embeddings', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'rag_embeddings' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] New rag_embeddings table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
                
            # Show created indexes
            index_query = """
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'rag_embeddings'
            """
            indexes = db_connector.fetch_data(index_query, connection=connection)
            if indexes:
                print(f"[{datetime.now()}] Created indexes:")
                for idx in indexes:
                    print(f"  - {idx[0]}: {idx[1]}")
        else:
            print(f"[{datetime.now()}] Warning: Table creation may have failed")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error during table creation process: {e}")
        raise
    finally:
        if connection:
            db_connector.close_db_connection(connection)

if __name__ == "__main__":
    print(f"[{datetime.now()}] RAG embeddings table creation script started")
    print("=" * 70)
    
    try:
        create_rag_embeddings_table()
        print("=" * 70)
        print(f"[{datetime.now()}] RAG embeddings table creation script completed successfully")
    except Exception as e:
        print("=" * 70)
        print(f"[{datetime.now()}] RAG embeddings table creation script failed: {e}")
