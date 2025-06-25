"""
RAG User Feedback Table Creator

This module creates the 'rag_user_feedback' table in the PostgreSQL database for storing
user feedback on RAG system responses. It follows the same patterns as other table
creation scripts in this project.

Table Schema:
- Stores user feedback with 1-5 rating scale and optional comments
- Links feedback to queries/responses for context
- Includes PII anonymisation fields for privacy compliance
- Tracks response sources and metadata

Functions:
- create_rag_user_feedback_table(): Creates the table with proper schema

Usage:
    python create_rag_user_feedback_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def create_rag_user_feedback_table():
    """
    Create the rag_user_feedback table with appropriate schema for feedback collection.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting rag_user_feedback table creation process")
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if rag_user_feedback table already exists
        if db_connector.table_exists('rag_user_feedback', connection):
            print(f"[{datetime.now()}] rag_user_feedback table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'rag_user_feedback' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing rag_user_feedback table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            return
        
        # Create rag_user_feedback table if it doesn't exist
        print(f"[{datetime.now()}] rag_user_feedback table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE rag_user_feedback (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            query_id VARCHAR(255) NOT NULL,
            query_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
            comment TEXT,
            response_sources TEXT[],
            anonymised_comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] rag_user_feedback table created successfully")
        
        # Create indexes for performance
        print(f"[{datetime.now()}] Creating indexes for rag_user_feedback table")
        
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_feedback_session ON rag_user_feedback(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON rag_user_feedback(rating)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_created ON rag_user_feedback(created_at)"
        ]
        
        for index_query in index_queries:
            db_connector.execute_query(index_query, connection=connection)
        
        print(f"[{datetime.now()}] Indexes created successfully")
        
        # Verify table creation and show structure
        if db_connector.table_exists('rag_user_feedback', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'rag_user_feedback' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] rag_user_feedback table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
                
            # Show indexes
            index_query = """
            SELECT indexname, indexdef
            FROM pg_indexes 
            WHERE tablename = 'rag_user_feedback'
            """
            
            indexes = db_connector.fetch_data(index_query, connection=connection)
            print(f"[{datetime.now()}] Table indexes:")
            for index in indexes:
                print(f"  - {index[0]}")
        else:
            print(f"[{datetime.now()}] ERROR: Table creation could not be verified")
            raise Exception("Table creation verification failed")
            
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to create rag_user_feedback table: {e}")
        raise
        
    finally:
        if connection:
            db_connector.close_db_connection(connection)
            print(f"[{datetime.now()}] Database connection closed")

if __name__ == "__main__":
    """
    Main execution block - create the rag_user_feedback table when script is run directly.
    """
    try:
        print(f"[{datetime.now()}] Starting RAG User Feedback table creation script")
        create_rag_user_feedback_table()
        print(f"[{datetime.now()}] RAG User Feedback table creation script completed successfully")
        
    except Exception as e:
        print(f"[{datetime.now()}] SCRIPT FAILED: {e}")
        exit(1)
