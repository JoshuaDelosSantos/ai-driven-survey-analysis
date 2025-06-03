"""
Learning Content Table Creator for Sentiment Analysis Project

This module creates the 'learning_content' table in the PostgreSQL database if it doesn't exist.
It handles table creation with proper schema definition and provides informative
logging about the table's existence status.

Functions:
- create_learning_content_table(): Creates the learning_content table with proper schema
- table_exists(): Checks if the learning_content table already exists

Usage:
    python create_learning_content_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def table_exists(table_name, connection):
    """
    Check if a table exists in the current database.
    
    Args:
        table_name (str): Name of the table to check
        connection: Database connection object
        
    Returns:
        bool: True if table exists, False otherwise
    """
    check_query = """
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = %s
    )
    """
    
    try:
        result = db_connector.fetch_data(check_query, (table_name,), connection=connection)
        return result[0][0]  # Returns True/False
    except Exception as e:
        print(f"[{datetime.now()}] Error checking table existence: {e}")
        return False

def create_learning_content_table():
    """
    Create the learning_content table with appropriate schema for the CSV data.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting learning_content table creation process")
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if learning_content table already exists
        if table_exists('learning_content', connection):
            print(f"[{datetime.now()}] Learning content table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'learning_content' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing learning_content table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            return
        
        # Create learning_content table if it doesn't exist
        print(f"[{datetime.now()}] Learning content table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE learning_content (
            surrogate_key INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            content_id INTEGER NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            target_level VARCHAR(50) NOT NULL,
            governing_bodies TEXT NOT NULL
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] Learning content table created successfully")
        
        # Verify table creation and show structure
        if table_exists('learning_content', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'learning_content' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] New learning_content table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
        else:
            print(f"[{datetime.now()}] Warning: Table creation may have failed")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error during table creation process: {e}")
        raise
    finally:
        if connection:
            db_connector.close_db_connection(connection)

if __name__ == "__main__":
    print(f"[{datetime.now()}] Learning content table creation script started")
    print("=" * 60)
    
    try:
        create_learning_content_table()
        print("=" * 60)
        print(f"[{datetime.now()}] Learning content table creation script completed successfully")
    except Exception as e:
        print("=" * 60)
        print(f"[{datetime.now()}] Learning content table creation script failed: {e}")
