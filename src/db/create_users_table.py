"""
Users Table Creator for Sentiment Analysis Project

This module creates the 'users' table in the PostgreSQL database if it doesn't exist.
It handles table creation with proper schema definition and provides informative
logging about the table's existence status.

Functions:
- create_users_table(): Creates the users table with proper schema

Usage:
    python create_users_table.py

Dependencies:
- db_connector: Database operations
- datetime: Timestamp logging
"""

import db_connector
from datetime import datetime

def create_users_table():
    """
    Create the users table with appropriate schema for the CSV data.
    If the table already exists, log that information instead of creating it.
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting users table creation process")
        
        # Connect to database
        connection = db_connector.get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # Check if users table already exists
        if db_connector.table_exists('users', connection):
            print(f"[{datetime.now()}] Users table already exists in the database")
            print(f"[{datetime.now()}] Skipping table creation")
            
            # Show existing table structure
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'users' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] Existing users table structure:")
            for row in results:
                nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                default = f" DEFAULT {row[3]}" if row[3] else ""
                print(f"  - {row[0]}: {row[1]} {nullable}{default}")
            
            return
        
        # Create users table if it doesn't exist
        print(f"[{datetime.now()}] Users table does not exist, creating new table")
        
        create_table_query = """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            user_level VARCHAR(50) NOT NULL,
            agency TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        db_connector.execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] Users table created successfully")
        
        # Verify table creation and show structure
        if db_connector.table_exists('users', connection):
            print(f"[{datetime.now()}] Table creation verified")
            
            structure_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'users' 
            ORDER BY ordinal_position
            """
            
            results = db_connector.fetch_data(structure_query, connection=connection)
            print(f"[{datetime.now()}] New users table structure:")
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
    print(f"[{datetime.now()}] Users table creation script started")
    print("=" * 60)
    
    try:
        create_users_table()
        print("=" * 60)
        print(f"[{datetime.now()}] Users table creation script completed successfully")
    except Exception as e:
        print("=" * 60)
        print(f"[{datetime.now()}] Users table creation script failed: {e}")
