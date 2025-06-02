"""
PostgreSQL Database Connector for Sentiment Analysis Project

This module provides a database connectivity layer for sentiment analysis operations.

Features:
- Secure database connections using environment variables (.env file support)
- Query execution for SELECT, INSERT, UPDATE, DELETE operations
- Batch insert operations for efficient data loading
- Parameterised queries to prevent SQL injection
- Automatic connection and cursor management
- Comprehensive error handling and logging
- Transaction management with rollback support

Environment Variables Required:
- POSTGRES_DB: Database name
- POSTGRES_USER: Database username
- POSTGRES_PASSWORD: Database password

Functions:
- get_db_connection(): Establish database connection
- close_db_connection(): Safely close connections and cursors
- fetch_data(): Execute SELECT queries and return results
- execute_query(): Execute INSERT/UPDATE/DELETE queries
- batch_insert_data(): Efficiently insert multiple rows
- test_database_operations(): Comprehensive test suite

Usage Example:
    from db_connector import get_db_connection, fetch_data
    
    # Get connection
    conn = get_db_connection()
    
    # Fetch data
    results = fetch_data("SELECT * FROM sentiment_data WHERE score > %s", (0.5,), conn)
    
    # Close connection
    close_db_connection(conn)

Dependencies:
- psycopg2-binary: PostgreSQL adapter
- python-dotenv: Environment variable loading
- datetime: Timestamp logging
- os: Environment variable access

Author: Joshua Delos Santos
Created: June 2025
"""

import psycopg2
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection(): 
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
        Connection object or None if connection fails
        
    Raises:
        Exception: If connection to the database fails
    """
    try:
        print(f"[{datetime.now()}] Attempting to connect to PostgreSQL database")
        
        # Get environment variables (they should now be loaded from .env)
        db_name = os.getenv("POSTGRES_DB")
        db_user = os.getenv("POSTGRES_USER") 
        db_password = os.getenv("POSTGRES_PASSWORD")
        
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host="localhost",
            port="5432"
        )
        print(f"[{datetime.now()}] Successfully connected to PostgreSQL database")
        return conn
    except psycopg2.OperationalError as e:
        print(f"[{datetime.now()}] ERROR: Database connection error: {e}")
        raise
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Unexpected error when connecting to database: {e}")
        raise


def close_db_connection(connection, cursor=None):
    """
    Close the database connection and cursor if they are open.
    """
    if cursor:
        cursor.close()
    if connection:
        connection.close()
    print(f"[{datetime.now()}] Database connection closed")


def fetch_data(query, params=None, connection=None):
    """
    Execute a SELECT query and return the results.
    
    Args:
        query (str): SQL SELECT query to execute
        params (tuple, optional): Parameters for parameterized query
        connection (psycopg2.connection, optional): Database connection. If None, creates a new one
        
    Returns:
        list: List of tuples containing query results
        
    Raises:
        Exception: If query execution fails
    """
    conn_created = False
    cursor = None
    
    try:
        # Use provided connection or create a new one
        if connection is None:
            connection = get_db_connection()
            conn_created = True
            
        cursor = connection.cursor()
        
        print(f"[{datetime.now()}] Executing SELECT query: {query[:100]}...")
        cursor.execute(query, params)
        
        results = cursor.fetchall()
        print(f"[{datetime.now()}] Query returned {len(results)} rows")
        
        return results
        
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to fetch data: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn_created and connection:
            connection.close()


def execute_query(query, params=None, connection=None):
    """
    Execute INSERT, UPDATE, or DELETE queries.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for parameterized query
        connection (psycopg2.connection, optional): Database connection. If None, creates a new one
        
    Returns:
        int: Number of rows affected
        
    Raises:
        Exception: If query execution fails
    """
    conn_created = False
    cursor = None
    
    try:
        # Use provided connection or create a new one
        if connection is None:
            connection = get_db_connection()
            conn_created = True
            
        cursor = connection.cursor()
        
        print(f"[{datetime.now()}] Executing query: {query[:100]}...")
        cursor.execute(query, params)
        
        # Get number of affected rows
        rows_affected = cursor.rowcount
        
        # Commit the transaction
        connection.commit()
        print(f"[{datetime.now()}] Query executed successfully. Rows affected: {rows_affected}")
        
        return rows_affected
        
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to execute query: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn_created and connection:
            connection.close()


def batch_insert_data(query, data_list, connection=None):
    """
    Efficiently insert multiple rows of data using executemany.
    
    Args:
        query (str): INSERT query with placeholders (e.g., "INSERT INTO table (col1, col2) VALUES (%s, %s)")
        data_list (list): List of tuples containing data to insert
        connection (psycopg2.connection, optional): Database connection. If None, creates a new one
        
    Returns:
        int: Number of rows inserted
        
    Raises:
        Exception: If batch insert fails
    """
    conn_created = False
    cursor = None
    
    try:
        # Use provided connection or create a new one
        if connection is None:
            connection = get_db_connection()
            conn_created = True
            
        cursor = connection.cursor()
        
        print(f"[{datetime.now()}] Executing batch insert: {len(data_list)} rows")
        cursor.executemany(query, data_list)
        
        # Get number of affected rows
        rows_affected = cursor.rowcount
        
        # Commit the transaction
        connection.commit()
        print(f"[{datetime.now()}] Batch insert completed successfully. Rows inserted: {rows_affected}")
        
        return rows_affected
        
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to execute batch insert: {e}")
        if connection:
            connection.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn_created and connection:
            connection.close()


def test_database_operations():
    """
    Comprehensive test function that:
    1. Connects to database
    2. Creates a test table
    3. Inserts test data
    4. Fetches and validates data
    5. Cleans up (drops test table)
    6. Closes connection
    """
    connection = None
    
    try:
        print(f"[{datetime.now()}] Starting database operations test")
        
        # 1. Connect to database
        connection = get_db_connection()
        print(f"[{datetime.now()}] Database connection established")
        
        # 2. Create test table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS test_sentiment_data (
            id SERIAL PRIMARY KEY,
            text_content TEXT NOT NULL,
            sentiment_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        execute_query(create_table_query, connection=connection)
        print(f"[{datetime.now()}] Test table created")
        
        # 3. Insert test data using batch insert
        test_data = [
            ("This is a positive review", 0.8),
            ("This product is terrible", -0.6),
            ("Neutral opinion about the service", 0.0),
            ("Amazing experience, highly recommend!", 0.9),
            ("Could be better", -0.2)
        ]
        
        insert_query = """
        INSERT INTO test_sentiment_data (text_content, sentiment_score) 
        VALUES (%s, %s)
        """
        rows_inserted = batch_insert_data(insert_query, test_data, connection=connection)
        print(f"[{datetime.now()}] Test data inserted: {rows_inserted} rows")
        
        # 4. Fetch and validate data
        select_query = "SELECT id, text_content, sentiment_score FROM test_sentiment_data ORDER BY id"
        results = fetch_data(select_query, connection=connection)
        print(f"[{datetime.now()}] Data fetched successfully: {len(results)} rows")
        
        # Display fetched data
        for row in results:
            print(f"[{datetime.now()}]   ID: {row[0]}, Text: '{row[1][:30]}...', Score: {row[2]}")
        
        # 5. Test single insert
        single_insert_query = """
        INSERT INTO test_sentiment_data (text_content, sentiment_score) 
        VALUES (%s, %s)
        """
        execute_query(single_insert_query, ("Single insert test", 0.5), connection=connection)
        print(f"[{datetime.now()}] Single insert test completed")
        
        # 6. Test parameterized select
        param_query = "SELECT COUNT(*) FROM test_sentiment_data WHERE sentiment_score > %s"
        positive_count = fetch_data(param_query, (0.0,), connection=connection)
        print(f"[{datetime.now()}] Parameterized query test: {positive_count[0][0]} positive sentiments")
        
        # 7. Clean up - drop test table
        drop_table_query = "DROP TABLE IF EXISTS test_sentiment_data"
        execute_query(drop_table_query, connection=connection)
        print(f"[{datetime.now()}] Test table cleaned up")
        
        print(f"[{datetime.now()}] All database operations test completed successfully!")
        
    except Exception as e:
        print(f"[{datetime.now()}] Test failed: {e}")
        raise
    finally:
        # 8. Close connection
        if connection:
            close_db_connection(connection)
            print(f"[{datetime.now()}] Database connection closed")


if __name__ == "__main__":
    print(f"[{datetime.now()}] Database connector test started")
    print("=" * 60)
    
    # Run comprehensive database operations test
    test_database_operations()
    
    print("=" * 60)
    print(f"[{datetime.now()}] Database connector test completed")

