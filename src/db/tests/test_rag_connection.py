#!/usr/bin/env python3
"""
Test RAG read-only database connection

This script verifies that the RAG read-only role can connect and perform
SELECT operations but cannot perform write operations.
"""

import os
import sys
import logging
import psycopg2
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rag_connection():
    """Test the RAG read-only database connection and permissions."""
    
    connection = None
    cursor = None
    
    try:
        # Connection parameters for RAG read-only role
        connection_params = {
            'host': os.getenv('RAG_DB_HOST'),
            'port': os.getenv('RAG_DB_PORT'),
            'database': os.getenv('RAG_DB_NAME'),
            'user': os.getenv('RAG_DB_USER'),
            'password': os.getenv('RAG_DB_PASSWORD')
        }
        
        logger.info("Testing RAG read-only database connection...")
        logger.info(f"Connecting to: {connection_params['user']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")
        
        # Test connection
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()
        
        logger.info("Connection successful")
        assert connection is not None, "Database connection should be established"
        
        # Test 1: SELECT operations (should work)
        logger.info("Testing SELECT operations...")
        
        test_queries = [
            "SELECT COUNT(*) FROM users;",
            "SELECT COUNT(*) FROM learning_content;", 
            "SELECT COUNT(*) FROM attendance;"
        ]
        
        select_successes = 0
        for query in test_queries:
            try:
                cursor.execute(query)
                result = cursor.fetchone()
                logger.info("%s -> %s records", query, result[0])
                assert result[0] >= 0, f"Query {query} should return a valid count"
                select_successes += 1
            except Exception as e:
                logger.error("%s failed: %s", query, e)
                
        assert select_successes == len(test_queries), f"All {len(test_queries)} SELECT queries should succeed"
        
        # Test 2: Write operations (should fail)
        logger.info("Testing write operations (these should fail)...")
        
        write_tests = [
            "INSERT INTO users (user_id, user_level, agency) VALUES ('test', 'Level 1', 'Test Agency');",
            "UPDATE users SET user_level = 'Level 2' WHERE user_id = 'test';",
            "DELETE FROM users WHERE user_id = 'test';",
            "CREATE TABLE test_table (id INT);"
        ]
        
        write_failures = 0
        for query in write_tests:
            try:
                cursor.execute(query)
                connection.commit()
                logger.error("SECURITY RISK: %s succeeded (should have failed)", query)
                assert False, f"Write operation should be blocked: {query}"
            except psycopg2.Error as e:
                logger.info("%s correctly blocked: %s", query, str(e).strip())
                connection.rollback()
                write_failures += 1
                
        assert write_failures == len(write_tests), f"All {len(write_tests)} write operations should be blocked"
        
        # Test 3: Join operations (complex reads should work)
        logger.info("Testing complex SELECT with JOINs...")
        
        complex_query = """
        SELECT u.agency, lc.content_type, COUNT(*) as attendance_count
        FROM attendance a
        JOIN users u ON a.user_id = u.user_id
        JOIN learning_content lc ON a.learning_content_surrogate_key = lc.surrogate_key
        GROUP BY u.agency, lc.content_type
        LIMIT 5;
        """
        
        cursor.execute(complex_query)
        results = cursor.fetchall()
        logger.info("Complex JOIN query succeeded, returned %s rows", len(results))
        for row in results:
            logger.info("   Agency: %s, Content Type: %s, Count: %s", row[0], row[1], row[2])
        
        assert len(results) >= 0, "Complex JOIN query should execute successfully"
        logger.info("RAG connection test completed successfully")
        
    except Exception as e:
        logger.error("Connection test failed: %s", e)
        raise  # Re-raise the exception to fail the test
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    print("Testing RAG Read-Only Database Connection")
    print("=" * 50)
    
    success = test_rag_connection()
    
    if success:
        print("\nAll tests passed! RAG database connection is properly configured.")
    else:
        print("\nConnection test failed. Please check configuration.")
        sys.exit(1)
