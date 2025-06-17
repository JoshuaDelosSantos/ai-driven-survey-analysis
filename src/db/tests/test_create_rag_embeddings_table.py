#!/usr/bin/env python3
"""
Test script for create_rag_embeddings_table.py

This script tests the creation of the rag_embeddings table, verifies
the schema structure, and validates pgvector extension functionality.
"""

import os
import sys
import logging
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Add the parent directory to path to import the module
sys.path.append(str(Path(__file__).parent.parent))

from create_rag_embeddings_table import create_rag_embeddings_table, enable_pgvector_extension
import db_connector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pgvector_extension():
    """Test that pgvector extension can be enabled."""
    logger.info("Testing pgvector extension...")
    
    enable_pgvector_extension()
    logger.info("✓ pgvector extension test passed")

def test_table_creation():
    """Test that rag_embeddings table can be created."""
    logger.info("Testing rag_embeddings table creation...")
    
    create_rag_embeddings_table()
    logger.info("✓ Table creation test passed")

def test_table_structure():
    """Test that rag_embeddings table has the correct structure."""
    logger.info("Testing rag_embeddings table structure...")
    
    connection = None
    try:
        connection = db_connector.get_db_connection()
        cursor = connection.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'rag_embeddings'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        assert table_exists, "rag_embeddings table should exist"
        
        # Check required columns
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'rag_embeddings' 
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        column_names = [col[0] for col in columns]
        
        required_columns = [
            'embedding_id', 'response_id', 'field_name', 'chunk_text',
            'chunk_index', 'embedding', 'model_version', 'metadata', 'created_at'
        ]
        
        for required_col in required_columns:
            assert required_col in column_names, f"Column '{required_col}' should exist"
        
        logger.info(f"✓ Table structure test passed - Found columns: {column_names}")
        
    finally:
        if connection:
            db_connector.close_db_connection(connection)

def test_foreign_key_constraint():
    """Test that foreign key constraint to evaluation table exists."""
    logger.info("Testing foreign key constraint...")
    
    connection = None
    try:
        connection = db_connector.get_db_connection()
        cursor = connection.cursor()
        
        # Check foreign key constraint
        cursor.execute("""
            SELECT 
                tc.constraint_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = 'rag_embeddings'
                AND kcu.column_name = 'response_id';
        """)
        
        fk_constraint = cursor.fetchone()
        assert fk_constraint is not None, "Foreign key constraint on response_id should exist"
        assert fk_constraint[2] == 'evaluation', "Foreign key should reference evaluation table"
        assert fk_constraint[3] == 'response_id', "Foreign key should reference response_id column"
        
        logger.info(f"✓ Foreign key constraint test passed - {fk_constraint[0]}")
        
    finally:
        if connection:
            db_connector.close_db_connection(connection)

def test_indexes():
    """Test that required indexes exist."""
    logger.info("Testing indexes...")
    
    connection = None
    try:
        connection = db_connector.get_db_connection()
        cursor = connection.cursor()
        
        # Check for indexes
        cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'rag_embeddings';
        """)
        
        indexes = cursor.fetchall()
        index_names = [idx[0] for idx in indexes]
        
        # Should have at least a primary key index and some performance indexes
        assert len(indexes) > 0, "Table should have indexes"
        
        # Check for vector index (ivfflat)
        vector_index_exists = any('ivfflat' in idx[1].lower() for idx in indexes)
        logger.info(f"Vector index (ivfflat) exists: {vector_index_exists}")
        
        logger.info(f"✓ Indexes test passed - Found {len(indexes)} indexes: {index_names}")
        
    finally:
        if connection:
            db_connector.close_db_connection(connection)

def test_vector_column():
    """Test that vector column has correct type and dimension."""
    logger.info("Testing vector column configuration...")
    
    connection = None
    try:
        connection = db_connector.get_db_connection()
        cursor = connection.cursor()
        
        # Check vector column type
        cursor.execute("""
            SELECT 
                column_name,
                data_type,
                udt_name
            FROM information_schema.columns 
            WHERE table_name = 'rag_embeddings' 
                AND column_name = 'embedding';
        """)
        
        vector_col = cursor.fetchone()
        assert vector_col is not None, "embedding column should exist"
        
        # pgvector columns show up as USER-DEFINED type
        assert vector_col[1] == 'USER-DEFINED' or 'vector' in vector_col[2].lower(), \
            f"embedding column should be vector type, got: {vector_col}"
        
        logger.info(f"✓ Vector column test passed - Type: {vector_col[1]}, UDT: {vector_col[2]}")
        
    finally:
        if connection:
            db_connector.close_db_connection(connection)

# Custom test runner for standalone execution (when not using pytest)
def run_all_tests_standalone():
    """Run all tests manually and return overall result (for standalone execution)."""
    logger.info("Starting rag_embeddings table tests...")
    logger.info("=" * 60)
    
    tests = [
        ("pgVector Extension", test_pgvector_extension),
        ("Table Creation", test_table_creation),
        ("Table Structure", test_table_structure),
        ("Foreign Key Constraint", test_foreign_key_constraint),
        ("Indexes", test_indexes),
        ("Vector Column", test_vector_column),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            test_func()
            results.append((test_name, True))
            logger.info(f"✓ {test_name} PASSED")
        except Exception as e:
            results.append((test_name, False))
            logger.error(f"✗ {test_name} FAILED: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    print("Testing RAG Embeddings Table Creation")
    print("=" * 50)
    
    try:
        success = run_all_tests_standalone()
        
        if success:
            print("\n🎉 All tests passed! RAG embeddings table is properly configured.")
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)
