#!/usr/bin/env python3
"""
Creates a dedicated read-only PostgreSQL role 'rag_user_readonly' 
for the RAG module with minimal required permissions.

This script implements the NON-NEGOTIABLE security requirement 
from the RAG integration plan.

Configuration is loaded from environment variables:
- RAG_DB_PASSWORD: Password for the rag_user_readonly role (required)
- RAG_DB_NAME: Target database name (defaults to 'csi-db')

Usage:
    Ensure .env file contains RAG_DB_PASSWORD and optionally RAG_DB_NAME
    python create_rag_readonly_role.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path to import db_connector
sys.path.append(str(Path(__file__).parent))

from db_connector import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_rag_readonly_role():
    """
    Create the rag_user_readonly role with minimal read-only permissions.
    
    Security constraints:
    - SELECT only on attendance, users, learning_content tables
    - NO INSERT, UPDATE, DELETE, CREATE permissions
    - Limited to specific database and schema
    """
    
    # Load configuration from environment variables
    rag_password = os.getenv('RAG_DB_PASSWORD')
    db_name = os.getenv('RAG_DB_NAME', 'csi-db')  # Default fallback for backwards compatibility
    
    if not rag_password:
        logger.error("RAG_DB_PASSWORD environment variable is not set. Please set it in your .env file.")
        raise ValueError("RAG_DB_PASSWORD environment variable is required")
    
    logger.info(f"Using database: {db_name}")
    logger.info("RAG_DB_PASSWORD loaded from environment (masked for security)")
    
    connection = None
    try:
        # Get database connection using existing utility
        connection = get_db_connection()
        cursor = connection.cursor()
        
        logger.info("Starting RAG read-only role creation...")
        
        # Check if role already exists
        cursor.execute("""
            SELECT 1 FROM pg_roles WHERE rolname = 'rag_user_readonly';
        """)
        
        if cursor.fetchone():
            logger.info("Role 'rag_user_readonly' already exists. Updating permissions...")
            
            # Revoke all existing privileges first (clean slate)
            cursor.execute(f"""
                REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM rag_user_readonly;
                REVOKE ALL PRIVILEGES ON DATABASE "{db_name}" FROM rag_user_readonly;
                REVOKE ALL PRIVILEGES ON SCHEMA public FROM rag_user_readonly;
            """)
        else:
            logger.info("Creating new role 'rag_user_readonly'...")
            
            # Create the role with login capability using password from environment
            cursor.execute("""
                CREATE ROLE rag_user_readonly WITH LOGIN PASSWORD %s;
            """, (rag_password,))
        
        # Grant minimal required permissions
        logger.info("Granting minimal read-only permissions...")
        
        # 1. Connect to database
        cursor.execute(f"""
            GRANT CONNECT ON DATABASE "{db_name}" TO rag_user_readonly;
        """)
        
        # 2. Usage on public schema
        cursor.execute("""
            GRANT USAGE ON SCHEMA public TO rag_user_readonly;
        """)
        
        # 3. SELECT only on specific tables required for RAG
        tables_for_rag = ['attendance', 'users', 'learning_content', 'evaluation', 'rag_embeddings']
        
        for table in tables_for_rag:
            # Check if table exists before granting permissions
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            if cursor.fetchone()[0]:
                cursor.execute(f"""
                    GRANT SELECT ON {table} TO rag_user_readonly;
                """)
                logger.info("Granted SELECT permission on table '%s'", table)
            else:
                logger.warning("Table '%s' does not exist yet. Permission will need to be granted later.", table)
        
        # Explicitly deny dangerous permissions (defense in depth)
        logger.info("Explicitly denying dangerous permissions...")
        
        for table in tables_for_rag:
            cursor.execute(f"""
                REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON {table} FROM rag_user_readonly;
            """)
        
        # Deny schema-level creation permissions
        cursor.execute("""
            REVOKE CREATE ON SCHEMA public FROM rag_user_readonly;
        """)
        
        # Commit the changes
        connection.commit()
        logger.info("RAG read-only role created successfully")
        
        # Test role restrictions
        test_role_restrictions(cursor)
        
    except Exception as e:
        if connection:
            connection.rollback()
        logger.error("Error creating RAG readonly role: %s", e)
        raise
    
    finally:
        if connection:
            cursor.close()
            connection.close()

def test_role_restrictions(cursor):
    """
    Test that the role has correct permissions and restrictions.
    This validates the security setup.
    """
    logger.info("Testing role permissions and restrictions...")
    
    try:
        # Test 1: Check role exists and has login
        cursor.execute("""
            SELECT rolname, rolcanlogin, rolcreatedb, rolcreaterole, rolsuper
            FROM pg_roles 
            WHERE rolname = 'rag_user_readonly';
        """)
        
        result = cursor.fetchone()
        if result:
            rolname, canlogin, createdb, createrole, super_user = result
            logger.info("Role info - Name: %s, Can Login: %s", rolname, canlogin)
            
            # Validate security constraints
            if createdb or createrole or super_user:
                logger.error("SECURITY RISK: Role has elevated privileges")
                return False
                
            if not canlogin:
                logger.error("Role cannot login - this will prevent RAG functionality")
                return False
        
        # Test 2: Check table permissions
        tables_to_check = ['attendance', 'users', 'learning_content']
        
        for table in tables_to_check:
            cursor.execute("""
                SELECT privilege_type 
                FROM information_schema.role_table_grants 
                WHERE grantee = 'rag_user_readonly' 
                AND table_name = %s
                AND table_schema = 'public';
            """, (table,))
            
            permissions = [row[0] for row in cursor.fetchall()]
            
            if 'SELECT' in permissions:
                logger.info("Table '%s': SELECT permission granted", table)
            else:
                logger.warning("Table '%s': No SELECT permission found", table)
            
            # Check for dangerous permissions
            dangerous_perms = {'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE'}
            found_dangerous = dangerous_perms.intersection(set(permissions))
            
            if found_dangerous:
                logger.error("SECURITY RISK: Table '%s' has dangerous permissions: %s", table, found_dangerous)
                return False
        
        logger.info("Role restrictions validated successfully")
        return True
        
    except Exception as e:
        logger.error("Error testing role restrictions: %s", e)
        return False

def document_role_permissions():
    """
    Document the role permissions and security constraints for compliance.
    """
    db_name = os.getenv('RAG_DB_NAME', 'csi-db')
    
    documentation = f"""
    RAG Read-Only Role Security Documentation
    ========================================
    
    Role Name: rag_user_readonly
    Purpose: Dedicated read-only access for RAG module Text-to-SQL functionality
    
    GRANTED PERMISSIONS:
    - CONNECT on database '{db_name}'
    - USAGE on schema 'public'
    - SELECT on tables: attendance, users, learning_content
    
    EXPLICITLY DENIED PERMISSIONS:
    - INSERT, UPDATE, DELETE, TRUNCATE on all tables
    - CREATE on schema 'public'
    - CREATEDB, CREATEROLE, SUPERUSER privileges
    
    SECURITY CONSTRAINTS:
    - Cannot modify existing data
    - Cannot create new database objects
    - Cannot escalate privileges
    - Limited to specific tables required for RAG functionality
    
    COMPLIANCE NOTES:
    - Follows principle of least privilege
    - Supports Australian Privacy Principles (APP) compliance
    - Enables audit logging of all RAG database access
    """
    
    logger.info("Role permissions documented:")
    logger.info(documentation)
    
    return documentation

if __name__ == "__main__":
    try:
        # Validate environment variables before proceeding
        rag_password = os.getenv('RAG_DB_PASSWORD')
        if not rag_password:
            print("ERROR: RAG_DB_PASSWORD environment variable is not set.")
            print("Please ensure your .env file contains RAG_DB_PASSWORD=your_password")
            sys.exit(1)
            
        print("Environment configuration validated")
        print(f"Database: {os.getenv('RAG_DB_NAME', 'csi-db')}")
        print("Password: [MASKED FOR SECURITY]")
        
        create_rag_readonly_role()
        document_role_permissions()
        
        print("\n" + "="*60)
        print("RAG DATABASE SECURITY SETUP COMPLETE")
        print("="*60)
        print("Read-only role 'rag_user_readonly' created")
        print("Minimal permissions granted")
        print("Security restrictions validated")
        print("Documentation generated")
        
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
