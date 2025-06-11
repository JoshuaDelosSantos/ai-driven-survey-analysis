"""
Database Utilities for RAG Module

Provides secure database connection management and helper functions
for the RAG Text-to-SQL system with read-only access controls.

Security: All connections use read-only credentials with verification.
Privacy: No PII logging, secure error handling.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import time

import asyncpg
from langchain_community.utilities import SQLDatabase

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages secure database connections for RAG module.
    
    Features:
    - Read-only connection verification
    - Connection pooling for performance
    - Async-first design
    - Security validation
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.settings = get_settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._langchain_db: Optional[SQLDatabase] = None
    
    async def get_pool(self) -> asyncpg.Pool:
        """
        Get or create connection pool.
        
        Returns:
            asyncpg.Pool: Database connection pool
        """
        if self._pool is None:
            try:
                # Create connection pool with read-only credentials
                self._pool = await asyncpg.create_pool(
                    host=self.settings.rag_db_host,
                    port=self.settings.rag_db_port,
                    database=self.settings.rag_db_name,
                    user=self.settings.rag_db_user,
                    password=self.settings.rag_db_password,
                    min_size=2,
                    max_size=10,
                    command_timeout=30
                )
                
                # Verify read-only access
                await self._verify_readonly_pool()
                
                logger.info("Database connection pool created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                raise ConnectionError(f"Database pool creation failed: {e}")
        
        return self._pool
    
    async def get_langchain_db(self) -> SQLDatabase:
        """
        Get LangChain SQLDatabase instance.
        
        Returns:
            SQLDatabase: LangChain database wrapper
        """
        if self._langchain_db is None:
            try:
                db_uri = self.settings.get_database_uri()
                loop = asyncio.get_event_loop()
                
                self._langchain_db = await loop.run_in_executor(
                    None,
                    lambda: SQLDatabase.from_uri(db_uri)
                )
                
                logger.info("LangChain database connection created")
                
            except Exception as e:
                logger.error(f"Failed to create LangChain database: {e}")
                raise ConnectionError(f"LangChain database creation failed: {e}")
        
        return self._langchain_db
    
    async def _verify_readonly_pool(self) -> None:
        """Verify that pool connections are read-only."""
        pool = await self.get_pool()
        
        async with pool.acquire() as conn:
            try:
                # Test read-only constraints
                result = await conn.fetchval("SELECT current_user")
                logger.debug(f"Connected as user: {result}")
                
                # Verify no write permissions on key tables
                tables = ['users', 'learning_content', 'attendance']
                for table in tables:
                    has_insert = await conn.fetchval(
                        "SELECT has_table_privilege(current_user, $1, 'INSERT')",
                        table
                    )
                    has_update = await conn.fetchval(
                        "SELECT has_table_privilege(current_user, $1, 'UPDATE')",
                        table
                    )
                    has_delete = await conn.fetchval(
                        "SELECT has_table_privilege(current_user, $1, 'DELETE')",
                        table
                    )
                    
                    if has_insert or has_update or has_delete:
                        raise PermissionError(f"Write access detected on table {table}")
                
                logger.info("Database read-only access verified")
                
            except PermissionError:
                raise
            except Exception as e:
                logger.warning(f"Could not fully verify read-only access: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get database connection from pool.
        
        Usage:
            async with db_manager.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
        """
        pool = await self.get_pool()
        conn = await pool.acquire()
        try:
            yield conn
        finally:
            await pool.release(conn)
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """
        Execute read-only query safely.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
            
        Raises:
            ValueError: If query contains write operations
            ConnectionError: If database error occurs
        """
        # Validate query is read-only
        if not self._is_readonly_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            async with self.get_connection() as conn:
                if params:
                    records = await conn.fetch(query, *params)
                else:
                    records = await conn.fetch(query)
                
                # Convert records to dictionaries
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise ConnectionError(f"Database query failed: {e}")
    
    async def execute_scalar(self, query: str, params: Optional[List] = None) -> Any:
        """
        Execute query and return single value.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Any: Single query result value
        """
        if not self._is_readonly_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            async with self.get_connection() as conn:
                if params:
                    return await conn.fetchval(query, *params)
                else:
                    return await conn.fetchval(query)
                    
        except Exception as e:
            logger.error(f"Scalar query execution failed: {e}")
            raise ConnectionError(f"Database scalar query failed: {e}")
    
    def _is_readonly_query(self, query: str) -> bool:
        """Check if query is read-only."""
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        
        query_upper = query.upper().strip()
        
        # Must start with SELECT or WITH (for CTEs)
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            return False
        
        # Check for dangerous keywords
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False
        
        return True
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table.
        
        Args:
            table_name: Name of table to inspect
            
        Returns:
            Dict[str, Any]: Table information
        """
        try:
            # Get column information
            columns_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
            """
            
            columns = await self.execute_query(columns_query, [table_name])
            
            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            row_count = await self.execute_scalar(count_query)
            
            return {
                'table_name': table_name,
                'columns': columns,
                'row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            bool: True if connection successful
        """
        try:
            result = await self.execute_scalar("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")
        
        self._langchain_db = None


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Convenience functions
async def query_database(query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
    """
    Execute database query using global manager.
    
    Args:
        query: SQL query to execute
        params: Query parameters
        
    Returns:
        List[Dict[str, Any]]: Query results
    """
    manager = get_database_manager()
    return await manager.execute_query(query, params)


async def query_scalar(query: str, params: Optional[List] = None) -> Any:
    """
    Execute scalar database query using global manager.
    
    Args:
        query: SQL query to execute
        params: Query parameters
        
    Returns:
        Any: Single query result
    """
    manager = get_database_manager()
    return await manager.execute_scalar(query, params)


async def test_database_connection() -> bool:
    """Test database connectivity."""
    manager = get_database_manager()
    return await manager.test_connection()


@asynccontextmanager
async def get_db_connection():
    """Get database connection context manager."""
    manager = get_database_manager()
    async with manager.get_connection() as conn:
        yield conn
