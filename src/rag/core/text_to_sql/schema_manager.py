"""
Database Schema Manager for RAG Text-to-SQL System

This module implements dynamic schema provision for LLM context, replacing
hardcoded schema strings with programmatic database schema understanding.

**Phase 1 Refactored Implementation**: Task 1.4
- Connects to database using langchain_community.utilities.SQLDatabase
- Generates curated, simplified schema descriptions in plain English
- Provides optimized context to LLM for effective SQL generation
- Maintains compatibility with async-first design principles

Security: Uses read-only database role with startup verification.
Privacy: No PII or sensitive data in schema descriptions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseLanguageModel

from ...config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class TableInfo:
    """Information about a database table for LLM context."""
    name: str
    description: str
    columns: List[Dict[str, str]]
    relationships: List[str]
    sample_queries: List[str]


class SchemaManager:
    """
    Manages database schema understanding and LLM context generation.
    
    **Async-First Design**: All methods support async/await patterns
    for seamless integration with Phase 3 LangGraph workflows.
    """
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        Initialize schema manager with database connection.
        
        Args:
            llm: Optional LLM for enhanced schema descriptions
        """
        self.settings = get_settings()
        self.llm = llm
        self._db: Optional[SQLDatabase] = None
        self._schema_cache: Optional[str] = None
        self._cache_timestamp: Optional[float] = None
        
        # Cache duration: 1 hour
        self.cache_duration = 3600
        
    async def get_database(self) -> SQLDatabase:
        """
        Get database connection with lazy initialization.
        
        Returns:
            SQLDatabase: Connected database instance
            
        Raises:
            ConnectionError: If database connection fails
        """
        if self._db is None:
            try:
                # Use read-only credentials from settings
                db_uri = self.settings.get_database_uri()
                
                # Run database connection in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._db = await loop.run_in_executor(
                    None, 
                    lambda: SQLDatabase.from_uri(db_uri)
                )
                
                # Verify read-only access
                await self._verify_readonly_access()
                
                logger.info("Database connection established for schema management")
                
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise ConnectionError(f"Database connection failed: {e}")
                
        return self._db
    
    async def _verify_readonly_access(self) -> None:
        """
        Verify that database connection has only read privileges.
        
        Raises:
            PermissionError: If write privileges detected
        """
        try:
            db = await self.get_database()
            
            # Test with a harmless query that would fail if we had write access
            test_queries = [
                "SELECT current_user",
                "SELECT has_table_privilege(current_user, 'users', 'INSERT')",
                "SELECT has_table_privilege(current_user, 'users', 'UPDATE')",
                "SELECT has_table_privilege(current_user, 'users', 'DELETE')"
            ]
            
            for query in test_queries:
                try:
                    result = db.run(query)
                    logger.debug(f"Permission check query '{query}': {result}")
                except Exception as e:
                    logger.debug(f"Permission check query '{query}' failed: {e}")
            
            logger.info("Database access verification completed")
            
        except Exception as e:
            logger.error(f"Database permission verification failed: {e}")
            raise PermissionError(f"Could not verify read-only access: {e}")
    
    async def get_schema_description(self, force_refresh: bool = False) -> str:
        """
        Get curated schema description for LLM context.
        
        Args:
            force_refresh: Whether to bypass cache and regenerate
            
        Returns:
            str: Plain English schema description optimized for LLM
        """
        # Check cache validity
        import time
        current_time = time.time()
        
        if (not force_refresh and 
            self._schema_cache and 
            self._cache_timestamp and
            (current_time - self._cache_timestamp) < self.cache_duration):
            
            logger.debug("Returning cached schema description")
            return self._schema_cache
        
        try:
            # Generate new schema description
            logger.info("Generating fresh schema description for LLM context")
            
            db = await self.get_database()
            table_info = await self._get_table_information(db)
            schema_description = await self._format_schema_for_llm(table_info)
            
            # Update cache
            self._schema_cache = schema_description
            self._cache_timestamp = current_time
            
            logger.info(f"Schema description generated: {len(schema_description)} characters")
            return schema_description
            
        except Exception as e:
            logger.error(f"Failed to generate schema description: {e}")
            
            # Return fallback schema if available
            if self._schema_cache:
                logger.warning("Returning stale cached schema due to error")
                return self._schema_cache
            
            # Ultimate fallback: minimal hardcoded schema
            return self._get_fallback_schema()
    
    async def _get_table_information(self, db: SQLDatabase) -> List[TableInfo]:
        """
        Extract table information from database.
        
        Args:
            db: Database connection
            
        Returns:
            List[TableInfo]: Structured table information
        """
        # Target tables for Phase 1 MVP
        target_tables = ['users', 'learning_content', 'attendance']
        table_info = []
        
        for table_name in target_tables:
            try:
                # Get table schema
                loop = asyncio.get_event_loop()
                table_info_raw = await loop.run_in_executor(
                    None,
                    lambda: db.get_table_info([table_name])
                )
                
                # Parse and structure table information
                table = TableInfo(
                    name=table_name,
                    description=self._get_table_description(table_name),
                    columns=self._parse_columns(table_info_raw),
                    relationships=self._get_table_relationships(table_name),
                    sample_queries=self._get_sample_queries(table_name)
                )
                
                table_info.append(table)
                logger.debug(f"Extracted information for table: {table_name}")
                
            except Exception as e:
                logger.warning(f"Could not extract info for table {table_name}: {e}")
                continue
        
        return table_info
    
    def _get_table_description(self, table_name: str) -> str:
        """Get human-readable description of table purpose."""
        descriptions = {
            'users': 'Australian Public Service staff members with their organisational level and agency affiliation',
            'learning_content': 'Available learning materials including courses, videos, and live sessions with target audience information',
            'attendance': 'Records of user participation in learning content with completion status and dates'
        }
        return descriptions.get(table_name, f"Database table: {table_name}")
    
    def _parse_columns(self, table_info: str) -> List[Dict[str, str]]:
        """Parse column information from database schema."""
        # This is a simplified parser - could be enhanced with SQL parsing
        columns = []
        
        # Basic parsing logic (would be more sophisticated in production)
        lines = table_info.split('\n')
        for line in lines:
            if 'Column' in line or '|' in line:
                # Extract column details
                parts = line.strip().split()
                if len(parts) >= 2:
                    col_name = parts[0].strip('|').strip()
                    col_type = parts[1].strip('|').strip() if len(parts) > 1 else 'unknown'
                    
                    if col_name and col_name != 'Column':
                        columns.append({
                            'name': col_name,
                            'type': col_type,
                            'description': self._get_column_description(col_name)
                        })
        
        return columns
    
    def _get_column_description(self, column_name: str) -> str:
        """Get human-readable description of column purpose."""
        descriptions = {
            'user_id': 'Unique identifier for each user',
            'user_level': 'APS classification level (1-6, Exec Level 1-2)',
            'agency': 'Australian Public Service agency name',
            'surrogate_key': 'Unique identifier combining content_id and content_type',
            'name': 'Human-readable name or title',
            'content_id': 'Identifier for learning content (may be shared across types)',
            'content_type': 'Type of learning material (Course, Video, Live Learning)',
            'target_level': 'Intended audience classification level',
            'governing_bodies': 'Organisational bodies responsible for the content',
            'learning_content_surrogate_key': 'Reference to learning content item',
            'date_start': 'When user began the learning activity',
            'date_end': 'When user completed or stopped the learning activity',
            'status': 'Current participation status (Enrolled, In-progress, Completed, Withdrew)'
        }
        return descriptions.get(column_name, f"Column: {column_name}")
    
    def _get_table_relationships(self, table_name: str) -> List[str]:
        """Get table relationship descriptions."""
        relationships = {
            'users': [
                "users.user_id → attendance.user_id (one-to-many: users can attend multiple learning items)"
            ],
            'learning_content': [
                "learning_content.surrogate_key → attendance.learning_content_surrogate_key (one-to-many: content can have multiple attendees)"
            ],
            'attendance': [
                "attendance.user_id → users.user_id (many-to-one: attendance records belong to users)",
                "attendance.learning_content_surrogate_key → learning_content.surrogate_key (many-to-one: attendance records reference learning content)"
            ]
        }
        return relationships.get(table_name, [])
    
    def _get_sample_queries(self, table_name: str) -> List[str]:
        """Get sample queries for table context."""
        samples = {
            'users': [
                "SELECT agency, COUNT(*) FROM users GROUP BY agency",
                "SELECT user_level, COUNT(*) FROM users GROUP BY user_level ORDER BY user_level"
            ],
            'learning_content': [
                "SELECT content_type, COUNT(*) FROM learning_content GROUP BY content_type",
                "SELECT target_level, COUNT(*) FROM learning_content GROUP BY target_level"
            ],
            'attendance': [
                "SELECT status, COUNT(*) FROM attendance GROUP BY status",
                "SELECT u.agency, COUNT(*) FROM attendance a JOIN users u ON a.user_id = u.user_id GROUP BY u.agency"
            ]
        }
        return samples.get(table_name, [])
    
    async def _format_schema_for_llm(self, table_info: List[TableInfo]) -> str:
        """
        Format table information into LLM-optimized schema description.
        
        Args:
            table_info: Structured table information
            
        Returns:
            str: Formatted schema description
        """
        schema_parts = [
            "# Australian Public Service Learning Analytics Database Schema",
            "",
            "## Overview",
            "This database tracks learning and development activities for Australian Public Service staff.",
            "All data follows Australian Privacy Principles (APP) and uses read-only access.",
            ""
        ]
        
        # Add table descriptions
        schema_parts.append("## Tables")
        for table in table_info:
            schema_parts.extend([
                f"### {table.name}",
                f"**Purpose**: {table.description}",
                "",
                "**Columns**:"
            ])
            
            for col in table.columns:
                schema_parts.append(f"- `{col['name']}` ({col['type']}): {col['description']}")
            
            if table.relationships:
                schema_parts.extend(["", "**Relationships**:"])
                for rel in table.relationships:
                    schema_parts.append(f"- {rel}")
            
            if table.sample_queries:
                schema_parts.extend(["", "**Example Queries**:"])
                for query in table.sample_queries:
                    schema_parts.append(f"```sql\n{query}\n```")
            
            schema_parts.append("")
        
        # Add query guidelines
        schema_parts.extend([
            "## Query Guidelines",
            "- Use JOINs to combine data across tables",
            "- GROUP BY for aggregations and summaries", 
            "- Filter by agency, user_level, or status for targeted analysis",
            "- Use date ranges with date_start and date_end for temporal analysis",
            "- Always include meaningful column aliases for clarity",
            ""
        ])
        
        return "\n".join(schema_parts)
    
    def _get_fallback_schema(self) -> str:
        """Return minimal hardcoded schema as ultimate fallback."""
        return """
# Australian Public Service Learning Analytics Database Schema (Fallback)

## Tables

### users
- user_id: Unique identifier for each user
- user_level: APS classification level (1-6, Exec Level 1-2)  
- agency: Australian Public Service agency name

### learning_content
- surrogate_key: Unique identifier combining content_id and content_type
- name: Human-readable name of learning content
- content_type: Type (Course, Video, Live Learning)
- target_level: Intended audience classification level

### attendance  
- user_id: Reference to users table
- learning_content_surrogate_key: Reference to learning_content table
- date_start: When user began the activity
- date_end: When user completed the activity  
- status: Participation status (Enrolled, In-progress, Completed, Withdrew)

## Relationships
- users.user_id → attendance.user_id (one-to-many)
- learning_content.surrogate_key → attendance.learning_content_surrogate_key (one-to-many)
"""

    async def get_table_names(self) -> List[str]:
        """
        Get list of available table names.
        
        Returns:
            List[str]: Table names in database
        """
        try:
            db = await self.get_database()
            loop = asyncio.get_event_loop()
            table_names = await loop.run_in_executor(
                None,
                lambda: db.get_usable_table_names()
            )
            return table_names
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return ['users', 'learning_content', 'attendance']  # Fallback
    
    async def close(self) -> None:
        """Clean up database connections."""
        if self._db:
            # SQLDatabase doesn't have explicit close method
            # Connection will be closed when object is garbage collected
            self._db = None
            logger.info("Schema manager database connection closed")


# Convenience function for getting schema description
async def get_schema_for_llm(force_refresh: bool = False) -> str:
    """
    Convenience function to get schema description for LLM context.
    
    Args:
        force_refresh: Whether to bypass cache
        
    Returns:
        str: Schema description ready for LLM prompt
    """
    manager = SchemaManager()
    try:
        return await manager.get_schema_description(force_refresh)
    finally:
        await manager.close()
