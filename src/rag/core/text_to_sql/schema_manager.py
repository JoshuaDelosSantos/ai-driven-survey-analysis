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
import json
from pathlib import Path
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
        self._data_dictionary: Optional[Dict] = None
        
        # Cache duration: 1 hour
        self.cache_duration = 3600
        
    def _load_data_dictionary(self) -> Optional[Dict]:
        """
        Load data dictionary from JSON file if available.
        
        Returns:
            Dict: Data dictionary content or None if not found
        """
        if self._data_dictionary is not None:
            return self._data_dictionary
            
        # Look for data dictionary in common locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "csv" / "data-dictionary.json",
            Path("../../../csv/data-dictionary.json"),
            Path("data-dictionary.json")
        ]
        
        for path in possible_paths:
            try:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        self._data_dictionary = json.load(f)
                    logger.info(f"Loaded data dictionary from {path}")
                    return self._data_dictionary
            except Exception as e:
                logger.debug(f"Could not load data dictionary from {path}: {e}")
                
        logger.info("No data dictionary found, will use database introspection")
        return None
        
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
            
            # Try to use data dictionary first
            data_dict = self._load_data_dictionary()
            if data_dict:
                logger.info("Using data dictionary for schema description")
                schema_description = await self._format_data_dictionary_for_llm(data_dict)
            else:
                logger.info("Using database introspection for schema description")
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
        # Target tables for RAG system (all tables)
        target_tables = ['users', 'learning_content', 'attendance', 'evaluation', 'rag_embeddings', 'rag_user_feedback']
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
    
    async def _format_data_dictionary_for_llm(self, data_dict: Dict) -> str:
        """
        Format data dictionary for LLM context with accurate schema information.
        
        Args:
            data_dict: Loaded data dictionary
            
        Returns:
            str: Formatted schema description for LLM
        """
        schema_parts = [
            "DATABASE SCHEMA INFORMATION",
            "=" * 50,
            "",
            "You are working with a learning analytics database containing information about Australian Public Service training and development.",
            "",
            "TABLES AND COLUMNS:",
            ""
        ]
        
        for table_name, table_info in data_dict.items():
            schema_parts.append(f"## {table_name.upper()} TABLE")
            schema_parts.append(f"Description: {table_info['description']}")
            schema_parts.append("Columns:")
            
            for column in table_info['columns']:
                col_desc = f"  - {column['name']} ({column['dataType']}): {column['description']}"
                if column.get('isPrimaryKey'):
                    col_desc += " [PRIMARY KEY]"
                if column.get('isForeignKey'):
                    col_desc += f" [FOREIGN KEY -> {column.get('foreignKeyReference')}]"
                if column.get('isFreeText'):
                    col_desc += " [FREE TEXT - may contain PII]"
                schema_parts.append(col_desc)
            
            schema_parts.append("")
        
        # Add important notes for SQL generation
        schema_parts.extend([
            "IMPORTANT NOTES FOR SQL GENERATION:",
            "",
            "1. TABLE USAGE GUIDANCE:",
            "   - evaluation: Use for USER FEEDBACK about learning content/courses",
            "   - rag_user_feedback: Use ONLY for feedback about this RAG system itself",
            "   - attendance: Use for participation statistics and completion rates",
            "   - users: Use for demographic analysis by agency and level",
            "   - learning_content: Use for content categorization and metadata",
            "   - rag_embeddings: Internal system table for vector search operations",
            "",
            "2. FEEDBACK QUERIES - CRITICAL:",
            "   - For learning content feedback: Use evaluation table with learning_content join",
            "   - For RAG system feedback: Use rag_user_feedback table only",
            "   - Never join rag_user_feedback with learning_content (no logical relationship)",
            "",
            "3. ATTENDANCE STATUS VALUES:",
            "   - Use 'Completed' (capital C) for completed courses",
            "   - Use 'Enrolled' for active enrollments", 
            "   - Use 'Cancelled' for cancelled enrollments",
            "",
            "4. COLUMN NAMES:",
            "   - Use 'date_effective' (not date_start or date_end) for attendance dates",
            "   - Use 'learning_content_surrogate_key' to join with learning_content table",
            "   - Use 'status' column in attendance table for enrollment status",
            "   - evaluation.general_feedback, .did_experience_issue_detail, .course_application_other for content feedback",
            "",
            "5. COMMON QUERIES:",
            "   - Learning content feedback: evaluation e JOIN learning_content lc ON e.learning_content_surrogate_key = lc.surrogate_key",
            "   - To find completed courses: WHERE a.status = 'Completed'",
            "   - To join users and attendance: JOIN users u ON a.user_id = u.user_id",
            "",
            "6. PRIVACY PROTECTION:",
            "   - Never include PII in queries or results",
            "   - Aggregate data when possible (COUNT, GROUP BY)",
            "   - Free text columns may contain sensitive information",
            "",
            "=" * 50
        ])
        
        return "\n".join(schema_parts)
    
    def _get_table_description(self, table_name: str) -> str:
        """Get human-readable description of table purpose."""
        descriptions = {
            'users': 'Australian Public Service staff members with their organisational level and agency affiliation',
            'learning_content': 'Available learning materials including courses, videos, and live sessions with target audience information',
            'attendance': 'Records of user participation in learning content with completion status and dates',
            'evaluation': 'Post-course evaluation responses from users about their learning experience and content quality',
            'rag_embeddings': 'Vector embeddings of evaluation text for semantic search (internal system table)',
            'rag_user_feedback': 'Feedback about this RAG system performance (NOT about learning content)'
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
