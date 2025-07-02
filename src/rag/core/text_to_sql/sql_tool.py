"""
SQL Tool Implementation for RAG Text-to-SQL System

This module implements the core SQL generation and execution functionality
using LangChain's SQL toolkit with async methods.

- Async-first design with all I/O operations using async/await
- LangChain SQLDatabaseToolkit integration
- QuerySQLDatabaseTool for SQL execution
- QuerySQLCheckerTool for basic validation
- Simple LangGraph node wrapper for MVP

Security: Read-only database access with SQL injection prevention.
Performance: Async operations for non-blocking execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
    InfoSQLDatabaseTool
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish

from .schema_manager import SchemaManager
from ...config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass 
class SQLResult:
    """Result of SQL query execution."""
    query: str
    result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None
    row_count: Optional[int] = None


class AsyncSQLTool:
    """
    Async SQL tool for Text-to-SQL query processing.
    
    **Async-First Design**: All methods support async/await for
    seamless integration with Phase 3 LangGraph workflows.
    """
    
    def __init__(self, llm: BaseLanguageModel, max_retries: int = 3):
        """
        Initialize SQL tool with LLM and database connection.
        
        Args:
            llm: Language model for SQL generation and checking
            max_retries: Maximum retry attempts for failed queries
        """
        self.llm = llm
        self.max_retries = max_retries
        self.settings = get_settings()
        
        # Initialie components
        self._db: Optional[SQLDatabase] = None
        self._schema_manager: Optional[SchemaManager] = None
        self._query_tool: Optional[QuerySQLDatabaseTool] = None
        self._checker_tool: Optional[QuerySQLCheckerTool] = None
        self._info_tool: Optional[InfoSQLDatabaseTool] = None
        
    async def initialize(self) -> None:
        """
        Initialize database connection and tools.
        
        Raises:
            ConnectionError: If database initialization fails
        """
        try:
            # Initialize schema manager (includes database connection)
            self._schema_manager = SchemaManager(llm=self.llm)
            self._db = await self._schema_manager.get_database()
            
            # Initialize LangChain SQL tools
            loop = asyncio.get_event_loop()
            
            # Create tools in thread pool to avoid blocking
            self._query_tool = await loop.run_in_executor(
                None,
                lambda: QuerySQLDatabaseTool(db=self._db, verbose=True)
            )
            
            self._checker_tool = await loop.run_in_executor(
                None,
                lambda: QuerySQLCheckerTool(db=self._db, llm=self.llm, verbose=True)
            )
            
            self._info_tool = await loop.run_in_executor(
                None,
                lambda: InfoSQLDatabaseTool(db=self._db, verbose=True)
            )
            
            logger.info("SQL tool initialized successfully")
            
        except Exception as e:
            logger.error(f"SQL tool initialization failed: {e}")
            raise ConnectionError(f"Failed to initialize SQL tool: {e}")
    
    async def generate_sql(self, question: str, classification_result: Optional[Any] = None) -> str:
        """
        Generate SQL query from natural language question with Phase 2 enhancements.
        
        Args:
            question: Natural language question
            classification_result: Optional classification result with feedback table guidance
            
        Returns:
            str: Generated SQL query
            
        Raises:
            ValueError: If SQL generation fails
        """
        if not self._schema_manager:
            await self.initialize()
        
        try:
            # Get schema context for LLM
            schema_description = await self._schema_manager.get_schema_description()
            
            # Phase 2 Enhancement: Include feedback table guidance if available
            feedback_guidance = ""
            if (classification_result and 
                hasattr(classification_result, 'feedback_table_suggestion') and 
                classification_result.feedback_table_suggestion):
                
                feedback_guidance = self._create_feedback_table_guidance(
                    classification_result.feedback_table_suggestion,
                    getattr(classification_result, 'feedback_confidence', 0.0)
                )
            
            # Create prompt for SQL generation with enhanced guidance
            prompt = self._create_sql_generation_prompt(question, schema_description, feedback_guidance)
            
            # Generate SQL using LLM
            loop = asyncio.get_event_loop()
            sql_response = await loop.run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            # Extract SQL from response
            sql_query = self._extract_sql_from_response(sql_response.content if hasattr(sql_response, 'content') else str(sql_response))
            
            logger.info(f"Generated SQL for question: {question[:50]}...")
            if feedback_guidance:
                logger.info(f"Applied feedback table guidance: {classification_result.feedback_table_suggestion}")
            logger.debug(f"Generated SQL: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation failed for question '{question}': {e}")
            raise ValueError(f"Could not generate SQL: {e}")
    
    async def validate_sql(self, sql_query: str) -> bool:
        """
        Validate SQL query for safety and correctness.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            bool: True if query is safe and valid
        """
        if not self._checker_tool:
            await self.initialize()
        
        try:
            # Check for dangerous operations
            if not self._is_safe_query(sql_query):
                logger.warning(f"Unsafe SQL query detected: {sql_query}")
                return False
            
            # Use LangChain validator
            loop = asyncio.get_event_loop()
            validation_result = await loop.run_in_executor(
                None,
                lambda: self._checker_tool.run(sql_query)
            )
            
            # Parse validation result
            is_valid = "error" not in validation_result.lower()
            
            if is_valid:
                logger.debug(f"SQL validation passed: {sql_query}")
            else:
                logger.warning(f"SQL validation failed: {validation_result}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"SQL validation error: {e}")
            return False
    
    async def execute_sql(self, sql_query: str) -> SQLResult:
        """
        Execute validated SQL query.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            SQLResult: Query execution result
        """
        if not self._query_tool:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate query first
            is_valid = await self.validate_sql(sql_query)
            if not is_valid:
                return SQLResult(
                    query=sql_query,
                    result=None,
                    execution_time=time.time() - start_time,
                    success=False,
                    error="SQL validation failed"
                )
            
            # Execute query
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._query_tool.run(sql_query)
            )
            
            execution_time = time.time() - start_time
            
            # Parse result to count rows
            row_count = self._count_result_rows(result)
            
            logger.info(f"SQL executed successfully in {execution_time:.3f}s, {row_count} rows")
            
            return SQLResult(
                query=sql_query,
                result=result,
                execution_time=execution_time,
                success=True,
                row_count=row_count
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SQL execution failed: {e}")
            
            return SQLResult(
                query=sql_query,
                result=None,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def process_question(self, question: str) -> SQLResult:
        """
        Process natural language question through complete SQL pipeline.
        
        Args:
            question: Natural language question
            
        Returns:
            SQLResult: Complete processing result
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Processing question (attempt {attempt + 1}): {question}")
                
                # Generate SQL
                sql_query = await self.generate_sql(question)
                
                # Execute SQL
                result = await self.execute_sql(sql_query)
                
                if result.success:
                    logger.info(f"Question processed successfully on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"SQL execution failed on attempt {attempt + 1}: {result.error}")
                    if attempt == self.max_retries - 1:
                        return result  # Return final failed result
                    
                    # Wait before retry
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Question processing error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return SQLResult(
                        query="",
                        result=None,
                        execution_time=0,
                        success=False,
                        error=f"Failed after {self.max_retries} attempts: {e}"
                    )
                
                await asyncio.sleep(1)
    
    def _create_sql_generation_prompt(self, question: str, schema: str, feedback_guidance: str = "") -> str:
        """Create prompt for SQL generation with Phase 2 feedback table guidance."""
        base_prompt = f"""You are an expert SQL query generator for an Australian Public Service learning analytics database.

Given the database schema and a natural language question, generate a valid PostgreSQL query.

# Database Schema
{schema}

# Question
{question}

# Instructions
- Generate a valid PostgreSQL SELECT query only
- Use proper JOINs to combine tables when needed
- Include appropriate WHERE clauses for filtering
- Use meaningful column aliases
- Return only the SQL query, no explanations
- Do not include INSERT, UPDATE, DELETE, or DDL statements

# CRITICAL FEEDBACK TABLE GUIDANCE (Phase 2 Enhancement)
When handling feedback queries, distinguish between:
- CONTENT FEEDBACK: Course/training feedback → Use 'evaluation' table (contains course ratings, satisfaction, instructor feedback)
- SYSTEM FEEDBACK: RAG system feedback → Use 'rag_user_feedback' table (contains search quality, system usability feedback)

NEVER join 'rag_user_feedback' with 'learning_content' - these are separate domains.
For learning content feedback, use: evaluation → learning_content (via learning_content_surrogate_key)

{feedback_guidance}

# SQL Query:"""
        
        return base_prompt
    
    def _create_feedback_table_guidance(self, recommended_table: str, confidence: float) -> str:
        """Create specific guidance for feedback table usage."""
        if recommended_table == "evaluation":
            return f"""
SPECIFIC GUIDANCE FOR THIS QUERY (Confidence: {confidence:.2f}):
- This appears to be CONTENT FEEDBACK about courses/training
- Use the 'evaluation' table which contains:
  * general_feedback: Free-text feedback about courses
  * did_experience_issue_detail: Specific issues mentioned
  * course_application_other: How users will apply learning
  * Satisfaction ratings: positive_learning_experience, effective_use_of_time, relevant_to_work
- Join with 'learning_content' via learning_content_surrogate_key for course context
- Join with 'users' via user_id for demographic analysis if needed
"""
        elif recommended_table == "rag_user_feedback":
            return f"""
SPECIFIC GUIDANCE FOR THIS QUERY (Confidence: {confidence:.2f}):
- This appears to be SYSTEM FEEDBACK about the RAG system itself
- Use the 'rag_user_feedback' table which contains:
  * Feedback about search quality and system responses
  * System usability and technical issues
  * AI response quality ratings
- This table is NOT related to learning content feedback
"""
        else:
            return ""

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["SQL Query:", "Query:", "SQL:"]
        for prefix in prefixes_to_remove:
            if response.upper().startswith(prefix.upper()):
                response = response[len(prefix):].strip()
                break
        
        # Ensure query starts with SELECT or has SELECT somewhere
        response_upper = response.upper()
        if not response_upper.startswith("SELECT"):
            if "SELECT" in response_upper:
                start_idx = response_upper.find("SELECT")
                response = response[start_idx:]
            else:
                # Try to find SQL-like patterns or return as-is for testing
                if any(keyword in response_upper for keyword in ["FROM", "WHERE", "JOIN"]):
                    # Looks like SQL but missing SELECT, add it
                    response = "SELECT " + response
                else:
                    raise ValueError("Generated response does not contain a valid SELECT query")
        
        # Remove trailing semicolon and clean up
        response = response.rstrip(";").strip()
        
        return response
    
    def _is_safe_query(self, sql_query: str) -> bool:
        """Check if SQL query is safe (read-only)."""
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        
        query_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.warning(f"Dangerous keyword '{keyword}' found in query")
                return False
        
        return True
    
    def _count_result_rows(self, result: Any) -> Optional[int]:
        """Count rows in query result."""
        try:
            if isinstance(result, str):
                # Count newlines as approximate row count
                return len(result.split('\n')) - 1  # Subtract header
            elif isinstance(result, list):
                return len(result)
            else:
                return None
        except:
            return None
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._schema_manager:
            await self._schema_manager.close()
        
        # Clear tool references
        self._query_tool = None
        self._checker_tool = None
        self._info_tool = None
        self._db = None
        
        logger.info("SQL tool resources cleaned up")


# LangGraph Node Wrapper (Simple single-node graph for MVP)
class SQLToolNode:
    """
    LangGraph-compatible node wrapper for SQL tool.
    
    This provides a simple single-node LangGraph workflow for Phase 1 MVP.
    Will be expanded into full multi-node workflow in Phase 3.
    """
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize SQL tool node."""
        self.sql_tool = AsyncSQLTool(llm)
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state through SQL tool node.
        
        Args:
            state: LangGraph state dictionary containing 'question'
            
        Returns:
            Dict[str, Any]: Updated state with SQL result
        """
        question = state.get('question', '')
        
        if not question:
            return {
                **state,
                'error': 'No question provided',
                'success': False
            }
        
        try:
            # Process question through SQL pipeline
            result = await self.sql_tool.process_question(question)
            
            return {
                **state,
                'sql_query': result.query,
                'sql_result': result.result,
                'execution_time': result.execution_time,
                'success': result.success,
                'error': result.error,
                'row_count': result.row_count
            }
            
        except Exception as e:
            logger.error(f"SQL tool node error: {e}")
            return {
                **state,
                'error': f"SQL processing failed: {e}",
                'success': False
            }
    
    async def initialize(self) -> None:
        """Initialize SQL tool."""
        await self.sql_tool.initialize()
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.sql_tool.close()


# Convenience function for direct usage
async def query_database(question: str, llm: BaseLanguageModel) -> SQLResult:
    """
    Convenience function to query database with natural language.
    
    Args:
        question: Natural language question
        llm: Language model for SQL generation
        
    Returns:
        SQLResult: Query result
    """
    sql_tool = AsyncSQLTool(llm)
    try:
        await sql_tool.initialize()
        return await sql_tool.process_question(question)
    finally:
        await sql_tool.close()
