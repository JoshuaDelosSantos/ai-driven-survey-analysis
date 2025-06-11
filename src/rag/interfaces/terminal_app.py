"""
Terminal Application Interface for RAG Text-to-SQL System

Implements the Phase 1 MVP terminal application with async support
for natural language to SQL query processing.

- Async event loop with asyncio.run() 
- Simple input loop for natural language queries
- Integration with LangGraph SQL workflow
- Basic error handling and result formatting
- Target query types for attendance and course statistics

Security: Read-only database access, no PII exposure in terminal.
Performance: Non-blocking async operations.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any
import time
import uuid

from ..core.text_to_sql.sql_tool import AsyncSQLTool
from ..utils.llm_utils import get_llm
from ..utils.logging_utils import get_logger
from ..config.settings import get_settings


logger = get_logger(__name__)


class TerminalApp:
    """
    Terminal application for RAG Text-to-SQL system.
    
    **Async-First Design**: All operations use async/await patterns
    for seamless integration with Phase 3 LangGraph workflows.
    """
    
    def __init__(self):
        """Initialize terminal application."""
        self.settings = get_settings()
        self.sql_tool: Optional[AsyncSQLTool] = None
        self.session_id = str(uuid.uuid4())[:8]
        
        # Example queries for user guidance
        self.example_queries = [
            "How many users completed courses in each agency?",
            "Show attendance status breakdown by user level",
            "Which courses have the highest enrollment?",
            "What are the completion rates by content type?",
            "How many Level 6 users are enrolled in courses?",
            "Show me attendance statistics for Executive Level staff"
        ]
    
    async def initialize(self) -> None:
        """
        Initialize SQL tool and verify system readiness.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info(f"Initializing terminal application (session: {self.session_id})")
            
            # Get configured LLM
            llm = get_llm()
            
            # Initialize SQL tool
            self.sql_tool = AsyncSQLTool(llm)
            await self.sql_tool.initialize()
            
            logger.info("Terminal application initialized successfully")
            
        except Exception as e:
            logger.error(f"Terminal application initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize application: {e}")
    
    async def run(self) -> None:
        """
        Run the main terminal application loop.
        
        This is the primary entry point that runs within an asyncio event loop.
        """
        try:
            await self.initialize()
            await self._display_welcome()
            await self._main_loop()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            logger.info("Application terminated by user")
            
        except Exception as e:
            print(f"\n\nApplication error: {e}")
            logger.error(f"Application error: {e}")
            
        finally:
            await self._cleanup()
    
    async def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("=" * 80)
        print("🎓 RAG Text-to-SQL System - Phase 1 MVP")
        print("   Learning Analytics")
        print("=" * 80)
        print()
        print("📊 Ask questions about learning and development data using natural language!")
        print("   The system will convert your questions into SQL queries and show results.")
        print()
        print("🔒 Security: Read-only database access, no data modification possible")
        print(f"🔧 Model: {self.settings.llm_model_name}")
        print(f"📋 Session: {self.session_id}")
        print()
        print("💡 Example questions you can ask:")
        for i, example in enumerate(self.example_queries, 1):
            print(f"   {i}. {example}")
        print()
        print("📝 Commands:")
        print("   - Type your question and press Enter")
        print("   - Type 'examples' to see example questions again")
        print("   - Type 'help' for more information")  
        print("   - Type 'quit' or 'exit' to leave")
        print("   - Press Ctrl+C to exit")
        print()
        print("-" * 80)
    
    async def _main_loop(self) -> None:
        """Main interaction loop."""
        while True:
            try:
                # Get user input
                print()
                question = input("Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    await self._show_help()
                    continue
                elif question.lower() == 'examples':
                    await self._show_examples()
                    continue
                
                # Process the question
                await self._process_question(question)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing question: {e}")
                logger.error(f"Question processing error: {e}")
    
    async def _process_question(self, question: str) -> None:
        """
        Process user question through SQL tool.
        
        Args:
            question: Natural language question
        """
        query_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            print(f"\nProcessing your question... (ID: {query_id})")
            
            # Process question through SQL tool
            result = await self.sql_tool.process_question(question)
            
            processing_time = time.time() - start_time
            
            if result.success:
                await self._display_success_result(result, processing_time)
                
                # Log successful query
                logger.log_user_query(
                    query_id=query_id,
                    query_type='sql',
                    processing_time=processing_time,
                    success=True
                )
                
            else:
                await self._display_error_result(result, processing_time)
                
                # Log failed query
                logger.log_user_query(
                    query_id=query_id,
                    query_type='sql',
                    processing_time=processing_time,
                    success=False,
                    error=result.error
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Failed to process question: {e}")
            logger.error(f"Question processing failed: {e}")
            
            # Log error
            logger.log_user_query(
                query_id=query_id,
                query_type='sql',
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    async def _display_success_result(self, result, processing_time: float) -> None:
        """Display successful query result."""
        print("Query completed successfully!")
        print()
        print("Generated SQL Query:")
        print("```sql")
        print(result.query)
        print("```")
        print()
        print("Results:")
        print("-" * 40)
        
        # Display result with formatting
        if result.result:
            # Try to format result nicely
            result_str = str(result.result)
            lines = result_str.split('\n')
            
            # Limit output length for terminal
            if len(lines) > 20:
                for line in lines[:20]:
                    print(line)
                print(f"... ({len(lines) - 20} more rows)")
            else:
                print(result_str)
        else:
            print("(No results returned)")
        
        print("-" * 40)
        print(f"Execution time: {result.execution_time:.3f}s")
        print(f"Total processing time: {processing_time:.3f}s")
        if result.row_count is not None:
            print(f"Rows returned: {result.row_count}")
    
    async def _display_error_result(self, result, processing_time: float) -> None:
        """Display error result."""
        print("Query failed")
        print()
        if result.query:
            print("Generated SQL Query:")
            print("```sql")
            print(result.query)
            print("```")
            print()
        
        print("Error Details:")
        print(f"   {result.error}")
        print()
        print(f"Processing time: {processing_time:.3f}s")
        print()
        print("Suggestions:")
        print("   - Try rephrasing your question")
        print("   - Be more specific about what data you want")
        print("   - Use terms like 'count', 'show', 'list', or 'breakdown'")
        print("   - Refer to 'users', 'courses', 'attendance', or 'agencies'")
    
    async def _show_help(self) -> None:
        """Display help information."""
        print()
        print("RAG Text-to-SQL Help")
        print("=" * 30)
        print()
        print("How it works:")
        print("   1. You ask a question in natural language")
        print("   2. The system converts it to a SQL query")
        print("   3. The query is executed against the database")
        print("   4. Results are displayed in a readable format")
        print()
        print("Available data:")
        print("   • Users: APS staff with levels and agencies")
        print("   • Learning Content: Courses, videos, live sessions")
        print("   • Attendance: Participation records and completion status")
        print()
        print("Question types that work well:")
        print("   • Counting: 'How many users...', 'Count of courses...'")
        print("   • Grouping: 'Breakdown by agency', 'Statistics by level'")
        print("   • Filtering: 'Completed courses', 'Level 6 users'")
        print("   • Top/Bottom: 'Highest enrollment', 'Most popular courses'")
        print()
        print("What doesn't work:")
        print("   • Data modification (INSERT, UPDATE, DELETE)")
        print("   • Questions about specific individuals")
        print("   • Real-time or future predictions")
        print("   • Data outside the learning analytics domain")
    
    async def _show_examples(self) -> None:
        """Display example questions."""
        print()
        print("Example Questions")
        print("=" * 25)
        print()
        for i, example in enumerate(self.example_queries, 1):
            print(f"{i:2}. {example}")
        print()
        print("Feel free to ask variations of these questions!")
        print("   The system can handle different phrasings and filters.")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.sql_tool:
                await self.sql_tool.close()
            logger.info(f"Terminal application session {self.session_id} ended")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def run_terminal_app() -> None:
    """
    Entry point for running the terminal application.
    
    This function is called from runner.py with asyncio.run()
    """
    app = TerminalApp()
    await app.run()


# For direct execution (development/testing)
if __name__ == "__main__":
    asyncio.run(run_terminal_app())
