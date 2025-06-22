"""
Terminal Application Interface for RAG System with LangGraph Agent

Implements the enhanced terminal application with LangGraph agent integration
for intelligent query routing between SQL analysis and vector search.

Phase 3 Features:
- LangGraph agent orchestration with intelligent query routing
- Multi-modal processing (SQL + Vector search + Hybrid)
- Interactive clarification handling for ambiguous queries  
- Feedback collection system (thumbs up/down rating)
- Progress indicators for long-running operations
- Enhanced error messaging with recovery suggestions

Security: Read-only database access, mandatory Australian PII protection.
Performance: Non-blocking async operations with intelligent caching.
Privacy: Australian Privacy Principles (APP) compliance maintained.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any
import time
import uuid

from ..core.agent import RAGAgent, AgentConfig, create_rag_agent
from ..core.text_to_sql.sql_tool import AsyncSQLTool
from ..utils.llm_utils import get_llm
from ..utils.logging_utils import get_logger
from ..config.settings import get_settings


logger = get_logger(__name__)


class TerminalApp:
    """
    Enhanced terminal application for RAG system with LangGraph agent integration.
    
    **LangGraph Integration**: Uses RAG agent as primary query processor with
    intelligent routing between SQL analysis, vector search, and hybrid processing.
    
    **Interactive Features**: Clarification handling, feedback collection, and
    progress indicators for enhanced user experience.
    """
    
    def __init__(self, enable_agent: bool = True):
        """
        Initialize terminal application.
        
        Args:
            enable_agent: Whether to use LangGraph agent (True) or legacy SQL-only mode (False)
        """
        self.settings = get_settings()
        self.enable_agent = enable_agent
        
        # Core components
        self.agent: Optional[RAGAgent] = None
        self.sql_tool: Optional[AsyncSQLTool] = None  # Fallback for legacy mode
        self.session_id = str(uuid.uuid4())[:8]
        
        # Enhanced features
        self.query_count = 0
        self.feedback_collected = {}
        
        # Example queries updated for multi-modal capabilities
        self.example_queries = [
            # SQL Analysis examples
            "How many users completed courses in each agency?",
            "Show attendance status breakdown by user level",
            "Which courses have the highest enrollment rates?",
            "What are the completion statistics by content type?",
            
            # Vector Search examples  
            "What feedback did users give about virtual learning?",
            "How do users feel about the new platform features?",
            "What are the main concerns in user comments?",
            "Show me positive experiences with online courses",
            
            # Hybrid Analysis examples
            "Analyze satisfaction trends with supporting user feedback",
            "Compare course completion rates with user sentiment",
            "Show performance metrics and related user comments",
            "Provide a comprehensive analysis of platform adoption"
        ]
    
    async def initialize(self) -> None:
        """
        Initialize RAG agent or SQL tool and verify system readiness.
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info(f"Initializing terminal application (session: {self.session_id}, agent: {self.enable_agent})")
            
            if self.enable_agent:
                # Initialize LangGraph RAG agent (Phase 3)
                agent_config = AgentConfig(
                    max_retries=3,
                    classification_timeout=5.0,
                    tool_timeout=30.0,
                    enable_parallel_execution=True,
                    enable_early_feedback=True,
                    pii_detection_required=True
                )
                
                self.agent = await create_rag_agent(agent_config)
                logger.info("RAG agent initialized successfully")
                
            else:
                # Initialize legacy SQL-only tool (backward compatibility)
                llm = get_llm()
                self.sql_tool = AsyncSQLTool(llm)
                await self.sql_tool.initialize()
                logger.info("SQL tool initialized successfully (legacy mode)")
            
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
        if self.enable_agent:
            print("🤖 RAG System with LangGraph Agent - Phase 3 Complete")
            print("   Intelligent Query Routing & Multi-Modal Analysis")
        else:
            print("🎓 RAG Text-to-SQL System - Legacy Mode")
            print("   SQL Analysis Only")
        print("=" * 80)
        print()
        
        if self.enable_agent:
            print("🧠 Ask questions about learning data using natural language!")
            print("   The system intelligently routes your questions to:")
            print("   📊 SQL Analysis - for statistics and numerical data")
            print("   💬 Vector Search - for user feedback and experiences")  
            print("   🔄 Hybrid Processing - for comprehensive insights")
            print()
            print("✨ Enhanced Features:")
            print("   • Intelligent query classification with confidence scoring")
            print("   • Interactive clarification for ambiguous questions")
            print("   • Feedback collection to improve system performance")
            print("   • Australian PII protection throughout processing")
        else:
            print("📊 Ask questions about learning data using natural language!")
            print("   The system will convert your questions into SQL queries.")
        
        print()
        print("🔒 Security: Read-only database access, mandatory PII protection")
        print(f"🔧 Model: {self.settings.llm_model_name}")
        print(f"📋 Session: {self.session_id}")
        print()
        print("💡 Example questions you can ask:")
        
        # Group examples by type if using agent
        if self.enable_agent:
            print("\n   📊 Statistical Analysis:")
            for example in self.example_queries[:4]:
                print(f"      • {example}")
            
            print("\n   💬 Feedback Analysis:")
            for example in self.example_queries[4:8]:
                print(f"      • {example}")
                
            print("\n   🔄 Hybrid Analysis:")
            for example in self.example_queries[8:]:
                print(f"      • {example}")
        else:
            for i, example in enumerate(self.example_queries[:6], 1):
                print(f"   {i}. {example}")
        
        print()
        print("📝 Commands:")
        print("   - Type your question and press Enter")
        print("   - Type 'examples' to see example questions again")
        print("   - Type 'help' for more information")
        if self.enable_agent:
            print("   - Type 'stats' to see session statistics")
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
                elif question.lower() == 'stats' and self.enable_agent:
                    await self._show_session_stats()
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
        Process user question through RAG agent or SQL tool.
        
        Args:
            question: Natural language question
        """
        query_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        self.query_count += 1
        
        try:
            print(f"\n🔄 Processing your question... (ID: {query_id})")
            
            if self.enable_agent:
                # Process through LangGraph RAG agent
                result = await self._process_with_agent(question, query_id)
                processing_time = time.time() - start_time
                
                # Display result based on agent output
                if result.get('error'):
                    await self._display_agent_error(result, processing_time)
                elif result.get('requires_clarification'):
                    await self._handle_clarification(result, question, query_id)
                else:
                    await self._display_agent_success(result, processing_time)
                    
                    # Collect feedback for successful responses
                    await self._collect_feedback(query_id, result)
                
                # Log query with agent metadata
                logger.log_user_query(
                    query_id=query_id,
                    query_type='agent',
                    processing_time=processing_time,
                    success=not result.get('error'),
                    error=result.get('error'),
                    metadata={
                        'classification': result.get('classification'),
                        'confidence': result.get('confidence'),
                        'tools_used': result.get('tools_used', []),
                        'requires_clarification': result.get('requires_clarification', False)
                    }
                )
                
            else:
                # Legacy SQL-only processing
                result = await self.sql_tool.process_question(question)
                processing_time = time.time() - start_time
                
                if result.success:
                    await self._display_success_result(result, processing_time)
                else:
                    await self._display_error_result(result, processing_time)
                
                # Log legacy query
                logger.log_user_query(
                    query_id=query_id,
                    query_type='sql',
                    processing_time=processing_time,
                    success=result.success,
                    error=result.error if not result.success else None
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Failed to process question: {e}")
            logger.error(f"Question processing failed: {e}")
            
            # Log error
            logger.log_user_query(
                query_id=query_id,
                query_type='agent' if self.enable_agent else 'sql',
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    async def _process_with_agent(self, question: str, query_id: str) -> Dict[str, Any]:
        """Process question through RAG agent with progress indicators."""
        try:
            # Show progress indicator
            print("🧠 Classifying query type...")
            
            # Prepare initial state for agent
            initial_state = {
                "query": question,
                "session_id": self.session_id,
                "retry_count": 0,
                "tools_used": [],
                "requires_clarification": False
            }
            
            # Process through agent
            final_state = await self.agent.ainvoke(initial_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            return {
                "error": f"Agent processing failed: {str(e)}",
                "query": question,
                "session_id": self.session_id,
                "tools_used": ["agent_error"]
            }
    
    async def _display_agent_success(self, result: Dict[str, Any], processing_time: float) -> None:
        """Display successful RAG agent result."""
        print("✅ Query processed successfully!")
        print()
        
        # Show classification information
        classification = result.get('classification', 'Unknown')
        confidence = result.get('confidence', 'Unknown')
        tools_used = result.get('tools_used', [])
        
        print(f"🧠 Query Classification: {classification} (Confidence: {confidence})")
        print(f"🔧 Tools Used: {', '.join(tools_used)}")
        print()
        
        # Display the synthesized answer
        final_answer = result.get('final_answer', 'No answer generated')
        print("📋 Analysis Result:")
        print("-" * 50)
        print(final_answer)
        print("-" * 50)
        
        # Show sources if available
        sources = result.get('sources', [])
        if sources:
            print(f"📚 Sources: {', '.join(sources)}")
            print()
        
        # Display performance metrics
        agent_processing_time = result.get('processing_time', 0)
        print(f"⏱️  Agent Processing: {agent_processing_time:.3f}s")
        print(f"⏱️  Total Time: {processing_time:.3f}s")
        
        # Show any additional metadata
        if result.get('sql_result') and result.get('vector_result'):
            print("🔄 Hybrid Analysis: Combined SQL and vector search results")
        elif result.get('sql_result'):
            print("📊 SQL Analysis: Database query executed")
        elif result.get('vector_result'):
            print("💬 Vector Search: Feedback analysis performed")
    
    async def _display_agent_error(self, result: Dict[str, Any], processing_time: float) -> None:
        """Display RAG agent error result."""
        print("❌ Query processing encountered an issue")
        print()
        
        error_message = result.get('error', 'Unknown error occurred')
        final_answer = result.get('final_answer', '')
        
        # Display user-friendly error response
        if final_answer:
            print("💡 Guidance:")
            print("-" * 30)
            print(final_answer)
            print("-" * 30)
        else:
            print(f"Error Details: {error_message}")
        
        print()
        print(f"⏱️  Processing time: {processing_time:.3f}s")
        
        # Show tools that were attempted
        tools_used = result.get('tools_used', [])
        if tools_used:
            print(f"🔧 Attempted: {', '.join(tools_used)}")
    
    async def _handle_clarification(self, result: Dict[str, Any], original_question: str, query_id: str) -> None:
        """Handle clarification requests from the agent."""
        print("🤔 Your question needs clarification")
        print()
        
        clarification_message = result.get('final_answer', 'Please clarify your question.')
        print(clarification_message)
        print()
        
        # Wait for user clarification
        while True:
            try:
                clarification = input("Your choice (A/B/C or rephrase your question): ").strip()
                
                if not clarification:
                    continue
                
                if clarification.upper() in ['A', 'B', 'C']:
                    # Process clarification choice
                    clarified_question = self._interpret_clarification_choice(
                        original_question, clarification.upper()
                    )
                    print(f"\n🔄 Processing clarified request: {clarified_question}")
                    
                    # Reprocess with clarification
                    await self._process_question(clarified_question)
                    break
                    
                else:
                    # User provided a rephrased question
                    print(f"\n🔄 Processing rephrased question...")
                    await self._process_question(clarification)
                    break
                    
            except KeyboardInterrupt:
                print("\n⏭️  Skipping clarification...")
                break
    
    def _interpret_clarification_choice(self, original_question: str, choice: str) -> str:
        """Interpret clarification choice and modify the original question."""
        if choice == 'A':
            return f"{original_question} - I want statistical summary and numerical breakdown"
        elif choice == 'B':
            return f"{original_question} - I want specific feedback, comments, and user experiences"
        elif choice == 'C':
            return f"{original_question} - I want combined analysis with both numbers and feedback"
        else:
            return original_question
    
    async def _collect_feedback(self, query_id: str, result: Dict[str, Any]) -> None:
        """Collect user feedback on the response quality."""
        try:
            print()
            feedback_response = input("👍 Was this response helpful? (y/n/skip): ").strip().lower()
            
            if feedback_response in ['y', 'yes', '👍']:
                self.feedback_collected[query_id] = {'helpful': True, 'rating': 'positive'}
                print("✅ Thank you for the positive feedback!")
                
            elif feedback_response in ['n', 'no', '👎']:
                self.feedback_collected[query_id] = {'helpful': False, 'rating': 'negative'}
                print("💡 Thank you for the feedback. We'll use it to improve the system.")
                
                # Optional: collect specific feedback
                specific_feedback = input("🔧 What could be improved? (optional): ").strip()
                if specific_feedback:
                    self.feedback_collected[query_id]['details'] = specific_feedback
                    
            # Log feedback for analytics
            if query_id in self.feedback_collected:
                logger.info(f"User feedback collected for query {query_id}: {self.feedback_collected[query_id]}")
                
        except KeyboardInterrupt:
            print("\n⏭️  Skipping feedback collection...")
    
    async def _show_session_stats(self) -> None:
        """Display session statistics for agent mode."""
        print()
        print("📊 Session Statistics")
        print("=" * 30)
        print(f"Session ID: {self.session_id}")
        print(f"Queries Processed: {self.query_count}")
        print(f"Feedback Collected: {len(self.feedback_collected)}")
        
        if self.feedback_collected:
            positive_feedback = sum(1 for f in self.feedback_collected.values() if f.get('helpful'))
            feedback_rate = (positive_feedback / len(self.feedback_collected)) * 100
            print(f"Positive Feedback Rate: {feedback_rate:.1f}%")
        
        print(f"Agent Mode: {'Enabled' if self.enable_agent else 'Disabled'}")
        print(f"Model: {self.settings.llm_model_name}")
        print()
    
    async def _display_success_result(self, result, processing_time: float) -> None:
        """Display successful query result (legacy SQL mode)."""
        print("✅ Query completed successfully!")
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
        """Display error result (legacy SQL mode)."""
        print("❌ Query failed")
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
        print("💡 Suggestions:")
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
