"""
LangGraph Agent for RAG System - Core Intelligence Orchestrator

This module implements the main async LangGraph agent that serves as the central
intelligence for the RAG system, routing queries between SQL and vector search tools
with comprehensive error handling and graceful degradation.

Key Features:
- Async-first design with LangGraph StateGraph orchestration
- Multi-stage query classification with fallback mechanisms
- Intelligent routing between SQL, Vector, and Hybrid processing
- Comprehensive error handling with graceful degradation
- Australian PII protection throughout the workflow
- Early feedback collection system integration

Security: Read-only database access, mandatory PII anonymization.
Performance: Non-blocking async operations with parallel tool execution.
Privacy: Australian Privacy Principles (APP) compliance maintained.

Example Usage:
    # Basic agent usage with default configuration
    async def main():
        # Create and initialize agent
        agent = await create_rag_agent()
        
        try:
            # Process a statistical query
            result = await agent.ainvoke({
                "query": "How many users completed courses in each agency?",
                "session_id": "user_123"
            })
            print(f"Answer: {result['final_answer']}")
            print(f"Tools used: {result['tools_used']}")
            
            # Process a feedback search query
            result = await agent.ainvoke({
                "query": "What feedback did users give about virtual learning?",
                "session_id": "user_123"
            })
            print(f"Answer: {result['final_answer']}")
            
        finally:
            await agent.close()
    
    # Advanced usage with custom configuration
    async def advanced_usage():
        config = AgentConfig(
            max_retries=5,
            classification_timeout=3.0,
            enable_parallel_execution=True,
            pii_detection_required=True
        )
        
        agent = RAGAgent(config)
        await agent.initialize()
        
        try:
            # Process complex hybrid query
            result = await agent.ainvoke({
                "query": "Analyze satisfaction trends across agencies with supporting feedback",
                "session_id": "analyst_456",
                "retry_count": 0,
                "tools_used": []
            })
            
            # Check if clarification is needed
            if result.get('requires_clarification'):
                print("Clarification needed:", result['final_answer'])
                # Handle user clarification response
                clarified_result = await agent.ainvoke({
                    "query": "Analyze satisfaction trends - I want both statistics and comments",
                    "session_id": "analyst_456",
                    "user_feedback": "Option C - Combined analysis"
                })
                print(f"Final answer: {clarified_result['final_answer']}")
            
        finally:
            await agent.close()
    
    # Error handling example
    async def error_handling_example():
        agent = await create_rag_agent()
        
        try:
            # This might trigger error handling
            result = await agent.ainvoke({
                "query": "",  # Empty query
                "session_id": "test_session"
            })
            
            if result.get('error'):
                print(f"Error occurred: {result['error']}")
                print(f"Error handling provided: {result['final_answer']}")
            
        finally:
            await agent.close()
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Literal, Union
from typing_extensions import TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLanguageModel

from .text_to_sql.sql_tool import AsyncSQLTool
from .vector_search.vector_search_tool import VectorSearchTool
from .vector_search.search_result import VectorSearchResponse
from .privacy.pii_detector import AustralianPIIDetector
from .routing.query_classifier import QueryClassifier
from .synthesis.answer_generator import AnswerGenerator
from ..utils.llm_utils import get_llm
from ..utils.logging_utils import get_logger
from ..config.settings import get_settings


logger = get_logger(__name__)


# Type definitions for agent state management
ClassificationType = Literal["SQL", "VECTOR", "HYBRID", "CLARIFICATION_NEEDED"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class AgentState(TypedDict):
    """
    Comprehensive state management for RAG agent workflow.
    
    This TypedDict defines the complete state structure that flows through
    the LangGraph nodes, enabling stateful processing and error recovery.
    """
    # Input
    query: str
    session_id: str
    
    # Classification
    classification: Optional[ClassificationType]
    confidence: Optional[ConfidenceLevel]
    classification_reasoning: Optional[str]
    
    # Tool Results
    sql_result: Optional[Dict[str, Any]]
    vector_result: Optional[Dict[str, Any]]
    
    # Synthesis
    final_answer: Optional[str]
    sources: Optional[List[str]]
    
    # Error Handling & Flow Control
    error: Optional[str]
    retry_count: int
    requires_clarification: bool
    user_feedback: Optional[str]
    
    # Metadata
    processing_time: Optional[float]
    tools_used: List[str]


@dataclass
class AgentConfig:
    """Configuration settings for RAG agent behavior."""
    max_retries: int = 3
    classification_timeout: float = 5.0
    tool_timeout: float = 30.0
    enable_parallel_execution: bool = True
    enable_early_feedback: bool = True
    pii_detection_required: bool = True


class RAGAgent:
    """
    Main async LangGraph agent for RAG system orchestration.
    
    This agent serves as the central intelligence that:
    1. Classifies incoming queries using multi-stage approach
    2. Routes queries to appropriate tools (SQL, Vector, or both)
    3. Synthesizes results into coherent answers
    4. Handles errors gracefully with fallback mechanisms
    5. Maintains Australian PII protection throughout
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize RAG agent with configuration and dependencies.
        
        Args:
            config: Agent configuration settings
        """
        self.config = config or AgentConfig()
        self.settings = get_settings()
        
        # Core components (initialized during setup)
        self._llm: Optional[BaseLanguageModel] = None
        self._sql_tool: Optional[AsyncSQLTool] = None
        self._vector_tool: Optional[VectorSearchTool] = None
        self._pii_detector: Optional[AustralianPIIDetector] = None
        self._query_classifier: Optional[QueryClassifier] = None
        self._answer_generator: Optional[AnswerGenerator] = None
        
        # LangGraph workflow
        self._graph: Optional[StateGraph] = None
        self._compiled_graph = None
        
        # Metrics tracking
        self._query_count = 0
        self._error_count = 0
        
    async def initialize(self) -> None:
        """
        Initialize all agent components and compile the LangGraph workflow.
        
        This method must be called before using the agent. It sets up:
        - LLM provider connection
        - SQL and Vector search tools
        - PII detection system
        - LangGraph workflow compilation
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Initializing RAG agent components...")
            start_time = time.time()
            
            # Initialize core LLM
            self._llm = get_llm()
            logger.info(f"LLM provider initialized: {type(self._llm).__name__}")
            
            # Initialize tools
            await self._initialize_tools()
            
            # Initialize PII detection
            if self.config.pii_detection_required:
                self._pii_detector = AustralianPIIDetector()
                logger.info("PII detection system initialized")
            
            # Initialize query classifier
            self._query_classifier = QueryClassifier(llm=self._llm)
            await self._query_classifier.initialize()
            logger.info("Query classifier initialized")
            
            # Initialize answer generator
            self._answer_generator = AnswerGenerator(
                llm=self._llm,
                pii_detector=self._pii_detector,
                enable_source_attribution=True
            )
            logger.info("Answer generator initialized")
            
            # Build and compile LangGraph workflow
            await self._build_graph()
            
            initialization_time = time.time() - start_time
            logger.info(f"RAG agent initialized successfully in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"RAG agent initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize RAG agent: {e}")
    
    async def _initialize_tools(self) -> None:
        """Initialize SQL and Vector search tools."""
        try:
            # Initialize SQL tool
            self._sql_tool = AsyncSQLTool(
                llm=self._llm,
                max_retries=self.config.max_retries
            )
            await self._sql_tool.initialize()
            logger.info("SQL tool initialized successfully")
            
            # Initialize Vector search tool
            self._vector_tool = VectorSearchTool()
            await self._vector_tool.initialize()
            logger.info("Vector search tool initialized successfully")
            
        except Exception as e:
            logger.error(f"Tool initialization failed: {e}")
            raise
    
    async def _build_graph(self) -> None:
        """Build and compile the LangGraph workflow."""
        try:
            # Create StateGraph with AgentState
            workflow = StateGraph(AgentState)
            
            # Add nodes
            workflow.add_node("classify_query", self._classify_query_node)
            workflow.add_node("sql_tool", self._sql_tool_node)
            workflow.add_node("vector_search_tool", self._vector_search_tool_node)
            workflow.add_node("hybrid_processing", self._hybrid_processing_node)
            workflow.add_node("synthesis", self._synthesis_node)
            workflow.add_node("clarification", self._clarification_node)
            workflow.add_node("error_handling", self._error_handling_node)
            
            # Set entry point
            workflow.set_entry_point("classify_query")
            
            # Add conditional routing with hybrid support
            workflow.add_conditional_edges(
                "classify_query",
                self._route_after_classification,
                {
                    "sql": "sql_tool",
                    "vector": "vector_search_tool",
                    "hybrid": "hybrid_processing",
                    "clarification": "clarification",
                    "error": "error_handling"
                }
            )
            
            # Add edges from tools to synthesis
            workflow.add_edge("sql_tool", "synthesis")
            workflow.add_edge("vector_search_tool", "synthesis")
            workflow.add_edge("hybrid_processing", "synthesis")
            
            # Add final edges
            workflow.add_edge("synthesis", END)
            workflow.add_edge("clarification", END)
            workflow.add_edge("error_handling", END)
            
            # Compile the graph
            self._compiled_graph = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"Graph compilation failed: {e}")
            raise
    
    async def ainvoke(self, initial_state: Dict[str, Any]) -> AgentState:
        """
        Process a query through the complete RAG agent workflow.
        
        Args:
            initial_state: Initial state containing user query and metadata
            
        Returns:
            Final agent state with results or error information
            
        Raises:
            RuntimeError: If agent is not initialized or processing fails
        """
        if self._compiled_graph is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            self._query_count += 1
            start_time = time.time()
            
            # Ensure required fields are present
            state = AgentState({
                "query": initial_state.get("query", ""),
                "session_id": initial_state.get("session_id", str(uuid.uuid4())[:8]),
                "classification": None,
                "confidence": None,
                "classification_reasoning": None,
                "sql_result": None,
                "vector_result": None,
                "final_answer": None,
                "sources": None,
                "error": None,
                "retry_count": initial_state.get("retry_count", 0),
                "requires_clarification": False,
                "user_feedback": None,
                "processing_time": None,
                "tools_used": initial_state.get("tools_used", [])
            })
            
            # Process through LangGraph
            final_state = await self._compiled_graph.ainvoke(state)
            
            # Add processing metadata
            final_state["processing_time"] = time.time() - start_time
            
            logger.info(
                f"Query processed successfully in {final_state['processing_time']:.2f}s "
                f"(session: {final_state['session_id']}, tools: {final_state['tools_used']})"
            )
            
            return final_state
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Query processing failed: {e}")
            
            # Return error state
            return AgentState({
                **initial_state,
                "error": f"Processing failed: {str(e)}",
                "processing_time": time.time() - start_time if 'start_time' in locals() else None,
                "tools_used": initial_state.get("tools_used", []) + ["error"]
            })
    
    # Placeholder node implementations
    # These will be implemented in subsequent phases
    
    async def _classify_query_node(self, state: AgentState) -> AgentState:
        """
        Classify query using multi-stage classifier with PII protection.
        
        Performs comprehensive query classification to determine optimal routing
        strategy while ensuring Australian PII compliance.
        """
        logger.info(f"Classifying query: {state['query'][:50]}...")
        
        try:
            if not state["query"].strip():
                logger.warning("Empty query received")
                return {
                    **state,
                    "error": "Query cannot be empty",
                    "tools_used": state["tools_used"] + ["classifier_error"]
                }
            
            # Use the actual query classifier
            classification_result = await self._query_classifier.classify_query(
                query=state["query"],
                session_id=state["session_id"]
            )
            
            logger.info(
                f"Query classified as {classification_result.classification} "
                f"with {classification_result.confidence} confidence"
            )
            
            return {
                **state,
                "classification": classification_result.classification,
                "confidence": classification_result.confidence,
                "classification_reasoning": classification_result.reasoning,
                "tools_used": state["tools_used"] + ["classifier"]
            }
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            
            # Fallback to basic classification on error
            query_lower = state["query"].lower()
            if any(word in query_lower for word in ["count", "how many", "average", "percentage"]):
                classification = "SQL"
                confidence = "MEDIUM"
                reasoning = "Fallback classification: statistical keywords detected"
            elif any(word in query_lower for word in ["feedback", "comment", "experience", "opinion"]):
                classification = "VECTOR"
                confidence = "MEDIUM"
                reasoning = "Fallback classification: feedback keywords detected"
            else:
                classification = "SQL"
                confidence = "LOW"
                reasoning = "Fallback classification: default to SQL"
            
            logger.warning(f"Using fallback classification: {classification}")
            
            return {
                **state,
                "classification": classification,
                "confidence": confidence,
                "classification_reasoning": reasoning,
                "tools_used": state["tools_used"] + ["classifier_fallback"]
            }
    
    async def _sql_tool_node(self, state: AgentState) -> AgentState:
        """
        Process query using SQL tool with comprehensive error handling.
        
        Executes database queries with privacy protection and robust error handling.
        """
        logger.info("Processing query with SQL tool...")
        
        try:
            if not self._sql_tool:
                raise RuntimeError("SQL tool not initialized")
            
            # Apply timeout for SQL operations
            import asyncio
            
            async def sql_with_timeout():
                return await self._sql_tool.process_question(state["query"])
            
            
            # Execute with timeout
            result = await asyncio.wait_for(
                sql_with_timeout(),
                timeout=self.config.tool_timeout
            )
            
            # Check if result indicates success
            if result.success:
                logger.info(f"SQL tool executed successfully, returned {len(str(result))} chars")
                
                return {
                    **state,
                    "sql_result": result,
                    "tools_used": state["tools_used"] + ["sql"]
                }
            else:
                # SQL tool returned error
                error_msg = result.get("error", "SQL execution failed")
                logger.warning(f"SQL tool returned error: {error_msg}")
                
                return {
                    **state,
                    "error": f"SQL processing error: {error_msg}",
                    "tools_used": state["tools_used"] + ["sql_error"]
                }
            
        except asyncio.TimeoutError:
            logger.error(f"SQL operation timed out after {self.config.tool_timeout}s")
            return {
                **state,
                "error": "SQL operation timed out. Please try a simpler query.",
                "tools_used": state["tools_used"] + ["sql_timeout"]
            }
            
        except Exception as e:
            logger.error(f"SQL tool failed: {e}")
            
            # Check for retry possibility
            if state["retry_count"] < self.config.max_retries:
                logger.info(f"Retrying SQL operation (attempt {state['retry_count'] + 1})")
                return {
                    **state,
                    "retry_count": state["retry_count"] + 1,
                    "tools_used": state["tools_used"] + ["sql_retry"]
                }
            else:
                return {
                    **state,
                    "error": f"SQL processing failed after {self.config.max_retries} retries: {str(e)}",
                    "tools_used": state["tools_used"] + ["sql_failed"]
                }
    
    async def _vector_search_tool_node(self, state: AgentState) -> AgentState:
        """
        Process query using vector search tool with comprehensive error handling.
        
        Executes semantic search on user feedback with privacy protection.
        """
        logger.info("Processing query with vector search tool...")
        
        try:
            if not self._vector_tool:
                raise RuntimeError("Vector search tool not initialized")
            
            # Apply timeout for vector search operations
            import asyncio
            
            async def vector_with_timeout():
                return await self._vector_tool.search(
                    query=state["query"],
                    max_results=10  # Configurable limit
                )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                vector_with_timeout(),
                timeout=self.config.tool_timeout
            )
            
            # Validate search results
            if result and result.results:
                logger.info(f"Vector search found {len(result.results)} results")
                
                return {
                    **state,
                    "vector_result": result,
                    "tools_used": state["tools_used"] + ["vector"]
                }
            else:
                # No results found
                logger.info("Vector search completed but found no relevant results")
                
                return {
                    **state,
                    "vector_result": {
                        "query": state["query"],
                        "results": [],
                        "message": "No relevant feedback found for this query"
                    },
                    "tools_used": state["tools_used"] + ["vector_empty"]
                }
            
        except asyncio.TimeoutError:
            logger.error(f"Vector search timed out after {self.config.tool_timeout}s")
            return {
                **state,
                "error": "Vector search timed out. Please try a more specific query.",
                "tools_used": state["tools_used"] + ["vector_timeout"]
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            
            # Check for retry possibility
            if state["retry_count"] < self.config.max_retries:
                logger.info(f"Retrying vector search (attempt {state['retry_count'] + 1})")
                return {
                    **state,
                    "retry_count": state["retry_count"] + 1,
                    "tools_used": state["tools_used"] + ["vector_retry"]
                }
            else:
                return {
                    **state,
                    "error": f"Vector search failed after {self.config.max_retries} retries: {str(e)}",
                    "tools_used": state["tools_used"] + ["vector_failed"]
                }
    
    async def _hybrid_processing_node(self, state: AgentState) -> AgentState:
        """
        Process query using both SQL and vector search tools in parallel or sequence.
        
        Executes hybrid processing strategy for comprehensive analysis combining
        statistical data with qualitative feedback.
        """
        logger.info("Processing query with hybrid approach (SQL + Vector)...")
        
        try:
            if self.config.enable_parallel_execution:
                # Execute both tools in parallel for better performance
                import asyncio
                
                async def run_parallel_tools():
                    # Create tasks for both tools
                    sql_task = asyncio.create_task(
                        self._execute_sql_safely(state)
                    )
                    vector_task = asyncio.create_task(
                        self._execute_vector_safely(state)
                    )
                    
                    # Wait for both to complete
                    sql_result, vector_result = await asyncio.gather(
                        sql_task, vector_task, return_exceptions=True
                    )
                    
                    return sql_result, vector_result
                
                # Execute with overall timeout
                sql_result, vector_result = await asyncio.wait_for(
                    run_parallel_tools(),
                    timeout=self.config.tool_timeout * 1.5  # Extra time for parallel execution
                )
                
            else:
                # Sequential execution
                logger.info("Executing SQL tool first...")
                sql_result = await self._execute_sql_safely(state)
                
                logger.info("Executing vector search tool...")
                vector_result = await self._execute_vector_safely(state)
            
            # Process results from both tools
            final_state = {**state}
            tools_used = state["tools_used"] + ["hybrid"]
            
            # Handle SQL results
            if isinstance(sql_result, Exception):
                logger.warning(f"SQL component of hybrid failed: {sql_result}")
                tools_used.append("sql_failed")
            elif sql_result and not sql_result.get("error"):
                final_state["sql_result"] = sql_result
                tools_used.append("sql")
            
            # Handle vector results
            if isinstance(vector_result, Exception):
                logger.warning(f"Vector component of hybrid failed: {vector_result}")
                tools_used.append("vector_failed")
            elif vector_result:
                # Handle both dict and VectorSearchResponse objects
                if hasattr(vector_result, 'results'):
                    # VectorSearchResponse object
                    final_state["vector_result"] = vector_result
                    tools_used.append("vector")
                elif isinstance(vector_result, dict) and not vector_result.get("error"):
                    # Dict result without error
                    final_state["vector_result"] = vector_result
                    tools_used.append("vector")
            
            # Check if at least one tool succeeded
            if not final_state.get("sql_result") and not final_state.get("vector_result"):
                return {
                    **state,
                    "error": "Both SQL and vector search failed in hybrid processing",
                    "tools_used": tools_used + ["hybrid_failed"]
                }
            
            logger.info("Hybrid processing completed successfully")
            
            return {
                **final_state,
                "tools_used": tools_used
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Hybrid processing timed out")
            return {
                **state,
                "error": "Hybrid processing timed out. Please try a simpler query.",
                "tools_used": state["tools_used"] + ["hybrid_timeout"]
            }
            
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return {
                **state,
                "error": f"Hybrid processing failed: {str(e)}",
                "tools_used": state["tools_used"] + ["hybrid_error"]
            }
    
    async def _execute_sql_safely(self, state: AgentState) -> Dict[str, Any]:
        """Execute SQL tool with error handling for hybrid processing."""
        try:
            if not self._sql_tool:
                raise RuntimeError("SQL tool not initialized")
            
            return await self._sql_tool.process_question(state["query"])
        except Exception as e:
            logger.error(f"SQL execution in hybrid failed: {e}")
            return {"error": str(e)}
    
    async def _execute_vector_safely(self, state: AgentState) -> Union[Dict[str, Any], VectorSearchResponse]:
        """Execute vector search tool with error handling for hybrid processing."""
        try:
            if not self._vector_tool:
                raise RuntimeError("Vector search tool not initialized")
            
            return await self._vector_tool.search(
                query=state["query"],
                max_results=10
            )
        except Exception as e:
            logger.error(f"Vector search in hybrid failed: {e}")
            return {"error": str(e)}
    
    async def _synthesis_node(self, state: AgentState) -> AgentState:
        """
        Synthesize comprehensive answer using intelligent answer generation.
        
        Combines results from SQL and vector search tools into coherent,
        well-structured responses with proper source attribution.
        """
        logger.info("Synthesizing final answer using AnswerGenerator...")
        
        try:
            if not self._answer_generator:
                raise RuntimeError("Answer generator not initialized")
            
            # Check if we have any results to synthesize
            has_sql = state.get("sql_result") is not None
            has_vector = state.get("vector_result") is not None
            
            if not has_sql and not has_vector:
                logger.warning("No results available for synthesis")
                return {
                    **state,
                    "final_answer": (
                        "I wasn't able to gather sufficient information to answer your question. "
                        "Please try rephrasing your query or being more specific."
                    ),
                    "sources": [],
                    "tools_used": state["tools_used"] + ["synthesis_no_data"]
                }
            
            # Synthesize answer using AnswerGenerator
            synthesis_result = await self._answer_generator.synthesize_answer(
                query=state["query"],
                sql_result=state.get("sql_result"),
                vector_result=state.get("vector_result"),
                session_id=state["session_id"],
                additional_context=f"Classification: {state.get('classification', 'Unknown')}"
            )
            
            logger.info(
                f"Answer synthesis completed: {synthesis_result.answer_type.value} "
                f"(confidence: {synthesis_result.confidence:.2f}, "
                f"length: {len(synthesis_result.answer)} chars)"
            )
            
            # Update state with synthesis results
            return {
                **state,
                "final_answer": synthesis_result.answer,
                "sources": synthesis_result.sources,
                "tools_used": state["tools_used"] + ["synthesis"]
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            
            # Fallback to basic synthesis
            answer_parts = []
            sources = []
            
            if state.get("sql_result") and state["sql_result"].get("success"):
                sql_data = state["sql_result"].get("result", "No data")
                answer_parts.append(f"📊 Database Analysis:\n{sql_data}")
                sources.append("Database Analysis")
            
            if state.get("vector_result") and state["vector_result"].get("results"):
                vector_data = state["vector_result"]["results"]
                if vector_data:
                    feedback_summary = "\n".join([
                        f"• {item}" for item in vector_data[:3]  # Show top 3 results
                    ])
                    answer_parts.append(f"💬 User Feedback:\n{feedback_summary}")
                    sources.append("User Feedback")
            
            fallback_answer = (
                "\n\n".join(answer_parts) if answer_parts 
                else "Unable to generate a comprehensive answer due to processing issues."
            )
            
            return {
                **state,
                "final_answer": fallback_answer,
                "sources": sources,
                "tools_used": state["tools_used"] + ["synthesis_fallback"]
            }
    
    async def _clarification_node(self, state: AgentState) -> AgentState:
        """
        Handle queries requiring clarification with interactive guidance.
        
        Provides structured clarification options based on query analysis
        and classification reasoning.
        """
        logger.info("Query requires clarification...")
        
        try:
            # Analyze the query to provide specific clarification options
            query = state["query"]
            classification_reasoning = state.get("classification_reasoning", "")
            
            # Generate context-aware clarification message
            clarification_message = self._generate_clarification_message(
                query, classification_reasoning
            )
            
            return {
                **state,
                "final_answer": clarification_message,
                "requires_clarification": True,
                "tools_used": state["tools_used"] + ["clarification"]
            }
            
        except Exception as e:
            logger.error(f"Clarification node failed: {e}")
            
            # Fallback clarification message
            fallback_message = (
                f"I need clarification for your query: '{state['query']}'\n\n"
                "Please specify what type of information you're looking for:\n"
                "A) 📊 Statistical summary or numerical breakdown\n"
                "B) 💬 Specific feedback, comments, or experiences\n"
                "C) 📈 Combined analysis with both numbers and feedback\n\n"
                "Please respond with A, B, or C to continue."
            )
            
            return {
                **state,
                "final_answer": fallback_message,
                "requires_clarification": True,
                "tools_used": state["tools_used"] + ["clarification_fallback"]
            }
    
    async def _error_handling_node(self, state: AgentState) -> AgentState:
        """
        Handle errors with informative responses and recovery suggestions.
        
        Provides specific guidance based on error type and context while
        maintaining user-friendly communication.
        """
        error = state.get("error", "Unknown error occurred")
        query = state["query"]
        tools_attempted = [tool for tool in state["tools_used"] if not tool.endswith("_failed")]
        
        logger.warning(f"Handling error for session {state['session_id']}: {error}")
        
        try:
            # Generate context-specific error message
            error_message = self._generate_error_message(error, query, tools_attempted)
            
            return {
                **state,
                "final_answer": error_message,
                "tools_used": state["tools_used"] + ["error_handler"]
            }
            
        except Exception as e:
            logger.error(f"Error handling node failed: {e}")
            
            # Ultra-safe fallback
            fallback_message = (
                "I encountered an issue processing your request. "
                "Please try rephrasing your question or contact support if the problem persists.\n\n"
                "💡 **Suggestion**: Try being more specific about what information you're looking for."
            )
            
            return {
                **state,
                "final_answer": fallback_message,
                "tools_used": state["tools_used"] + ["error_handler_fallback"]
            }
    
    def _generate_clarification_message(self, query: str, reasoning: str) -> str:
        """Generate context-aware clarification message."""
        base_message = f"I need clarification for your query: '{query}'\n\n"
        
        # Analyze query to provide more specific options
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["trend", "analysis", "pattern", "over time"]):
            options = [
                "A) 📈 **Trend Analysis**: Statistical trends over time periods",
                "B) 💭 **Sentiment Trends**: How user opinions have changed",
                "C) 📊 **Combined View**: Both statistical and sentiment trends"
            ]
        elif any(word in query_lower for word in ["satisfaction", "rating", "score"]):
            options = [
                "A) 🔢 **Numerical Ratings**: Average scores and distributions",
                "B) 💬 **Detailed Feedback**: Specific comments about satisfaction",
                "C) 📋 **Complete Analysis**: Ratings with supporting comments"
            ]
        elif any(word in query_lower for word in ["compare", "comparison", "vs", "versus"]):
            options = [
                "A) 📊 **Statistical Comparison**: Numbers and percentages",
                "B) 💭 **Feedback Comparison**: What users said about each option",
                "C) 🔍 **Comprehensive Comparison**: Both data and feedback"
            ]
        else:
            # Generic options
            options = [
                "A) 📊 **Statistical Summary**: Numbers, counts, and percentages",
                "B) 💬 **User Feedback**: Comments, experiences, and opinions",
                "C) 📈 **Combined Analysis**: Both statistical data and user feedback"
            ]
        
        clarification_text = base_message + "\n".join(options)
        clarification_text += "\n\n**Please respond with A, B, or C to continue.**"
        
        if reasoning:
            clarification_text += f"\n\n*Note: {reasoning}*"
        
        return clarification_text
    
    def _generate_error_message(self, error: str, query: str, tools_attempted: List[str]) -> str:
        """Generate context-specific error message with recovery suggestions."""
        base_message = "I encountered an issue processing your request.\n\n"
        
        # Categorize error types
        if "timeout" in error.lower():
            specific_message = (
                "🕐 **Timeout Error**: Your query took longer than expected to process.\n\n"
                "**Suggestions**:\n"
                "• Try a more specific or simpler question\n"
                "• Break complex queries into smaller parts\n"
                "• If asking about large datasets, consider narrowing the scope"
            )
        elif "sql" in error.lower() and "sql" in tools_attempted:
            specific_message = (
                "🗃️ **Database Query Issue**: I had trouble analyzing the statistical data.\n\n"
                "**Suggestions**:\n"
                "• Try rephrasing your question with clearer criteria\n"
                "• Specify time periods or categories if relevant\n"
                "• Ask about user feedback instead if you need qualitative insights"
            )
        elif "vector" in error.lower() and "vector" in tools_attempted:
            specific_message = (
                "🔍 **Search Issue**: I had trouble finding relevant user feedback.\n\n"
                "**Suggestions**:\n"
                "• Try different keywords related to your topic\n"
                "• Ask about statistical data instead if you need numbers\n"
                "• Be more specific about what type of feedback you're looking for"
            )
        elif "pii" in error.lower():
            specific_message = (
                "🔒 **Privacy Protection**: Your query may contain sensitive information.\n\n"
                "**Suggestions**:\n"
                "• Remove any personal names or identifying information\n"
                "• Ask about general patterns rather than specific individuals\n"
                "• Focus on aggregate data and trends"
            )
        else:
            # Generic error message
            specific_message = (
                "⚠️ **Processing Error**: I encountered an unexpected issue.\n\n"
                "**Suggestions**:\n"
                "• Try rephrasing your question\n"
                "• Be more specific about what you're looking for\n"
                "• Contact support if the problem persists"
            )
        
        return base_message + specific_message
    
    def _route_after_classification(self, state: AgentState) -> str:
        """
        Route to appropriate node based on classification results.
        
        Determines the optimal processing path based on query classification,
        confidence levels, and error conditions.
        """
        if state.get("error"):
            return "error"
        
        classification = state.get("classification")
        confidence = state.get("confidence")
        
        # Route based on classification and confidence
        if confidence == "LOW" or classification == "CLARIFICATION_NEEDED":
            return "clarification"
        elif classification == "SQL":
            return "sql"
        elif classification == "VECTOR":
            return "vector"
        elif classification == "HYBRID":
            return "hybrid"
        else:
            # Default to SQL for unrecognized classifications
            logger.warning(f"Unknown classification '{classification}', defaulting to SQL")
            return "sql"
    
    async def close(self) -> None:
        """Clean up agent resources."""
        try:
            if self._sql_tool:
                await self._sql_tool.close()
            if self._vector_tool and hasattr(self._vector_tool, 'close'):
                await self._vector_tool.close()
                
            logger.info(
                f"RAG agent closed. Processed {self._query_count} queries "
                f"with {self._error_count} errors."
            )
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")


# Factory function for easy agent creation
async def create_rag_agent(config: Optional[AgentConfig] = None) -> RAGAgent:
    """
    Create and initialize a RAG agent.
    
    Args:
        config: Optional agent configuration
        
    Returns:
        Initialized RAG agent ready for use
    """
    agent = RAGAgent(config)
    await agent.initialize()
    return agent
