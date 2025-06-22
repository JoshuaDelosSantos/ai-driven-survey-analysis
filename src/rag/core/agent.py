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
from typing import Dict, Any, Optional, List, Literal
from typing_extensions import TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLanguageModel

from .text_to_sql.sql_tool import AsyncSQLTool
from .vector_search.vector_search_tool import VectorSearchTool
from .privacy.pii_detector import AustralianPIIDetector
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
            
            # Add nodes (placeholder implementations for now)
            workflow.add_node("classify_query", self._classify_query_node)
            workflow.add_node("sql_tool", self._sql_tool_node)
            workflow.add_node("vector_search_tool", self._vector_search_tool_node)
            workflow.add_node("synthesis", self._synthesis_node)
            workflow.add_node("clarification", self._clarification_node)
            workflow.add_node("error_handling", self._error_handling_node)
            
            # Set entry point
            workflow.set_entry_point("classify_query")
            
            # Add conditional routing (basic structure for now)
            workflow.add_conditional_edges(
                "classify_query",
                self._route_after_classification,
                {
                    "sql": "sql_tool",
                    "vector": "vector_search_tool",
                    "clarification": "clarification",
                    "error": "error_handling"
                }
            )
            
            # Add edges from tools to synthesis
            workflow.add_edge("sql_tool", "synthesis")
            workflow.add_edge("vector_search_tool", "synthesis")
            
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
        """Placeholder for query classification node."""
        logger.info(f"Classifying query: {state['query'][:50]}...")
        
        # Temporary basic classification for testing
        query_lower = state["query"].lower()
        if any(word in query_lower for word in ["count", "how many", "average", "percentage"]):
            classification = "SQL"
            confidence = "HIGH"
        elif any(word in query_lower for word in ["feedback", "comment", "experience", "opinion"]):
            classification = "VECTOR"
            confidence = "HIGH"
        else:
            classification = "SQL"  # Default to SQL for now
            confidence = "MEDIUM"
        
        return {
            **state,
            "classification": classification,
            "confidence": confidence,
            "classification_reasoning": "Basic keyword-based classification (placeholder)",
            "tools_used": state["tools_used"] + ["classifier"]
        }
    
    async def _sql_tool_node(self, state: AgentState) -> AgentState:
        """Placeholder for SQL tool node."""
        logger.info("Processing query with SQL tool...")
        
        try:
            # This will be implemented with actual SQL tool integration
            result = {
                "query": state["query"],
                "result": "SQL tool placeholder result",
                "success": True
            }
            
            return {
                **state,
                "sql_result": result,
                "tools_used": state["tools_used"] + ["sql"]
            }
            
        except Exception as e:
            logger.error(f"SQL tool failed: {e}")
            return {
                **state,
                "error": f"SQL processing failed: {str(e)}",
                "tools_used": state["tools_used"] + ["sql_failed"]
            }
    
    async def _vector_search_tool_node(self, state: AgentState) -> AgentState:
        """Placeholder for vector search tool node."""
        logger.info("Processing query with vector search tool...")
        
        try:
            # This will be implemented with actual vector search integration
            result = {
                "query": state["query"],
                "results": ["Vector search placeholder result"],
                "success": True
            }
            
            return {
                **state,
                "vector_result": result,
                "tools_used": state["tools_used"] + ["vector"]
            }
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {
                **state,
                "error": f"Vector search failed: {str(e)}",
                "tools_used": state["tools_used"] + ["vector_failed"]
            }
    
    async def _synthesis_node(self, state: AgentState) -> AgentState:
        """Placeholder for answer synthesis node."""
        logger.info("Synthesizing final answer...")
        
        # Combine results from available tools
        answer_parts = []
        sources = []
        
        if state.get("sql_result"):
            answer_parts.append(f"SQL Analysis: {state['sql_result'].get('result', 'No result')}")
            sources.append("Database Analysis")
        
        if state.get("vector_result"):
            answer_parts.append(f"Feedback Search: {state['vector_result'].get('results', ['No results'])[0]}")
            sources.append("User Feedback")
        
        final_answer = "\n\n".join(answer_parts) if answer_parts else "Unable to generate answer"
        
        return {
            **state,
            "final_answer": final_answer,
            "sources": sources,
            "tools_used": state["tools_used"] + ["synthesis"]
        }
    
    async def _clarification_node(self, state: AgentState) -> AgentState:
        """Placeholder for clarification node."""
        logger.info("Query requires clarification...")
        
        clarification_message = (
            f"I need clarification for your query: '{state['query']}'\n\n"
            "Please specify:\n"
            "A) 📊 Statistical summary or numerical breakdown\n"
            "B) 💬 Specific feedback, comments, or experiences\n"
            "C) 📈 Combined analysis with both numbers and feedback\n\n"
            "Type A, B, or C to continue."
        )
        
        return {
            **state,
            "final_answer": clarification_message,
            "requires_clarification": True,
            "tools_used": state["tools_used"] + ["clarification"]
        }
    
    async def _error_handling_node(self, state: AgentState) -> AgentState:
        """Placeholder for error handling node."""
        logger.warning(f"Handling error: {state.get('error', 'Unknown error')}")
        
        error_message = (
            "I encountered an issue processing your query. "
            "Please try rephrasing your question or contact support if the problem persists."
        )
        
        return {
            **state,
            "final_answer": error_message,
            "tools_used": state["tools_used"] + ["error_handler"]
        }
    
    def _route_after_classification(self, state: AgentState) -> str:
        """Route to appropriate node based on classification results."""
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
        else:
            # Default to SQL for unrecognized classifications
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
