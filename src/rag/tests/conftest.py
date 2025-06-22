#!/usr/bin/env python3
"""
Shared test fixtures and configuration for Phase 3 RAG testing.

This module provides:
- Environment setup for test execution
- Mock fixtures for external dependencies  
- Shared test data and utilities
- Australian PII detection for privacy testing
- Configuration overrides for testing

IMPORTANT: For tests to run properly, ensure you have a .env file in the project root
with proper configuration values. This file only provides fallback defaults for testing.
"""

import asyncio
import os
import sys
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists for real configuration values
env_file = project_root / '.env'
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

# Set environment variables for testing before importing RAG modules
# These are test defaults - real values should be in .env file
os.environ.setdefault('RAG_DB_NAME', 'test-db')
os.environ.setdefault('RAG_DB_USER', 'test_user')
os.environ.setdefault('RAG_DB_PASSWORD', 'test_password')
os.environ.setdefault('RAG_DB_HOST', 'localhost')
os.environ.setdefault('RAG_DB_PORT', '5432')
os.environ.setdefault('LLM_API_KEY', 'test-api-key-placeholder')
os.environ.setdefault('LLM_MODEL_NAME', 'test-model')
os.environ.setdefault('LLM_TEMPERATURE', '0.1')
os.environ.setdefault('LLM_MAX_TOKENS', '1000')
os.environ.setdefault('EMBEDDING_PROVIDER', 'sentence_transformers')
os.environ.setdefault('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
os.environ.setdefault('EMBEDDING_DIMENSION', '384')
os.environ.setdefault('EMBEDDING_API_KEY', '')
os.environ.setdefault('EMBEDDING_BATCH_SIZE', '100')
os.environ.setdefault('CHUNK_SIZE', '500')
os.environ.setdefault('CHUNK_OVERLAP', '50')
os.environ.setdefault('MAX_QUERY_RESULTS', '100')
os.environ.setdefault('QUERY_TIMEOUT_SECONDS', '30')
os.environ.setdefault('ENABLE_QUERY_CACHING', 'true')
os.environ.setdefault('ENABLE_SQL_VALIDATION', 'true')
os.environ.setdefault('MAX_SQL_COMPLEXITY_SCORE', '10')
os.environ.setdefault('RAG_LOG_LEVEL', 'INFO')
os.environ.setdefault('LOG_SQL_QUERIES', 'true')
os.environ.setdefault('RAG_DEBUG_MODE', 'false')
os.environ.setdefault('MOCK_LLM_RESPONSES', 'true')  # Enable mocking for tests


# Core Test Fixtures

@pytest_asyncio.fixture
async def mock_llm():
    """Mock LLM for testing without API calls."""
    mock = AsyncMock()
    # Default successful classification response
    mock.ainvoke.return_value = MagicMock(
        content='{"classification": "SQL", "confidence": "HIGH", "reasoning": "Statistical query detected"}'
    )
    return mock


@pytest_asyncio.fixture
async def pii_detector():
    """Australian PII detector for privacy testing."""
    try:
        from src.rag.core.privacy.pii_detector import AustralianPIIDetector
        detector = AustralianPIIDetector()
        await detector.initialise()
        return detector
    except ImportError:
        # Return mock if PII detector not available
        mock = AsyncMock()
        mock.detect_pii = AsyncMock(return_value=[])
        mock.anonymise_text = AsyncMock(side_effect=lambda x: x)
        return mock


@pytest_asyncio.fixture
async def rag_agent():
    """Configured RAG agent for testing."""
    try:
        from src.rag.core.agent import RAGAgent
        agent = RAGAgent()
        await agent.initialize()
        return agent
    except ImportError:
        # Return mock if agent not available
        return AsyncMock()


@pytest_asyncio.fixture
async def terminal_app():
    """Terminal application instance for testing."""
    try:
        from src.rag.interfaces.terminal_app import RAGTerminalApp
        app = RAGTerminalApp(enable_agent=True)
        await app.initialize()
        return app
    except ImportError:
        # Return mock if terminal app not available
        return AsyncMock()


# Sample Data Fixtures

@pytest.fixture
def sample_queries():
    """Sample queries for different classification types."""
    return {
        'sql': [
            "How many users completed courses?",
            "Show me the attendance breakdown by agency",
            "What's the average completion rate?",
            "Count users by level"
        ],
        'vector': [
            "What feedback did users give about virtual learning?",
            "How do people feel about the platform?",
            "What experiences did users share?",
            "Summarise user opinions"
        ],
        'hybrid': [
            "Analyse satisfaction trends with supporting feedback",
            "Compare completion rates with user sentiment",
            "Show statistics with relevant user comments",
            "Breakdown performance with qualitative insights"
        ],
        'unclear': [
            "Tell me about stuff",
            "What happened?",
            "Analysis please",
            "Show me things"
        ]
    }


@pytest.fixture
def sample_sql_result():
    """Sample SQL query result for testing."""
    return {
        'success': True,
        'data': [
            {'agency': 'Agency A', 'completed': 45, 'total': 50, 'percentage': 90.0},
            {'agency': 'Agency B', 'completed': 38, 'total': 42, 'percentage': 90.5},
            {'agency': 'Agency C', 'completed': 29, 'total': 35, 'percentage': 82.9}
        ],
        'query': "SELECT agency, COUNT(*) as completed FROM users WHERE status='completed' GROUP BY agency",
        'execution_time': 0.15,
        'row_count': 3
    }


@pytest.fixture
def sample_vector_result():
    """Sample vector search result for testing."""
    return {
        'success': True,
        'results': [
            {
                'content': 'The virtual learning platform was very user-friendly and engaging.',
                'metadata': {'source': 'evaluation_1', 'user_level': 'beginner'},
                'similarity_score': 0.92
            },
            {
                'content': 'I found the online course format convenient and well-structured.',
                'metadata': {'source': 'evaluation_2', 'user_level': 'intermediate'},
                'similarity_score': 0.89
            },
            {
                'content': 'The digital learning tools helped me understand the material better.',
                'metadata': {'source': 'evaluation_3', 'user_level': 'advanced'},
                'similarity_score': 0.87
            }
        ],
        'query': 'virtual learning platform experience',
        'total_results': 3,
        'search_time': 0.08
    }


@pytest.fixture
def sample_pii_text():
    """Sample text with Australian PII for testing."""
    return {
        'with_pii': "John Smith from Sydney called about his Medicare number 1234567890A and ABN 12345678901.",
        'without_pii': "A user from Sydney called about their Medicare number [MEDICARE_NUMBER] and ABN [ABN].",
        'phone': "Call me on 0412 345 678 or email john.smith@example.com",
        'tfn': "My tax file number is 123 456 789 for processing"
    }


# Mock External Services

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=[])
    mock_db.fetch = AsyncMock(return_value=[])
    mock_db.fetchrow = AsyncMock(return_value=None)
    return mock_db


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock_service = AsyncMock()
    mock_service.embed_query = AsyncMock(return_value=[0.1] * 384)  # 384-dim embedding
    mock_service.embed_documents = AsyncMock(return_value=[[0.1] * 384])
    return mock_service


# Configuration Fixtures

@pytest.fixture
def test_config():
    """Test configuration overrides."""
    return {
        'llm_model_name': 'gemini-2.0-flash',
        'max_query_results': 50,
        'query_timeout_seconds': 15,
        'mock_llm_responses': True,
        'rag_debug_mode': True
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for all tests."""
    # Ensure test environment variables are set
    original_env = dict(os.environ)
    
    # Override for testing
    os.environ['MOCK_LLM_RESPONSES'] = 'true'
    os.environ['RAG_DEBUG_MODE'] = 'true'
    os.environ['QUERY_TIMEOUT_SECONDS'] = '5'  # Shorter timeout for tests
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Utility Functions

def assert_pii_safe(text: str) -> bool:
    """Assert that text contains no Australian PII."""
    import re
    
    # Check for common Australian PII patterns
    patterns = [
        r'\b\d{4}\s?\d{3}\s?\d{3}\b',  # TFN pattern
        r'\b\d{11}\b',  # ABN pattern
        r'\b\d{10}[A-Z]\b',  # Medicare pattern
        r'\b04\d{2}\s?\d{3}\s?\d{3}\b',  # Mobile phone pattern
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'  # Email pattern
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return False
    return True


def create_mock_agent_state(**overrides) -> Dict[str, Any]:
    """Create a mock agent state for testing."""
    default_state = {
        'query': 'Test query',
        'session_id': 'test-session-123',
        'classification': None,
        'confidence': None,
        'classification_reasoning': None,
        'sql_result': None,
        'vector_result': None,
        'final_answer': None,
        'sources': [],
        'error': None,
        'retry_count': 0,
        'requires_clarification': False,
        'user_feedback': None,
        'processing_time': None,
        'tools_used': []
    }
    default_state.update(overrides)
    return default_state


# Pytest Configuration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "privacy: mark test as a privacy compliance test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test name patterns
        if "privacy" in item.name.lower() or "pii" in item.name.lower():
            item.add_marker(pytest.mark.privacy)
        
        if "performance" in item.name.lower() or "speed" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        if "real_llm" in item.name.lower() or "real_" in item.name.lower():
            item.add_marker(pytest.mark.slow)
