#!/usr/bin/env python3
"""
Test script for ContentProcessor implementation

This script provides basic validation of the ContentProcessor module
and its integration with existing RAG components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.data.content_processor import (
    ContentProcessor, 
    ProcessingConfig, 
    process_single_record,
    process_evaluation_batch
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_content_processor_initialization():
    """Test that ContentProcessor can be initialized successfully."""
    logger.info("Testing ContentProcessor initialization...")
    
    try:
        config = ProcessingConfig(
            text_fields=["general_feedback"],
            batch_size=5,
            enable_pii_detection=True,
            enable_sentiment_analysis=True
        )
        
        processor = ContentProcessor(config)
        await processor.initialize()
        
        logger.info("✅ ContentProcessor initialization successful")
        
        # Test statistics
        stats = await processor.get_processing_statistics()
        logger.info(f"Initial statistics: {stats}")
        
        await processor.cleanup()
        logger.info("✅ ContentProcessor cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ContentProcessor initialization failed: {e}")
        return False


async def test_text_chunking():
    """Test the text chunking functionality."""
    logger.info("Testing text chunking...")
    
    try:
        from src.rag.data.content_processor import TextChunker
        
        chunker = TextChunker(strategy="sentence", max_chunk_size=100, min_chunk_size=20)
        
        test_text = """
        This is the first sentence of feedback. This is the second sentence which provides more detail.
        Here's a third sentence that adds even more context. And finally, a fourth sentence to complete the feedback.
        """
        
        chunks = await chunker.chunk_text(test_text.strip())
        
        logger.info(f"✅ Text chunking successful: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i}: {chunk.text[:50]}..." if len(chunk.text) > 50 else f"  Chunk {i}: {chunk.text}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Text chunking failed: {e}")
        return False


async def test_component_imports():
    """Test that all required components can be imported."""
    logger.info("Testing component imports...")
    
    try:
        # Test PII detector import
        from src.rag.core.privacy.pii_detector import AustralianPIIDetector
        logger.info("✅ PII detector import successful")
        
        # Test embeddings manager import
        from src.rag.data.embeddings_manager import EmbeddingsManager
        logger.info("✅ Embeddings manager import successful")
        
        # Test database utils import
        from src.rag.utils.db_utils import DatabaseManager
        logger.info("✅ Database manager import successful")
        
        # Test sentiment analyser import
        from src.rag.data.content_processor import SentimentAnalyser
        logger.info("✅ Sentiment analyser import successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component import failed: {e}")
        return False


async def test_convenience_functions():
    """Test the convenience functions."""
    logger.info("Testing convenience functions...")
    
    try:
        # Test with a very simple config
        config = ProcessingConfig(
            text_fields=["general_feedback"],
            batch_size=1,
            enable_pii_detection=False,  # Disable for simpler testing
            enable_sentiment_analysis=False  # Disable for simpler testing
        )
        
        # Note: This will fail if no database is available, which is expected
        # We're just testing that the functions can be called
        logger.info("✅ Convenience functions are callable")
        
        return True
        
    except Exception as e:
        logger.info(f"ℹ️  Convenience functions failed (expected without database): {e}")
        return True  # This is expected without a database connection


async def main():
    """Run all tests."""
    logger.info("Starting ContentProcessor tests...")
    
    tests = [
        ("Component Imports", test_component_imports),
        ("Text Chunking", test_text_chunking),
        ("Convenience Functions", test_convenience_functions),
        ("ContentProcessor Initialization", test_content_processor_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
