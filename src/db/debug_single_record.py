#!/usr/bin/env python3
"""
Debug script to test embedding generation for a single record.
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data.content_processor import ContentProcessor, ProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_record():
    """Test processing a single evaluation record."""
    print("🔍 Testing single record processing...")
    
    # Create a minimal config
    config = ProcessingConfig(
        text_fields=["general_feedback", "did_experience_issue_detail", "course_application_other"],
        enable_pii_detection=True,
        enable_sentiment_analysis=True,
        chunk_strategy="sentence",
        max_chunk_size=500,
        min_chunk_size=50
    )
    
    try:
        # Initialize the content processor
        processor = ContentProcessor(config)
        await processor.initialize()
        
        print("✅ Content processor initialized")
        
        # Test with response_id 13 (the one that was successful)
        test_response_id = 13
        print(f"📝 Processing response_id {test_response_id}...")
        
        result = await processor.process_single_evaluation(test_response_id)
        
        print(f"🎯 Result: {result}")
        print(f"   Success: {result.success}")
        print(f"   Chunks: {result.chunks_processed}")
        print(f"   Embeddings: {result.embeddings_stored}")
        print(f"   Errors: {result.errors}")
        print(f"   Warnings: {result.warnings}")
        
        if result.field_results:
            for field_name, field_result in result.field_results.items():
                print(f"   Field {field_name}: {field_result}")
        
        await processor.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_record())
