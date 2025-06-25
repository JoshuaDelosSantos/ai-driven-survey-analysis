#!/usr/bin/env python3
"""
Debug the two failed records that actually contain content.
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data.content_processor import ContentProcessor, ProcessingConfig

# Configure logging to see detailed errors
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def debug_failed_records():
    """Debug the specific records that failed despite having content."""
    print("🔍 Debugging failed records with actual content...")
    
    # Create a minimal config
    config = ProcessingConfig(
        text_fields=["general_feedback", "did_experience_issue_detail", "course_application_other"],
        enable_pii_detection=True,
        enable_sentiment_analysis=True,
        chunk_strategy="sentence",
        max_chunk_size=500,
        min_chunk_size=10  
    )
    
    try:
        processor = ContentProcessor(config)
        await processor.initialize()
        
        # Test the two failed records with actual content
        failed_ids = [1, 5]
        
        for response_id in failed_ids:
            print(f"\n📝 Processing response_id {response_id}...")
            
            try:
                result = await processor.process_single_evaluation(response_id)
                
                print(f"🎯 Result for {response_id}: {result.success}")
                print(f"   Chunks: {result.chunks_processed}")
                print(f"   Embeddings: {result.embeddings_stored}")
                print(f"   Errors: {result.errors}")
                print(f"   Warnings: {result.warnings}")
                
                if result.field_results:
                    for field_name, field_result in result.field_results.items():
                        print(f"   Field {field_name}: success={field_result.get('success')}, errors={field_result.get('errors')}")
                
            except Exception as e:
                print(f"❌ Exception processing {response_id}: {e}")
                import traceback
                traceback.print_exc()
        
        await processor.cleanup()
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_failed_records())
