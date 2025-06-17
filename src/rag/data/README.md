# RAG Data Module

This module handles vector embeddings and content processing for the RAG system, supporting both cloud and local embedding providers with comprehensive testing infrastructure and unified text processing pipeline.

## Overview

The data module provides comprehensive functionality for:
- **Vector Embedding Management**: Async operations for storing and retrieving embeddings
- **Unified Content Processing**: Complete pipeline from raw text to embedded vectors ✅ **Phase 2 Task 2.3 Complete**
- **Multi-Provider Support**: OpenAI and local Sentence Transformers embedding providers
- **Batch Processing**: Efficient handling of large text datasets
- **Metadata Filtering**: Rich search capabilities with metadata-based filtering
- **Model Versioning**: Support for embedding model upgrades and migration
- **Production Ready**: Fully tested with comprehensive test coverage

## Current Files

### `content_processor.py` ✅ **NEW - Phase 2 Task 2.3**
Complete implementation of the unified text processing and ingestion pipeline with:
- **Five-Stage Pipeline**: Extract → Anonymise → Analyse → Chunk → Store
- **Australian PII Protection**: Mandatory anonymisation using Presidio
- **Sentiment Integration**: In-memory sentiment analysis for each text field
- **Async Architecture**: Non-blocking operations with batch processing
- **Error Resilience**: Comprehensive error handling and recovery
- **Rich Metadata**: Contextual information stored with each embedding

### `embeddings_manager.py`
Complete implementation of the EmbeddingsManager class with:
- **Async Architecture**: All operations use async/await patterns
- **Multi-Provider Support**: OpenAI and Sentence Transformers providers
- **Database Integration**: PostgreSQL with pgvector extension
- **Error Handling**: Comprehensive error handling and logging
- **Australian Context**: Designed for Australian Public Service evaluation data

### `__init__.py`
Module initialisation file for Python package structure.

## Components

### ContentProcessor (`content_processor.py`) **Phase 2 Task 2.3 Complete**

The unified text processing and ingestion pipeline with the following capabilities:

#### Five-Stage Processing Pipeline
1. **Extract**: Retrieve free-text fields from evaluation records
2. **Anonymise**: Mandatory PII detection and anonymisation using Australian-specific patterns
3. **Analyse**: Generate sentiment scores using local RoBERTa model
4. **Chunk**: Split anonymised text into sentence-level chunks for optimal embedding
5. **Store**: Generate embeddings and persist with rich contextual metadata

#### Key Features
- **Async Operations**: All processing operations are async for optimal performance
- **Batch Processing**: Efficient handling of multiple records with configurable batch sizes
- **Error Resilience**: Comprehensive error handling with detailed result reporting
- **Australian PII Protection**: Mandatory anonymisation before any LLM or embedding processing
- **Rich Metadata**: Context preservation including sentiment scores, user levels, agencies
- **Resumption Capability**: Can resume processing from specific points for large datasets

#### Usage Examples

```python
from src.rag.data.content_processor import ContentProcessor, ProcessingConfig

# Initialize with custom configuration
config = ProcessingConfig(
    text_fields=["general_feedback", "did_experience_issue_detail", "course_application_other"],
    batch_size=50,
    enable_pii_detection=True,
    enable_sentiment_analysis=True
)

# Process all evaluation records
async with ContentProcessor(config) as processor:
    results = await processor.process_all_evaluations()
    
    # Get processing statistics
    stats = await processor.get_processing_statistics()
    print(f"Processed {stats['total_processed']} records")
    print(f"Success rate: {stats['success_rate']:.1f}%")

# Process specific records
response_ids = [1, 2, 3, 100, 150]
results = await processor.process_evaluation_records(response_ids)

# Process single record with detailed results
result = await processor.process_single_evaluation(
    response_id=123,
    include_metadata=True
)
```

#### Processing Result Structure
```python
ProcessingResult(
    response_id=123,
    success=True,
    field_results={
        "general_feedback": {
            "success": True,
            "chunks_processed": 3,
            "embeddings_stored": 3,
            "pii_detected": False,
            "sentiment_scores": {"neg": 0.1, "neu": 0.7, "pos": 0.2}
        }
    },
    chunks_processed=5,
    embeddings_stored=5,
    pii_detected=False,
    sentiment_scores={"general_feedback": {"neg": 0.1, "neu": 0.7, "pos": 0.2}},
    processing_time=2.45
)
```

### EmbeddingsManager (`embeddings_manager.py`)

The core class for managing vector embeddings with the following capabilities:

#### Key Features
- **Async Operations**: All database and embedding operations are async for performance
- **Connection Pooling**: Efficient database connection management
- **Error Handling**: Comprehensive error handling and logging
- **Configurable Dimensions**: Support for different embedding model dimensions
- **Batch Processing**: Efficient batch embedding generation and storage

#### Usage Examples

```python
from src.rag.data.embeddings_manager import EmbeddingsManager

# Initialize manager
manager = EmbeddingsManager()
await manager.initialize()

# Store embeddings for evaluation text
embedding_ids = await manager.store_embeddings(
    response_id=123,
    field_name="general_feedback",
    text_chunks=["Great course content", "Could improve delivery"],
    metadata={
        "user_level": 5,
        "agency": "ATO",
        "sentiment_scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
    }
)

# Search for similar content
results = await manager.search_similar(
    query_text="course feedback about delivery",
    field_name="general_feedback",
    metadata_filter={"agency": "ATO"},
    limit=10,
    similarity_threshold=0.7
)

# Get statistics
stats = await manager.get_stats()
print(f"Total embeddings: {stats['total_embeddings']}")

# Clean up
await manager.close()
```

## Embedding Providers

### OpenAI Provider
- Uses OpenAI's `text-embedding-ada-002` (default) or other OpenAI embedding models
- Requires API key (can use same key as LLM or separate embedding key)
- 1536-dimensional vectors (configurable)
- High quality embeddings with API cost

### Sentence Transformers Provider ✅ **Currently Configured**
- Local embedding generation using Hugging Face models
- Current model: `all-MiniLM-L6-v2` (384 dimensions)
- No API costs, runs locally with full privacy control
- Excellent quality for Australian government use cases
- Offline capability for secure environments

## Configuration

All embedding settings are managed through the RAG configuration system:

```python
# Environment variables for OpenAI (High Quality)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-ada-002
EMBEDDING_DIMENSION=1536
EMBEDDING_API_KEY=your_api_key  # optional if using LLM_API_KEY
EMBEDDING_BATCH_SIZE=100
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Environment variables for Local Model (Current Configuration)
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_API_KEY=  # Not required for local models
EMBEDDING_BATCH_SIZE=100
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Database Schema

The `rag_embeddings` table stores vector embeddings with rich metadata and configurable dimensions:

```sql
CREATE TABLE rag_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    response_id INTEGER NOT NULL REFERENCES evaluation(response_id),
    field_name VARCHAR(50) NOT NULL,     -- 'general_feedback', 'did_experience_issue_detail', etc.
    chunk_text TEXT NOT NULL,            -- Anonymised text chunk
    chunk_index INTEGER NOT NULL,        -- Position within original text
    embedding VECTOR(384) NOT NULL,      -- Configurable dimension (384 for all-MiniLM-L6-v2, 1536 for OpenAI)
    model_version VARCHAR(50) NOT NULL,  -- e.g., 'sentence-transformers-all-MiniLM-L6-v2-v1'
    metadata JSONB,                      -- Rich metadata for filtering
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(response_id, field_name, chunk_index)
);
```

**Note**: Vector dimension is automatically configured based on `EMBEDDING_DIMENSION` environment variable when creating the table.

### Metadata Structure

The metadata JSONB field can contain:
```json
{
    "user_level": 5,
    "agency": "ATO",
    "course_type": "Live Learning",
    "sentiment_scores": {
        "positive": 0.8,
        "negative": 0.1,
        "neutral": 0.1
    },
    "chunk_length": 156,
    "original_text_length": 1234
}
```

## Performance Considerations

### Indexing Strategy
- **Vector Index**: `ivfflat` index for cosine similarity search
- **Metadata Index**: GIN index for JSONB metadata filtering
- **Foreign Key Index**: Standard B-tree indexes for joins

### Batch Processing
- **Embedding Generation**: Process texts in batches for API efficiency
- **Database Insertion**: Use transactions for consistency
- **Connection Pooling**: Maintain connection pool for concurrent operations

### Memory Management
- **Async Operations**: Non-blocking I/O for better resource utilization
- **Lazy Loading**: Models loaded only when needed
- **Connection Cleanup**: Proper resource cleanup and connection closing

## Error Handling

### Database Errors
- Connection failures with retry logic
- Transaction rollbacks on errors
- Constraint violation handling (duplicate chunks)

### Embedding Provider Errors
- API rate limiting and retry with exponential backoff
- Model loading failures for local providers
- Dimension mismatch detection and warnings

### Data Validation
- Text chunk validation and cleaning
- Metadata format validation
- Vector dimension verification

## Future Enhancements

### Planned Features
- **Embedding Model Migration**: Tools for upgrading to new embedding models
- **Compression**: Vector compression for storage efficiency
- **Caching**: Embedding result caching for frequently accessed content
- **Analytics**: Embedding quality metrics and performance monitoring

### Integration Points
- **Content Processor**: Integration with unified text processing pipeline ✅ **Phase 2 Task 2.3**
- **Vector Search Tool**: LangChain tool integration for query routing
- **Sentiment Analysis**: Direct integration with sentiment analysis results

## Testing Status

### Comprehensive Test Coverage (23/23 tests passing) ✅ **Updated**
The data module functionality has been thoroughly tested with:
- **ContentProcessor Testing**: Complete pipeline validation with PII protection ✅ **NEW**
- **Text Chunking**: Sentence-level segmentation with configurable strategies ✅ **NEW**
- **Component Integration**: Seamless integration with PII detector, sentiment analyser, embeddings manager ✅ **NEW**
- **Provider Testing**: SentenceTransformerProvider validation
- **Manager Initialisation**: Configuration and database connection testing
- **Storage Operations**: Single field, multiple chunks, and cross-field storage
- **Search Functionality**: Semantic search, metadata filtering, and similarity queries
- **Database Integration**: Schema compatibility and foreign key validation
- **Error Handling**: Edge cases, empty data, and complex metadata scenarios

### Test Environment
- **Local Model**: all-MiniLM-L6-v2 with 384-dimensional vectors
- **Database**: PostgreSQL with pgvector extension
- **Real Data**: Tests use actual Australian Public Service evaluation data structure
- **Security**: Read-only database access with proper permission validation
- **PII Protection**: Australian-specific anonymisation testing ✅ **NEW**

---
**Last Updated**: 17 June 2025 - Phase 2 Task 2.3 Complete ✅
