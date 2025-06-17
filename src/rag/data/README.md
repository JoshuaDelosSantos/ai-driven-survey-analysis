# RAG Data Module

This module handles vector embeddings and content processing for the RAG system.

## Overview

The data module provides comprehensive functionality for:
- **Vector Embedding Management**: Async operations for storing and retrieving embeddings
- **Multi-Provider Support**: OpenAI and Sentence Transformers embedding providers
- **Batch Processing**: Efficient handling of large text datasets
- **Metadata Filtering**: Rich search capabilities with metadata-based filtering
- **Model Versioning**: Support for embedding model upgrades and migration

## Components

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

### Sentence Transformers Provider
- Local embedding generation using Hugging Face models
- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- No API costs, runs locally
- Good quality for most use cases

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
- **Content Processor**: Integration with unified text processing pipeline
- **Vector Search Tool**: LangChain tool integration for query routing
- **Sentiment Analysis**: Direct integration with sentiment analysis results
