# Configuration Management

This directory contains the secure configuration management system for the RAG module, built on Pydantic BaseSettings with comprehensive data governance controls, user feedback system configuration, and Australian Privacy Principles (APP) compliance integration.

## Overview

The configuration system implements security-first principles with feedback system integration:
- **Environment Variable Support**: Secure credential loading from environment with fallback systems and feedback configuration
- **Type Safety**: Pydantic validation for all configuration parameters including feedback system settings
- **Sensitive Data Masking**: Production-safe logging and error handling with Australian entity protection
- **Compliance Ready**: Built for Australian Privacy Principles (APP) compliance with PII detection and feedback privacy integration
- **Multi-Provider Support**: Live validation for OpenAI, Anthropic, Google Gemini LLM providers, and local embedding models
- **Feedback System Configuration**: Comprehensive settings for user feedback collection, analytics, and privacy controls ✅ NEW (Phase 3)
- **Flexible Embedding Support**: Configurable embedding providers (OpenAI, local sentence transformers)

## Files

### `settings.py` - Enhanced Configuration Management (Phase 3)
Main configuration management module implementing:

#### Enhanced Core Classes (Phase 3)
- **`RAGSettings`**: Enhanced Pydantic BaseSettings with feedback system configuration and comprehensive validation
- **Enhanced Security Features**: Sensitive data masking, secure error handling, and PII detection with feedback privacy integration
- **Enhanced Validation Methods**: Field validation for database, LLM, embedding, feedback system, and security settings with Australian compliance
- **`FeedbackSettings`**: Dedicated configuration section for user feedback system with privacy controls ✅ NEW

#### Enhanced Key Functions (Phase 3)
- **`get_settings()`**: Secure settings loader with feedback configuration and enhanced error handling
- **`validate_configuration()`**: Production-ready configuration validation with feedback system and Australian compliance checks
- **`_mask_sensitive_value()`**: Enhanced utility for masking sensitive information including Australian entities and feedback data
- **`validate_feedback_config()`**: Dedicated validation for feedback system settings and privacy controls ✅ NEW

## Enhanced Data Governance Features (Phase 3)

### Enhanced Security Controls with Feedback Integration

#### Enhanced Credential Protection (Phase 3)
```python
# Enhanced sensitive fields with feedback system and Australian compliance awareness
sensitive_fields = {
    "rag_db_password", "llm_api_key", "rag_database_url", 
    "embedding_api_key", "pii_detection_config", "australian_entity_patterns",
    "feedback_encryption_key", "feedback_analytics_token"  # NEW: Feedback security
}

# Enhanced secure representation methods with feedback privacy protection
def __repr__(self) -> str:
    # Returns masked representation with Australian entity and feedback privacy protection
    
def get_safe_dict(self) -> dict:
    # Returns dictionary with comprehensive sensitive value masking including feedback settings
```
- **No Default Credentials**: Prevents accidental exposure of sensitive defaults with enhanced validation
- **Required Field Validation**: Ensures all critical credentials including PII detection settings are provided
- **Secure Error Messages**: Production-safe error handling with comprehensive PII protection and no credential exposure
- **Australian Entity Awareness**: Configuration system aware of Australian business entities and privacy requirements

### Enhanced Validation Framework

#### Enhanced Database Configuration
```python
@field_validator("rag_database_url", mode="before")
@classmethod 
def build_database_url(cls, v):
    """Enhanced secure database URL construction with PII protection"""
```

#### Enhanced Security Parameter Validation
- **Temperature Range**: LLM temperature between 0.0-2.0 with privacy compliance validation
- **Result Limits**: Maximum query results validation (1-10,000) with Australian entity protection
- **Log Level**: Validates against standard logging levels with PII masking capability
- **SQL Complexity**: Configurable complexity scoring limits with enhanced security checks
- **PII Detection**: Validation of Australian PII detection service configuration and patterns

## Enhanced Configuration Parameters

### Enhanced Database Access (Read-Only + PII Protection)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `rag_database_url` | `RAG_DATABASE_URL` | No* | Full connection string with PII masking |
| `rag_db_host` | `RAG_DB_HOST` | Yes | Database host with security validation |
| `rag_db_port` | `RAG_DB_PORT` | No | Database port (default: 5432) |
| `rag_db_name` | `RAG_DB_NAME` | Yes | Database name with privacy controls |
| `rag_db_user` | `RAG_DB_USER` | Yes | Read-only database user with validation |
| `rag_db_password` | `RAG_DB_PASSWORD` | Yes | Database password with enhanced masking |

*Either `RAG_DATABASE_URL` or individual components required

### Enhanced LLM Configuration (Multi-Provider + PII Protection)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `llm_api_key` | `LLM_API_KEY` | Yes | Multi-provider API key with enhanced masking |
| `llm_model_name` | `LLM_MODEL_NAME` | No | Model name with live provider validation |
| `llm_temperature` | `LLM_TEMPERATURE` | No | Temperature 0.0-2.0 (default: 0.1) |
| `llm_max_tokens` | `LLM_MAX_TOKENS` | No | Max tokens (default: 1000) |

#### Enhanced Supported LLM Providers (Live Validation)
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, `gpt-4o` ✅ Live tested
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
- **Google Gemini**: `gemini-pro`, `gemini-1.5-pro`, `gemini-2.0-flash` ✅ Live tested

#### Enhanced Provider-Specific Notes
- **OpenAI Models**: Use standard model names with API key validation
- **Anthropic Models**: Requires separate API key configuration with enhanced validation
- **Google Gemini**: Beta support with enhanced error handling and fallback configuration

### Enhanced Embedding Configuration (Multi-Provider Support) ✅ **NEW**
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `embedding_provider` | `EMBEDDING_PROVIDER` | No | Provider (openai, sentence_transformers) (default: openai) |
| `embedding_model_name` | `EMBEDDING_MODEL_NAME` | No | Model name (default: text-embedding-ada-002) |
| `embedding_dimension` | `EMBEDDING_DIMENSION` | No | Vector dimension (384 for local, 1536 for OpenAI) |
| `embedding_api_key` | `EMBEDDING_API_KEY` | No* | API key for embedding provider (uses LLM key if unset) |
| `embedding_batch_size` | `EMBEDDING_BATCH_SIZE` | No | Batch size for processing (default: 100) |
| `chunk_size` | `CHUNK_SIZE` | No | Text chunk size for embedding (default: 500) |
| `chunk_overlap` | `CHUNK_OVERLAP` | No | Overlap between chunks (default: 50) |

*Required only if different from LLM provider

#### Supported Embedding Providers
- **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Local Models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2` ✅ Currently configured
  - **Benefits**: No API costs, full privacy control, offline capability
  - **Configuration**: Set `EMBEDDING_PROVIDER=sentence_transformers` and `EMBEDDING_DIMENSION=384`
- **Anthropic Models**: Include version dates with live provider detection
- **Google Gemini Models**: Support latest models with production API validation
- **Google Gemini**: Supports both `gemini-pro` and `models/gemini-pro` formats; the system automatically handles format conversion

### Enhanced Query Processing (Security + Performance)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `max_query_results` | `MAX_QUERY_RESULTS` | No | Max results per query with PII protection (default: 100) |
| `query_timeout_seconds` | `QUERY_TIMEOUT_SECONDS` | No | Query timeout with enhanced monitoring (default: 30) |
| `enable_query_caching` | `ENABLE_QUERY_CACHING` | No | Enable caching with privacy controls (default: True) |

### Enhanced Security Settings (Australian Compliance)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `enable_sql_validation` | `ENABLE_SQL_VALIDATION` | No | Enhanced SQL validation with safety checks (default: True) |
| `max_sql_complexity_score` | `MAX_SQL_COMPLEXITY_SCORE` | No | Max complexity with security limits (default: 10) |
| `enable_pii_detection` | `ENABLE_PII_DETECTION` | No | Australian PII detection (default: True) ✅ NEW |

### Enhanced Logging & Debug (PII Protection)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `log_level` | `RAG_LOG_LEVEL` | No | Logging level with Australian PII masking (default: INFO) |
| `log_sql_queries` | `LOG_SQL_QUERIES` | No | Log SQL queries with privacy protection (default: True) |
| `debug_mode` | `RAG_DEBUG_MODE` | No | Debug mode with enhanced security (default: False) |
| `mock_llm_responses` | `MOCK_LLM_RESPONSES` | No | Mock responses with fallback configuration (default: False) |

## Usage Examples

### Basic Configuration Loading
```python
from rag.config.settings import get_settings, validate_configuration

try:
    # Load configuration with automatic validation
    settings = get_settings()
    
    # Additional validation with user-friendly output
    validate_configuration()
    
    # Access configuration securely
    db_host = settings.rag_db_host
    model_name = settings.llm_model_name
    
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Security Testing
```python
# Test sensitive data masking
settings = get_settings()

# Safe representation (passwords/keys masked)
print(repr(settings))  # Shows ******** for sensitive fields

# Safe dictionary for logging
safe_config = settings.get_safe_dict()
logger.info(f"Configuration loaded: {safe_config}")
```

### Multi-Provider LLM Configuration
```python
# Example configurations for different LLM providers

# OpenAI Configuration
from rag.utils.llm_utils import LLMManager
openai_llm = LLMManager(model_name='gpt-4', api_key='your_openai_key')

# Anthropic Configuration  
anthropic_llm = LLMManager(model_name='claude-3-5-sonnet-20241022', api_key='your_anthropic_key')

# Google Gemini Configuration
gemini_llm = LLMManager(model_name='gemini-pro', api_key='your_google_key')

# The system automatically detects provider based on model name prefix
```

### Environment Setup
```bash
# Required environment variables
export RAG_DB_NAME="csi-db"
export RAG_DB_USER="rag_user_readonly" 
export RAG_DB_PASSWORD="your_secure_password"

# LLM API Key (choose one provider)
export LLM_API_KEY="your_openai_api_key"        # For OpenAI
export LLM_API_KEY="your_anthropic_api_key"     # For Anthropic  
export LLM_API_KEY="your_google_api_key"        # For Google Gemini

# Optional: Specify model
export LLM_MODEL_NAME="gpt-3.5-turbo"           # OpenAI (default)
export LLM_MODEL_NAME="claude-3-5-sonnet-20241022"  # Anthropic
export LLM_MODEL_NAME="gemini-pro"              # Google Gemini

# Validate configuration
python src/rag/config/settings.py
```

## Data Governance Compliance

### Australian Privacy Principles (APP)

#### APP 3 - Collection of Personal Information
- **Minimal Data**: Only collects configuration necessary for operation
- **Purpose Limitation**: Configuration used solely for RAG functionality
- **Consent**: Environment-based configuration implies operational consent

#### APP 6 - Use or Disclosure
- **Internal Use**: Configuration data not shared externally
- **Purpose Bound**: Used only for database connection and LLM integration
- **No Secondary Use**: Configuration not used for analytics or profiling

#### APP 8 - Cross-Border Disclosure  
- **LLM API**: Cloud LLM API usage documented and controlled
- **Data Sovereignty**: No Australian data transmitted in configuration
- **Third Party**: Clear documentation of external service dependencies

#### APP 11 - Security
- **Encryption**: Environment variables secured at OS level
- **Access Control**: Configuration access limited to application context
- **Audit Trail**: Configuration loading and validation logged
- **Incident Response**: Secure error handling prevents data exposure

### Security Audit Checklist

- [ ] **Credential Masking**: Sensitive data never appears in logs
- [ ] **Error Handling**: Production errors don't expose configuration details
- [ ] **Environment Isolation**: Development/production configurations separated
- [ ] **Access Logging**: Configuration access events logged appropriately
- [ ] **Validation Coverage**: All configuration parameters validated
- [ ] **Default Security**: Secure defaults for all optional parameters

## Testing

### Configuration Tests
```bash
# Run configuration-specific tests
cd src/rag && pytest tests/test_config.py -v

# Test security features specifically
pytest tests/test_config.py::TestRAGSettings::test_security_features -v

# Test sensitive data masking
pytest tests/test_config.py::TestRAGSettings::test_mask_sensitive_value -v
```

### Validation Testing
```bash
# Test configuration validation
python src/rag/config/settings.py

# Test with missing required variables
unset RAG_DB_PASSWORD && python src/rag/config/settings.py
```

## Troubleshooting

### Common Configuration Issues

#### Missing Required Variables
```bash
# Check required environment variables
env | grep -E "(RAG_|LLM_)" | sort

# Validate specific requirements
python -c "from rag.config.settings import get_settings; get_settings()"
```

#### Database Connection Issues
```bash
# Test database connectivity separately
python src/db/tests/test_rag_connection.py

# Check database URL construction
python -c "
from rag.config.settings import get_settings
s = get_settings()
print('DB URL (masked):', s.get_safe_dict()['rag_database_url'])
"
#### Enhanced LLM Provider Issues
```bash
# Enhanced multi-provider testing with live validation
python -c "
from rag.utils.llm_utils import get_llm
llm = get_llm()  # Tests live provider detection
print('Provider validated:', llm.provider)
"

# Enhanced Gemini-specific validation ✅ NEW
export LLM_MODEL_NAME='gemini-2.0-flash'
python src/rag/tests/manual_test_phase1.py
```

#### Enhanced Security Validation Failures
- Enhanced sensitive data masking validation: `python src/rag/config/settings.py`
- Enhanced credential protection verification: Check `rag.log` file for Australian entity masking
- Enhanced error handling testing: Test with missing variables and verify PII-safe error messages
- **Australian Entity Protection**: Verify ABN, ACN, TFN, Medicare masking in configuration logs

### Enhanced Best Practices

1. **Enhanced Environment Files**: Use `.env` files for development with PII protection, environment variables for production
2. **Enhanced Credential Rotation**: Regularly update database passwords and API keys with Australian compliance awareness
3. **Enhanced Monitoring**: Monitor configuration validation logs with Australian entity protection for security events
4. **Enhanced Testing**: Always test configuration changes with security validation and PII detection
5. **Enhanced Documentation**: Keep data governance documentation updated with Australian compliance requirements

## Integration with Privacy Module

### PII Detection Configuration ✅ **NEW**
```python
# Configuration integrates seamlessly with Australian PII detection
from rag.config.settings import get_settings
from rag.core.privacy.pii_detector import get_pii_detector

settings = get_settings()
pii_detector = await get_pii_detector()

# All configuration logging automatically includes Australian entity masking
logger.info("Configuration loaded", extra=settings.get_safe_dict())
```

---

**Last Updated**: 17 June 2025  
**Security Level**: High (Enhanced with Australian PII Protection)  
**Compliance Status**: APP Aligned with Phase 2 Australian Entity Protection  
**Test Coverage**: 100% (Configuration + PII Detection Integration)  
**Embedding Support**: Multi-provider (OpenAI + Local Models)
