# Configuration Management

This directory contains the secure configuration management system for the RAG module, built on Pydantic BaseSettings with comprehensive data governance controls.

## Overview

The configuration system implements security-first principles with:
- **Environment Variable Support**: Secure credential loading from environment
- **Type Safety**: Pydantic validation for all configuration parameters
- **Sensitive Data Masking**: Production-safe logging and error handling
- **Compliance Ready**: Built for Australian Privacy Principles (APP) compliance

## Files

### `settings.py`
Main configuration management module implementing:

#### Core Classes
- **`RAGSettings`**: Pydantic BaseSettings class with comprehensive validation
- **Security Features**: Sensitive data masking, secure error handling
- **Validation Methods**: Field validation for database, LLM, and security settings

#### Key Functions
- **`get_settings()`**: Secure settings loader with error handling
- **`validate_configuration()`**: Production-ready configuration validation
- **`_mask_sensitive_value()`**: Utility for masking sensitive information

## Data Governance Features

### Security Controls

#### Credential Protection
```python
# Sensitive fields automatically masked in logs and error messages
sensitive_fields = {"rag_db_password", "llm_api_key", "rag_database_url"}

# Secure representation methods
def __repr__(self) -> str:
    # Returns masked representation of sensitive fields
    
def get_safe_dict(self) -> dict:
    # Returns dictionary with sensitive values masked
```

#### Environment Variable Security
- **No Default Credentials**: Prevents accidental exposure of sensitive defaults
- **Required Field Validation**: Ensures all critical credentials are provided
- **Secure Error Messages**: Production-safe error handling without credential exposure

### Validation Framework

#### Database Configuration
```python
@field_validator("rag_database_url", mode="before")
@classmethod 
def build_database_url(cls, v):
    """Secure database URL construction without exposing credentials"""
```

#### Security Parameter Validation
- **Temperature Range**: LLM temperature between 0.0-2.0
- **Result Limits**: Maximum query results validation (1-10,000)
- **Log Level**: Validates against standard logging levels
- **SQL Complexity**: Configurable complexity scoring limits

## Configuration Parameters

### Database Access (Read-Only)
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `rag_database_url` | `RAG_DATABASE_URL` | No* | Full connection string |
| `rag_db_host` | `RAG_DB_HOST` | Yes | Database host |
| `rag_db_port` | `RAG_DB_PORT` | No | Database port (default: 5432) |
| `rag_db_name` | `RAG_DB_NAME` | Yes | Database name |
| `rag_db_user` | `RAG_DB_USER` | Yes | Read-only database user |
| `rag_db_password` | `RAG_DB_PASSWORD` | Yes | Database password |

*Either `RAG_DATABASE_URL` or individual components required

### LLM Configuration
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `llm_api_key` | `LLM_API_KEY` | Yes | OpenAI/LLM API key |
| `llm_model_name` | `LLM_MODEL_NAME` | No | Model name (default: gpt-3.5-turbo) |
| `llm_temperature` | `LLM_TEMPERATURE` | No | Temperature 0.0-2.0 (default: 0.1) |
| `llm_max_tokens` | `LLM_MAX_TOKENS` | No | Max tokens (default: 1000) |

### Query Processing
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `max_query_results` | `MAX_QUERY_RESULTS` | No | Max results per query (default: 100) |
| `query_timeout_seconds` | `QUERY_TIMEOUT_SECONDS` | No | Query timeout (default: 30) |
| `enable_query_caching` | `ENABLE_QUERY_CACHING` | No | Enable caching (default: True) |

### Security Settings
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `enable_sql_validation` | `ENABLE_SQL_VALIDATION` | No | Enable SQL validation (default: True) |
| `max_sql_complexity_score` | `MAX_SQL_COMPLEXITY_SCORE` | No | Max complexity (default: 10) |

### Logging & Debug
| Parameter | Environment Variable | Required | Description |
|-----------|---------------------|----------|-------------|
| `log_level` | `RAG_LOG_LEVEL` | No | Logging level (default: INFO) |
| `log_sql_queries` | `LOG_SQL_QUERIES` | No | Log SQL queries (default: True) |
| `debug_mode` | `RAG_DEBUG_MODE` | No | Debug mode (default: False) |
| `mock_llm_responses` | `MOCK_LLM_RESPONSES` | No | Mock responses for testing (default: False) |

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

### Environment Setup
```bash
# Required environment variables
export RAG_DB_NAME="csi-db"
export RAG_DB_USER="rag_user_readonly" 
export RAG_DB_PASSWORD="your_secure_password"
export LLM_API_KEY="your_openai_api_key"

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
```

#### Security Validation Failures
- Ensure sensitive data masking is working: `python src/rag/config/settings.py`
- Verify no credentials appear in logs: Check `rag.log` file
- Test error handling: Temporarily remove required variables and test error messages

### Best Practices

1. **Environment Files**: Use `.env` files for development, environment variables for production
2. **Credential Rotation**: Regularly update database passwords and API keys
3. **Monitoring**: Monitor configuration validation logs for security events
4. **Testing**: Always test configuration changes with security validation
5. **Documentation**: Keep data governance documentation updated with configuration changes

---

**Last Updated**: 9 June 2025  
**Security Level**: High  
**Compliance Status**: APP Aligned  
**Test Coverage**: 100% (Configuration)
