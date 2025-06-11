#!/usr/bin/env python3
"""
RAG Module Configuration Management

Pydantic-based configuration system for the RAG module with environment
variable support, validation, and type safety.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """
    RAG module configuration using Pydantic BaseSettings.
    
    Automatically loads from environment variables and .env file.
    Provides type validation and default values.
    """
    
    # Database Configuration (Read-only RAG access)
    rag_database_url: Optional[str] = Field(
        default=None, 
        alias="RAG_DATABASE_URL"
    )
    rag_db_host: str = Field(
        default="localhost", 
        alias="RAG_DB_HOST"
    )
    rag_db_port: int = Field(
        default=5432, 
        alias="RAG_DB_PORT"
    )
    rag_db_name: str = Field(
        ..., 
        alias="RAG_DB_NAME"
    )
    rag_db_user: str = Field(
        ..., 
        alias="RAG_DB_USER"
    )
    rag_db_password: str = Field(
        ..., 
        alias="RAG_DB_PASSWORD"
    )
    
    # LLM Configuration
    llm_api_key: str = Field(
        ..., 
        description="API key for LLM provider (OpenAI, etc.)",
        alias="LLM_API_KEY"
    )
    llm_model_name: str = Field(
        default="gpt-3.5-turbo", 
        description="LLM model name",
        alias="LLM_MODEL_NAME"
    )
    llm_temperature: float = Field(
        default=0.1, 
        description="LLM temperature for SQL generation",
        alias="LLM_TEMPERATURE"
    )
    llm_max_tokens: int = Field(
        default=1000, 
        description="Maximum tokens for LLM responses",
        alias="LLM_MAX_TOKENS"
    )
    
    # Query Processing Configuration
    max_query_results: int = Field(
        default=100, 
        description="Maximum number of results per query",
        alias="MAX_QUERY_RESULTS"
    )
    query_timeout_seconds: int = Field(
        default=30, 
        description="Query timeout in seconds",
        alias="QUERY_TIMEOUT_SECONDS"
    )
    enable_query_caching: bool = Field(
        default=True, 
        description="Enable query result caching",
        alias="ENABLE_QUERY_CACHING"
    )
    
    # Security Configuration
    enable_sql_validation: bool = Field(
        default=True, 
        description="Enable SQL validation and sanitization",
        alias="ENABLE_SQL_VALIDATION"
    )
    max_sql_complexity_score: int = Field(
        default=10, 
        description="Maximum allowed SQL complexity score",
        alias="MAX_SQL_COMPLEXITY_SCORE"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO", 
        description="Logging level",
        alias="RAG_LOG_LEVEL"
    )
    log_sql_queries: bool = Field(
        default=True, 
        description="Log generated SQL queries for audit",
        alias="LOG_SQL_QUERIES"
    )
    
    # Development/Testing Configuration
    debug_mode: bool = Field(
        default=False, 
        description="Enable debug mode",
        alias="RAG_DEBUG_MODE"
    )
    mock_llm_responses: bool = Field(
        default=False, 
        description="Use mock LLM responses for testing",
        alias="MOCK_LLM_RESPONSES"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def model_post_init(self, __context) -> None:
        """Post-initialisation to build database URL if not provided."""
        if not self.rag_database_url:
            # Build from components
            self.rag_database_url = f"postgresql://{self.rag_db_user}:{self.rag_db_password}@{self.rag_db_host}:{self.rag_db_port}/{self.rag_db_name}"
    
    def __repr__(self) -> str:
        """Secure string representation that masks sensitive fields."""
        sensitive_fields = {"rag_db_password", "llm_api_key", "rag_database_url"}
        
        field_strs = []
        for field_name, field_value in self.__dict__.items():
            if field_name in sensitive_fields:
                masked_value = "*" * 8 if field_value else "None"
                field_strs.append(f"{field_name}={masked_value}")
            else:
                field_strs.append(f"{field_name}={field_value!r}")
        
        return f"RAGSettings({', '.join(field_strs)})"
    
    def get_database_uri(self) -> str:
        """
        Get the database URI for connections.
        
        Returns:
            str: PostgreSQL connection URI
        """
        if self.rag_database_url:
            return self.rag_database_url
        
        return f"postgresql://{self.rag_db_user}:{self.rag_db_password}@{self.rag_db_host}:{self.rag_db_port}/{self.rag_db_name}"
    
    def get_safe_dict(self) -> dict:
        """
        Get a dictionary representation with sensitive fields masked.
        
        Returns:
            dict: Configuration with sensitive values masked
        """
        sensitive_fields = {"rag_db_password", "llm_api_key", "rag_database_url"}
        
        safe_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_name in sensitive_fields:
                safe_dict[field_name] = "*" * 8 if field_value else None
            else:
                safe_dict[field_name] = field_value
        
        return safe_dict
        
    @field_validator("rag_database_url", mode="before")
    @classmethod 
    def build_database_url(cls, v):
        """
        Build database URL from components if not provided directly.
        """
        if v:
            return v
        return None  # Will be built in model_post_init
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate LLM temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("LLM temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator("max_query_results")
    @classmethod
    def validate_max_results(cls, v):
        """Validate maximum query results."""
        if v <= 0 or v > 10000:
            raise ValueError("Max query results must be between 1 and 10000")
        return v


def get_settings() -> RAGSettings:
    """
    Get RAG settings instance with environment variable loading.
    
    Returns:
        RAGSettings: Configured settings instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        return RAGSettings()
    except Exception as e:
        # Don't expose detailed configuration errors in production
        raise ValueError("Invalid RAG configuration. Please check environment variables and .env file.")


def _mask_sensitive_value(value: str, mask_char: str = "*", show_chars: int = 3) -> str:
    """
    Mask sensitive values for safe logging/display.
    
    Args:
        value: The value to mask
        mask_char: Character to use for masking
        show_chars: Number of characters to show at the end
        
    Returns:
        str: Masked value
    """
    if not value or len(value) <= show_chars:
        return mask_char * 8
    return mask_char * (len(value) - show_chars) + value[-show_chars:]


def validate_configuration() -> bool:
    """
    Validate the current configuration setup.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        Exception: If configuration validation fails
    """
    try:
        settings = get_settings()
        
        # Test database URL format (without exposing credentials)
        if not settings.rag_database_url.startswith("postgresql://"):
            raise ValueError("Invalid database URL format")
        
        # Validate required fields are present (without exposing values)
        required_fields = ["llm_api_key", "rag_db_password"]
        for field in required_fields:
            if not getattr(settings, field, None):
                raise ValueError(f"Required configuration field is missing: {field}")
        
        print("RAG configuration validation successful")
        print(f"   Database: {settings.rag_db_host}:{settings.rag_db_port}/{settings.rag_db_name}")
        print(f"   Database User: {_mask_sensitive_value(settings.rag_db_user)}")
        print(f"   LLM Model: {settings.llm_model_name}")
        print(f"   LLM API Key: {_mask_sensitive_value(settings.llm_api_key)}")
        print(f"   Log Level: {settings.log_level}")
        print(f"   Max Results: {settings.max_query_results}")
        print(f"   Debug Mode: {settings.debug_mode}")
        
        return True
        
    except Exception as e:
        print(f"RAG configuration validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    """Test configuration loading and validation."""
    print("RAG Configuration Validation")
    print("=" * 40)
    
    try:
        validate_configuration()
        print("\nConfiguration is ready for RAG module.")
    except Exception as e:
        print(f"\nConfiguration error: {e}")
        print("Please fix the configuration before proceeding.")
        exit(1)
