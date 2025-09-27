"""
Configuration management for ZauriScore using Pydantic settings.

This module provides a hierarchical configuration system with environment variable
support, type validation, and environment-specific overrides.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field, validator, root_validator, DirectoryPath
import logging

def load_environment() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT: str = "development"
    TESTING: str = "testing"
    PRODUCTION: str = "production"
    
    @classmethod
    def is_production(cls) -> bool:
        return cls._get_current() == cls.PRODUCTION
    
    @classmethod
    def is_development(cls) -> bool:
        return cls._get_current() == cls.DEVELOPMENT
    
    @classmethod
    def _get_current(cls):
        # This will be set by the Settings class
        return os.getenv('ENVIRONMENT', cls.DEVELOPMENT)

class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class APISettings(BaseSettings):
    """API configuration settings.
    
    Environment Variables:
        ETHERSCAN_API_KEY: str - Required API key for Etherscan
        REQUEST_TIMEOUT: int - HTTP request timeout in seconds (default: 30)
        MAX_RETRIES: int - Maximum retry attempts for API calls (default: 3)
    """
    ETHERSCAN_API_KEY: str = Field(
        ...,
        env="ETHERSCAN_API_KEY",  # Fixed typo from ETHERSCON_API_KEY
        description="Etherscan API key for fetching contract data",
        min_length=32,
        example="YourEtherscanAPIKeyHere12345"
    )
    REQUEST_TIMEOUT: int = Field(
        30,
        env="REQUEST_TIMEOUT",
        description="Timeout for HTTP requests in seconds",
        gt=0,
    )
    MAX_RETRIES: int = Field(
        3,
        env="MAX_RETRIES",
        description="Maximum number of retries for failed requests",
        ge=0,
    )

class PathSettings(BaseSettings):
    """Filesystem path configuration."""
    BASE_DIR: DirectoryPath = Field(
        Path(__file__).parent.parent.parent.parent,
        description="Base project directory",
    )
    REPORTS_DIR: DirectoryPath = Field(
        "reports",
        env="REPORTS_DIR",
        description="Directory for storing analysis reports",
    )
    CACHE_DIR: DirectoryPath = Field(
        ".cache",
        env="CACHE_DIR",
        description="Directory for caching data",
    )
    
    @validator('REPORTS_DIR', 'CACHE_DIR', pre=True)
    def resolve_paths(cls, v: str, values: Dict[str, Any], **kwargs) -> Path:
        """Resolve relative paths relative to BASE_DIR."""
        base_dir = values.get('BASE_DIR', Path.cwd())
        return base_dir / v

class AnalysisSettings(BaseSettings):
    """Analysis configuration."""
    MAX_CONTRACT_SIZE: int = Field(
        24_576,  # 24KB
        env="MAX_CONTRACT_SIZE",
        description="Maximum contract size in bytes",
        gt=0,
    )
    ENABLE_HEURISTICS: bool = Field(
        True,
        env="ENABLE_HEURISTICS",
        description="Enable heuristic analysis",
    )
    ENABLE_ML: bool = Field(
        True,
        env="ENABLE_ML",
        description="Enable machine learning analysis",
    )

class LoggingSettings(BaseSettings):
    """Logging configuration."""
    LEVEL: LogLevel = Field(
        LogLevel.INFO,
        env="LOG_LEVEL",
        description="Logging level",
    )
    FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    JSON_LOGS: bool = Field(
        False,
        env="JSON_LOGS",
        description="Use JSON format for logs",
    )

class CORSConfig(BaseSettings):
    """CORS configuration."""
    ORIGINS: List[str] = Field(
        ["*"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins"
    )
    METHODS: List[str] = Field(
        ["GET", "POST"],
        env="CORS_METHODS",
        description="Allowed HTTP methods"
    )
    HEADERS: List[str] = Field(
        ["*"],
        env="CORS_HEADERS",
        description="Allowed HTTP headers"
    )
    
    @validator('ORIGINS', 'METHODS', 'HEADERS', pre=True)
    def split_string(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

class Settings(BaseSettings):
    """Main application settings."""
    
    # Core settings
    ENVIRONMENT: Environment = Field(
        Environment.DEVELOPMENT,
        env="ENVIRONMENT",
        description="Application environment",
    )
    DEBUG: bool = Field(
        False,
        env="DEBUG",
        description="Enable debug mode",
    )
    
    # Sub-configurations
    API: APISettings = Field(default_factory=APISettings)
    PATHS: PathSettings = Field(default_factory=PathSettings)
    ANALYSIS: AnalysisSettings = Field(default_factory=AnalysisSettings)
    LOGGING: LoggingSettings = Field(default_factory=LoggingSettings)
    CORS: CORSConfig = Field(default_factory=CORSConfig)
    
    class Config:
        env_nested_delimiter = "__"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            load_environment()  # Load .env when settings are initialized
            return env_settings, init_settings, file_secret_settings
    
    @validator('ENVIRONMENT', pre=True)
    def validate_environment(cls, v):
        try:
            return Environment(v.lower())
        except ValueError:
            raise ValueError(
                f"Invalid environment '{v}'. "
                f"Must be one of: {', '.join(e.value for e in Environment)}"
            )
    
    @root_validator
    def validate_production_settings(cls, values):
        if values.get('ENVIRONMENT') == Environment.PRODUCTION and values.get('DEBUG'):
            raise ValueError("Debug mode cannot be enabled in production")
        return values
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings instance with environment variable overrides."""
        return cls()

# Global settings instance
settings = Settings.from_env()

def get_config() -> Dict[str, Any]:
    """Get configuration as a dictionary (for backward compatibility)."""
    return settings.dict()

def setup_logging(settings: Settings) -> None:
    """Configure logging based on settings."""
    log_config = {
        'level': settings.LOGGING.LEVEL.value,
        'format': settings.LOGGING.FORMAT,
    }
    
    if settings.LOGGING.JSON_LOGS:
        log_config['format'] = '%(message)s'
        log_config['handlers'] = [{
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }]
    
    logging.basicConfig(**log_config)

# Initialize settings and logging
settings = Settings.from_env()
setup_logging(settings)

# Ensure required directories exist
settings.PATHS.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
settings.PATHS.CACHE_DIR.mkdir(parents=True, exist_ok=True)