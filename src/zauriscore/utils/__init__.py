"""
ZauriScore Utils Package

This package contains utility functions and helper modules:
- Logging: Structured logging with JSON support
- Report Generator: Generate analysis reports
- Environment Checker: Verify tool dependencies
- Async HTTP Client: For making HTTP requests
- Error Handling: Standardized error handling utilities
"""

# Core exceptions
from ..exceptions import (
    ZauriScoreError,
    ConfigurationError,
    ValidationError,
    APIClientError,
    APITimeoutError,
    APIConnectionError,
    APISecurityError,
    APIResponseError,
    ResourceError,
    ResourceNotFoundError,
    AuthenticationError,
)

# Utilities
from .report_generator import generate_report
from .check_slither_env import check_slither_installation
from .logger import logger, get_logger, setup_logger
from .async_client import AsyncAPIClient
from .error_handling import (
    handle_errors,
    async_handle_errors,
    resource_manager,
    retry_on_failure,
)

__all__ = [
    # Logging
    'logger',
    'get_logger',
    'setup_logger',
    
    # Reporting
    'generate_report',
    
    # Environment
    'check_slither_installation',
    
    # Async HTTP Client
    'AsyncAPIClient',
    
    # Error Handling
    'handle_errors',
    'async_handle_errors',
    'resource_manager',
    'retry_on_failure',
    
    # Exceptions
    'ZauriScoreError',
    'ConfigurationError',
    'ValidationError',
    'APIClientError',
    'APITimeoutError',
    'APIConnectionError',
    'APISecurityError',
    'APIResponseError',
    'ResourceError',
    'ResourceNotFoundError',
    'AuthenticationError',
]
