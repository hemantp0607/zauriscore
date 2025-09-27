"""
Exception hierarchy for ZauriScore.

This module defines all custom exceptions used throughout the codebase.
"""
from typing import Optional, Dict, Any


class ZauriScoreError(Exception):
    """Base exception for all ZauriScore-specific exceptions."""
    pass


class ConfigurationError(ZauriScoreError):
    """Raised for configuration-related errors."""
    pass


class ValidationError(ZauriScoreError):
    """Raised when data validation fails."""
    pass


class APIClientError(ZauriScoreError):
    """Base exception for API client errors."""
    pass


class APITimeoutError(APIClientError):
    """Raised when an API request times out."""
    pass


class APIConnectionError(APIClientError):
    """Raised when there's a connection error."""
    pass


class APISecurityError(APIClientError):
    """Raised when a security check fails."""
    pass


class APIResponseError(APIClientError):
    """Raised when the API returns an error response."""
    
    def __init__(self, status: int, message: str, response_data: Optional[Dict[str, Any]] = None):
        self.status = status
        self.response_data = response_data or {}
        super().__init__(f"API request failed with status {status}: {message}")


class ResourceError(ZauriScoreError):
    """Raised when there's an error accessing a resource."""
    pass


class ResourceNotFoundError(ResourceError):
    """Raised when a requested resource is not found."""
    pass


class AuthenticationError(ZauriScoreError):
    """Raised when authentication fails."""
    pass
