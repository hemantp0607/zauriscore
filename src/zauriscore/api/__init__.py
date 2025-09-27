"""
ZauriScore API Package

This package contains the web API implementation for ZauriScore's smart contract
analysis service. The API provides endpoints for:
- Contract analysis and scoring
- Report generation
- System health checks

Note:
- This __init__ file is intentionally kept minimal to prevent implicit imports
  of web framework components during module loading.
- API routes and handlers are organized in the `routes` subpackage.
"""

from typing import List

# Public API exports
__all__: List[str] = [
    'routes',
    'schemas',
    'middlewares',
]

# Initialize API components
def init_api() -> None:
    """Initialize API components and dependencies."""
    from .routes import router  # Lazy import to avoid circular imports
    return router
