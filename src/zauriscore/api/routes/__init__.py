"""API routes package."""
from fastapi import APIRouter
from . import health, analysis

# Create main API router
api_router = APIRouter()

# Include route modules
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
