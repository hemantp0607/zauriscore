"""Health check endpoints."""
from datetime import datetime
from fastapi import APIRouter
from ...core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }
