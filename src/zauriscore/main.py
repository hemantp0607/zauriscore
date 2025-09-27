"""
ZauriScore FastAPI Application

This module initializes and configures the FastAPI application with:
- API routing
- CORS middleware
- Request/response handling
- Documentation setup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import init_api
from .core.config import settings


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Initialize FastAPI with metadata
    app = FastAPI(
        title="ZauriScore API",
        description="AI-powered smart contract vulnerability analyzer",
        version=settings.VERSION,
        debug=settings.DEBUG,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS.ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS.METHODS,
        allow_headers=settings.CORS.HEADERS,
    )
    
    # Initialize and include API routes
    api_router = init_api()
    app.include_router(api_router, prefix=settings.API_V1_STR)
    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("zauriscore.main:app", host="0.0.0.0", port=8000, reload=True)
