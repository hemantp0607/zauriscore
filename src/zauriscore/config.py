"""Configuration management for ZauriScore."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Config:
    """Centralized configuration for ZauriScore."""
    
    # API Keys
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
    
    # Timeouts
    SLITHER_TIMEOUT: int = int(os.getenv("SLITHER_TIMEOUT", "300"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Paths
    REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", ".cache"))
    
    # Analysis
    MAX_CONTRACT_SIZE: int = int(os.getenv("MAX_CONTRACT_SIZE", "24576"))  # bytes
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[Path] = Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }

# Initialize config
config = Config()
