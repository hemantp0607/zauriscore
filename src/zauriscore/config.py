"""
Configuration settings for ZauriScore.

This module contains configuration settings for the ZauriScore application,
including model paths, API keys, and other configuration parameters.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent.absolute()

# Model paths
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "model_weights.pth"
TOKENIZER_PATH = MODEL_DIR / "tokenizer"

# Data paths
DATA_DIR = BASE_DIR / "data"
CONTRACTS_DIR = DATA_DIR / "contracts"
REPORTS_DIR = DATA_DIR / "reports"

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, CONTRACTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API settings
API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security settings
API_KEYS = {
    "default": os.getenv("ZAURISCORE_API_KEY", "")
}

# Analysis settings
DEFAULT_ANALYSIS_CONFIG = {
    "enable_heuristics": True,
    "enable_ml": True,
    "confidence_threshold": 0.7,
    "timeout": 300,  # 5 minutes
}

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Export settings as a dictionary for easier access
settings = {
    "BASE_DIR": BASE_DIR,
    "MODEL_DIR": MODEL_DIR,
    "DEFAULT_MODEL_PATH": DEFAULT_MODEL_PATH,
    "TOKENIZER_PATH": TOKENIZER_PATH,
    "DATA_DIR": DATA_DIR,
    "CONTRACTS_DIR": CONTRACTS_DIR,
    "REPORTS_DIR": REPORTS_DIR,
    "API_TIMEOUT": API_TIMEOUT,
    "MAX_RETRIES": MAX_RETRIES,
    "LOG_LEVEL": LOG_LEVEL,
    "LOG_FORMAT": LOG_FORMAT,
    "API_KEYS": API_KEYS,
    "DEFAULT_ANALYSIS_CONFIG": DEFAULT_ANALYSIS_CONFIG,
}

# Make settings available as module-level variables
locals().update(settings)

__all__ = ["settings"] + list(settings.keys())
