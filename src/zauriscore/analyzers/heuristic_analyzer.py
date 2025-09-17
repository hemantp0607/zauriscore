"""
Heuristic Analyzer for Smart Contract Security Assessment

This module provides heuristic-based analysis of Solidity smart contracts to identify
potential security vulnerabilities and code quality issues. It integrates with Slither
and provides ML-based vulnerability detection.

Author: Zauriscore Team
Date: 2025-09-17
"""

import re
import logging
import subprocess
import json
import os
import shutil
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SLITHER_TIMEOUT = 120  # seconds
SLITHER_INFORMATIONAL_EXIT_CODE = 4294967295  # 0xFFFFFFFF
TEMP_CONTRACTS_DIR = os.path.join(os.getcwd(), "temp_contracts")
ZAURISCORE_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Heuristic scoring weights
HEURISTIC_POINTS = {
    'reentrancy_guard_present': 5,
    'reentrancy_guard_absent': -15,
    'unguarded_call': -20,
    'owner_selfdestruct': -20,
    'access_control_present': 5,
    'access_control_absent': -10,
    'natspec_present': 10,
    'etherscan_verified': 10,
    'no_external_calls': 10,
    'economic_risk_low': 10,
    'economic_risk_medium': -5,
    'economic_risk_high': -15,
    'compiler_vulnerability': -25
}

# ML Dependencies
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.preprocessing import MinMaxScaler
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning('Machine learning dependencies not fully available. Some features will be disabled: %s', str(e))
    ML_DEPENDENCIES_AVAILABLE = False


class MLVulnerabilityWeightCalculator:
    """Analyzes contract code for ML-based vulnerability patterns using BERT embeddings."""

    def __init__(self, model_path: str = 'microsoft/codebert-base') -> None:
        """Initialize the MLVulnerabilityWeightCalculator.
        
        Args:
            model_path: Path to the pre-trained model or model identifier from huggingface.co/models
            
        Raises:
            ImportError: If required ML dependencies are not available
        """
        if not ML_DEPENDENCIES_AVAILABLE:
            error_msg = 'Machine learning dependencies not available'
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            
            self.vulnerability_embeddings = {
                'reentrancy': self._embed_vulnerability_type('reentrancy attack external call state change'),
                'access_control': self._embed_vulnerability_type('unauthorized access control ownership'),
                'selfdestruct': self._embed_vulnerability_type('contract self destruction owner control'),
                'external_call': self._embed_vulnerability_type('unguarded external call low-level call'),
                'economic_manipulation': self._embed_vulnerability_type('price oracle liquidity manipulation')
            }
            logger.info("ML model and tokenizer initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize ML model: %s", str(e))
            raise

    def _embed_vulnerability_type(self, description: str) -> torch.Tensor:
        """Generate embedding for a vulnerability type description.
        
        Args:
            description: Text description of the vulnerability type
            
        Returns:
            torch.Tensor: Embedding vector
        """
        try:
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).flatten()
        except Exception as e:
            logger.error("Error generating embedding: %s", str(e))
            raise

    def calculate_code_vulnerability_similarity(self, contract_code: str) -> Dict[str, float]:
        """Calculate similarity scores between contract code and known vulnerability types.
        
        Args:
            contract_code: Source code of the contract to analyze
            
        Returns:
            Dict[str, float]: Dictionary mapping vulnerability types to similarity scores
        """
        try:
            contract_embedding = self._embed_vulnerability_type(contract_code)
            similarities = {}
            
            for vuln_type, vuln_embedding in self.vulnerability_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(
                    contract_embedding.unsqueeze(0), 
                    vuln_embedding.unsqueeze(0)
                ).item()
                similarities[vuln_type] = similarity
                
            return similarities
        except Exception as e:
            logger.error("Error calculating code vulnerability similarity: %s", str(e))
            return {}

    def assess_economic_risk(self, contract_code: str) -> str:
        """Assess the economic risk level of a contract.
        
        Args:
            contract_code: Source code of the contract to analyze
            
        Returns:
            str: Risk level ('economic_risk_low', 'economic_risk_medium', 'economic_risk_high')
        """
        try:
            similarities = self.calculate_code_vulnerability_similarity(contract_code)
            economic_risk_score = similarities.get('economic_manipulation', 0)
            
            if economic_risk_score < 0.2:
                return 'economic_risk_low'
            elif economic_risk_score < 0.5:
                return 'economic_risk_medium'
            return 'economic_risk_high'
        except Exception as e:
            logger.error("Error assessing economic risk: %s", str(e))
            return 'economic_risk_medium'  # Default to medium on error


def run_slither(contract_code: Union[str, Dict[str, Any]], 
               main_source_unit_name: str = "MainContract.sol", 
               verbose: bool = False) -> Dict[str, Any]:
    """Run Slither static analysis on a smart contract.
    
    Args:
        contract_code: Either a string of Solidity code or a dictionary of source files
        main_source_unit_name: Name of the main contract file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dict containing Slither analysis results
    """
    slither_path = shutil.which("slither")
    if not slither_path:
        error_msg = "Slither executable not found in PATH. Please install Slither: pip install slither-analyzer"
        logger.error(error_msg)
        return {
            "success": False, 
            "error": error_msg, 
            "results": {"detectors": []}, 
            "inheritance_graph_dot": None, 
            "raw_detector_json_output": None, 
            "raw_inheritance_json_output": None
        }

    # Setup temporary directory
    temp_run_dir = setup_temp_directory()
    slither_results = initialize_slither_results()
    slither_target_for_cmd = None

    try:
        # Process contract code and set up analysis environment
        slither_target_for_cmd = process_contract_code(
            contract_code, main_source_unit_name, temp_run_dir
        )
        
        # Run Slither detectors
        detector_results = run_slither_detectors(slither_path, slither_target_for_cmd)
        slither_results.update(detector_results)
        
        # Run Slither inheritance analysis
        inheritance_results = run_slither_inheritance(slither_path, slither_target_for_cmd)
        slither_results.update(inheritance_results)

    except subprocess.TimeoutExpired:
        error_msg = f"Slither execution timed out for {slither_target_for_cmd}"
        logger.error(error_msg)
        slither_results.update({"success": False, "error": error_msg})
    except Exception as e:
        error_msg = f"Error during Slither execution for {slither_target_for_cmd}: {str(e)}"
        logger.exception(error_msg)
        slither_results.update({"success": False, "error": error_msg})
    finally:
        cleanup_temp_directory(temp_run_dir)
    
    return slither_results


# [Rest of the helper functions would follow the same pattern...]


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
    """
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def normalize_score(score: float, min_score: float = -100.0, max_score: float = 50.0) -> float:
    """Normalize a score to a 0-100 scale.
    
    Args:
        score: Raw score to normalize
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        float: Normalized score between 0 and 100
    """
    if max_score <= min_score:
        logger.warning("max_score must be greater than min_score")
        return 50.0  # Default to middle value
    
    normalized = ((score - min_score) / (max_score - min_score)) * 100
    return max(0.0, min(100.0, normalized))  # Clamp between 0 and 100


def test_heuristic_analyzer() -> None:
    """Run basic tests for the heuristic analyzer."""
    import doctest
    import pytest
    
    # Run doctests
    doctest.testmod()
    
    # Run pytest tests
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    if os.path.exists(test_dir):
        pytest.main([test_dir])
    else:
        logger.warning("Test directory not found: %s", test_dir)


if __name__ == "__main__":
    # Configure logging when run directly
    setup_logging()
    logger.info("Starting Heuristic Analyzer")
    
    # Run tests if --test flag is provided
    if "--test" in sys.argv:
        test_heuristic_analyzer()
    else:
        logger.info("Use --test to run tests")