"""Heuristic Analyzer for Smart Contract Security Assessment.

This module provides heuristic-based analysis of Solidity smart contracts to identify
potential security vulnerabilities and code quality issues. It integrates with Slither
and provides ML-based vulnerability detection.
"""
# Standard library imports
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import (
    Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, TypeVar, TypedDict, Union
)

# Third-party imports
import numpy as np
from slither import Slither
from slither.core.declarations import (
    Contract, Event, Enum, Function, Modifier, SolidityFunction,
    SolidityVariableComposed, Structure
)
from slither.core.expressions.expression import Expression
from slither.core.variables.state_variable import StateVariable
from slither.slithir.operations import (
    EventCall, HighLevelCall, InternalCall, InternalDynamicCall,
    LowLevelCall, Send, SolidityCall, Transfer
)

# Local imports
from .base_analyzer import BaseAnalyzer
from ..utils.logger import get_logger

# Type variable for generic type hints
T = TypeVar('T')

# Initialize logger
logger = get_logger(__name__)

# Constants
SLITHER_TIMEOUT = 120  # seconds
SLITHER_INFORMATIONAL_EXIT_CODE = 4294967295  # 0xFFFFFFFF
DEFAULT_TEMP_DIR = "temp_contracts"

# Configure paths
ZAURISCORE_BASE_DIR = Path(__file__).parent.parent.parent
TEMP_CONTRACTS_DIR = Path(os.getenv('TEMP_CONTRACTS_DIR', ZAURISCORE_BASE_DIR / DEFAULT_TEMP_DIR))

# Configuration and Types
@dataclass
class AnalyzerConfig:
    """Configuration for the heuristic analyzer."""
    # Default values
    DEFAULT_TIMEOUT: ClassVar[int] = 120
    DEFAULT_TEMP_DIR: ClassVar[str] = "temp_contracts"
    
    # Configurable parameters
    timeout: int = DEFAULT_TIMEOUT
    temp_dir: str = DEFAULT_TEMP_DIR
    
    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> 'AnalyzerConfig':
        """Create config from dictionary with validation."""
        if config is None:
            return cls()
        return cls(
            timeout=int(config.get('timeout', cls.DEFAULT_TIMEOUT)),
            temp_dir=str(config.get('temp_dir', cls.DEFAULT_TEMP_DIR))
        )

# Heuristic scoring weights with type hints
class HeuristicWeights(TypedDict):
    """Type definition for heuristic scoring weights."""
    reentrancy_guard_present: int
    reentrancy_guard_absent: int
    unguarded_call: int
    owner_selfdestruct: int
    access_control_present: int
    access_control_absent: int
    natspec_present: int
    etherscan_verified: int
    no_external_calls: int
    economic_risk_low: int
    economic_risk_medium: int
    economic_risk_high: int
    compiler_vulnerability: int

# Default heuristic scoring weights
HEURISTIC_POINTS: HeuristicWeights = {
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

# Type aliases
EconomicRisk = Literal['economic_risk_low', 'economic_risk_medium', 'economic_risk_high']

# ML Dependencies
try:
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from transformers import AutoModel, AutoTokenizer
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(
        'Machine learning dependencies not fully available. '
        'Some features will be disabled: %s', str(e)
    )
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


class SlitherAnalysisError(Exception):
    """Base exception for Slither analysis errors."""
    pass

class SlitherTimeoutError(SlitherAnalysisError):
    """Raised when Slither analysis times out."""
    pass

def cleanup_temp_directory(temp_dir: Path) -> None:
    """Safely cleanup temporary directory with better error handling."""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    except PermissionError as e:
        logger.error(f"Permission denied cleaning up {temp_dir}: {e}")
        # Attempt to change permissions and retry
        try:
            os.chmod(temp_dir, 0o777)
            shutil.rmtree(temp_dir)
        except Exception as retry_error:
            logger.critical(f"Failed to cleanup {temp_dir} after retry: {retry_error}")
    except Exception as e:
        logger.error(f"Unexpected error cleaning up {temp_dir}: {e}")

def run_slither(
    contract_code: Union[str, Dict[str, Any]], 
    main_source_unit_name: str = "MainContract.sol", 
    verbose: bool = False
) -> Dict[str, Any]:
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

    temp_run_dir = None
    try:
        # Set up temporary directory
        temp_run_dir = setup_temp_directory()
        slither_results = initialize_slither_results()

        # Process contract code and get path to the main contract file
        slither_target_for_cmd = process_contract_code(
            contract_code, main_source_unit_name, temp_run_dir
        )

        # Run Slither detectors with timeout
        try:
            detector_results = run_slither_detectors(
                slither_path="slither",
                target=slither_target_for_cmd
            )
            slither_results.update(detector_results)
        except subprocess.TimeoutExpired as te:
            error_msg = f"Slither detector analysis timed out after {te.timeout} seconds"
            logger.error(error_msg)
            slither_results.update({
                "success": False,
                "error": error_msg,
                "type": "timeout"
            })
        except subprocess.CalledProcessError as cpe:
            error_msg = f"Slither detector execution failed with code {cpe.returncode}: {cpe.stderr}"
            logger.error(error_msg)
            slither_results.update({
                "success": False,
                "error": error_msg,
                "type": "detector_error"
            })

        # Run Slither inheritance analysis if detectors succeeded
        if slither_results.get("success", True):
            try:
                inheritance_results = run_slither_inheritance(
                    slither_path="slither",
                    target=slither_target_for_cmd
                )
                slither_results.update(inheritance_results)
            except subprocess.TimeoutExpired as te:
                error_msg = f"Slither inheritance analysis timed out after {te.timeout} seconds"
                logger.error(error_msg)
                slither_results.update({
                    "success": False,
                    "error": error_msg,
                    "type": "timeout"
                })
            except subprocess.CalledProcessError as cpe:
                error_msg = f"Slither inheritance analysis failed with code {cpe.returncode}: {cpe.stderr}"
                logger.error(error_msg)
                slither_results.update({
                    "success": False,
                    "error": error_msg,
                    "type": "inheritance_error"
                })

        return slither_results

    except Exception as e:
        error_msg = f"Unexpected error during Slither analysis: {str(e)}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "type": "unexpected_error"
        }

    finally:
        # Clean up temporary directory
        if temp_run_dir is not None and temp_run_dir.exists():
            try:
                cleanup_temp_directory(temp_run_dir)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary directory: {cleanup_error}")
    return slither_results


def setup_temp_directory() -> Path:
    """Create a temporary directory for Slither analysis.
    
    Returns:
        Path: Path to the created temporary directory
    """
    temp_dir = Path(TEMP_CONTRACTS_DIR) / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def initialize_slither_results() -> Dict[str, Any]:
    """Initialize a dictionary to store Slither analysis results.
    
    Returns:
        Dict containing initialized result structure
    """
    return {
        "success": True,
        "error": None,
        "results": {"detectors": []},
        "inheritance_graph_dot": None,
        "raw_detector_json_output": None,
        "raw_inheritance_json_output": None
    }


def process_contract_code(
    contract_code: Union[str, Dict[str, Any]],
    main_source_unit_name: str,
    temp_dir: Path
) -> str:
    """Process contract code and save to a temporary file.
    
    Args:
        contract_code: Contract code as string or dictionary of files
        main_source_unit_name: Name of the main contract file
        temp_dir: Directory to save the contract files
        
    Returns:
        str: Path to the main contract file
    """
    if isinstance(contract_code, dict):
        # Handle multiple source files
        for filename, content in contract_code.items():
            file_path = temp_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
        return str(temp_dir / main_source_unit_name)
    else:
        # Handle single file
        file_path = temp_dir / main_source_unit_name
        file_path.write_text(contract_code, encoding='utf-8')
        return str(file_path)


def run_slither_detectors(slither_path: str, target: str) -> Dict[str, Any]:
    """Run Slither vulnerability detectors on the target.
    
    Args:
        slither_path: Path to the Slither executable
        target: Path to the target contract file or directory
        
    Returns:
        Dict containing detector results
    """
    try:
        # Run Slither detectors
        result = subprocess.run(
            [slither_path, target, "--json", "-"],
            capture_output=True,
            text=True,
            timeout=SLITHER_TIMEOUT
        )
        
        # Check for errors
        if result.returncode != 0 and result.returncode != SLITHER_INFORMATIONAL_EXIT_CODE:
            return {
                "success": False,
                "error": f"Slither failed with exit code {result.returncode}: {result.stderr}"
            }
            
        # Parse and return results
        return {
            "success": True,
            "raw_detector_json_output": json.loads(result.stdout) if result.stdout else {}
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Slither detector analysis timed out"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Failed to parse Slither output as JSON"}


def run_slither_inheritance(slither_path: str, target: str) -> Dict[str, Any]:
    """Run Slither inheritance analysis on the target.
    
    Args:
        slither_path: Path to the Slither executable
        target: Path to the target contract file or directory
        
    Returns:
        Dict containing inheritance analysis results
    """
    try:
        # Run Slither with JSON output for better compatibility
        result = subprocess.run(
            [slither_path, target, "--json", "-"],
            capture_output=True,
            text=True,
            timeout=SLITHER_TIMEOUT
        )
        
        # Check for errors
        if result.returncode != 0 and result.returncode != SLITHER_INFORMATIONAL_EXIT_CODE:
            # If JSON output fails, try a basic analysis without the --json flag
            try:
                result = subprocess.run(
                    [slither_path, target],
                    capture_output=True,
                    text=True,
                    timeout=SLITHER_TIMEOUT
                )
                if result.returncode != 0 and result.returncode != SLITHER_INFORMATIONAL_EXIT_CODE:
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
                
                # Basic success if we get here
                return {
                    "success": True,
                    "inheritance": {},
                    "warnings": ["Using basic analysis mode"]
                }
                
            except subprocess.CalledProcessError as e:
                return {
                    "success": False, 
                    "error": f"Slither analysis failed: {e.stderr}",
                    "returncode": e.returncode,
                    "stderr": e.stderr
                }
            
        return {
            "success": True,
            "inheritance_graph_dot": result.stdout
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Slither inheritance analysis failed with return code {e.returncode}: {e.stderr}")
        return {
            "success": False,
            "error": f"Slither execution failed: {str(e)}",
            "returncode": e.returncode,
            "stderr": e.stderr
        }
    except subprocess.TimeoutExpired:
        logger.error("Slither inheritance analysis timed out")
        return {
            "success": False,
            "error": "Analysis timed out"
        }
    except Exception as e:
        logger.error(f"Unexpected error during Slither inheritance analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the module.
    
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
    
    # Ensure score is within bounds
    score = max(min(score, max_score), min_score)
    
    # Normalize to 0-100 range
    return ((score - min_score) / (max_score - min_score)) * 100


class HeuristicAnalyzer(BaseAnalyzer):
    """Heuristic analyzer for smart contract security assessment.
    
    This analyzer checks for common security patterns and vulnerabilities in Solidity contracts.
    It uses a combination of pattern matching and ML-based analysis.
    """
    
    # Class level pattern constants
    REENTRANCY_PATTERNS = {
        "nonReentrant",
        "ReentrancyGuard",
        "mutex",
        "lock"
    }
    
    ACCESS_CONTROL_PATTERNS = {
        "onlyOwner",
        "onlyAdmin",
        "onlyRole",
        "modifier only"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the HeuristicAnalyzer.
        
        Args:
            config: Optional configuration dictionary that can override default settings
        """
        super().__init__(config or {})
        self.logger = get_logger(__name__)
        self.ml_analyzer = None
        
        # Initialize ML analyzer if dependencies are available
        if ML_DEPENDENCIES_AVAILABLE:
            try:
                self.ml_analyzer = MLVulnerabilityWeightCalculator()
            except Exception as e:
                self.logger.warning(f"Failed to initialize ML analyzer: {e}")

    def analyze(self, contract_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Analyze a smart contract for security vulnerabilities.
        
        Args:
            contract_path: Path to the Solidity contract file
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "success": False,
            "score": 0,
            "vulnerabilities": [],
            "warnings": [],
            "heuristics": {},
            "ml_analysis": {},
            "contract_path": str(contract_path)
        }
        
        try:
            # Convert to Path object if it's a string
            contract_path = Path(contract_path)
            
            # Validate contract path
            if not contract_path.exists():
                raise FileNotFoundError(f"Contract file not found: {contract_path}")
            
            # Read the contract source code
            with open(contract_path, 'r', encoding='utf-8') as f:
                contract_code = f.read()
            
            # Run Slither analysis
            self.logger.info(f"Running Slither analysis on {contract_path.name}")
            slither_results = run_slither(contract_code, contract_path.name)
            
            if not slither_results.get('success', False):
                error_msg = f"Slither analysis failed: {slither_results.get('error', 'Unknown error')}"
                self.logger.error(error_msg)
                results['error'] = error_msg
                return results
            
            # Apply heuristics
            self.logger.info("Applying heuristic analysis")
            heuristic_scores = self._apply_heuristics(slither_results, contract_code)
            results["heuristics"] = heuristic_scores
            
            # Run ML analysis if available
            if self.ml_analyzer:
                try:
                    self.logger.info("Running ML-based vulnerability analysis")
                    ml_results = self.ml_analyzer.calculate_code_vulnerability_similarity(contract_code)
                    results["ml_analysis"] = ml_results
                except Exception as e:
                    error_msg = f"ML analysis failed: {e}"
                    self.logger.error(error_msg)
                    results["warnings"].append(error_msg)
            
            # Calculate overall score
            results["score"] = self._calculate_score(heuristic_scores, results.get("ml_analysis", {}))
            results["success"] = True
            
            self.logger.info(f"Analysis completed successfully. Score: {results['score']:.2f}")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results["error"] = error_msg
        
        return results

    def _apply_heuristics(self, slither_results: Dict[str, Any], contract_code: str) -> Dict[str, Any]:
        """Apply heuristic rules to the analysis results.
        
        Args:
            slither_results: Results from Slither analysis
            contract_code: Raw contract source code
            
        Returns:
            Dictionary of heuristic scores
        """
        scores = {
            "reentrancy_guard": self._check_reentrancy_guard(contract_code),
            "access_control": self._check_access_control(contract_code),
            "safe_math": self._check_safe_math(contract_code),
            "compiler_version": self._check_compiler_version(slither_results)
        }
        return scores

    def _check_reentrancy_guard(self, contract_code: str) -> Dict[str, Any]:
        """Check for reentrancy guard patterns.
        
        Args:
            contract_code: The Solidity contract source code to analyze
            
        Returns:
            Dict containing:
                - score (int): The calculated security score
                - has_guard (bool): Whether reentrancy guard was detected
        """
        lower_code = contract_code.lower()
        has_guard = any(p.lower() in lower_code for p in self.REENTRANCY_PATTERNS)
        return {
            "score": (HEURISTIC_POINTS['reentrancy_guard_present'] 
                     if has_guard 
                     else HEURISTIC_POINTS['reentrancy_guard_absent']),
            "has_guard": has_guard
        }

    def _check_access_control(self, contract_code: str) -> Dict[str, Any]:
        """Check for access control patterns.
        
        Args:
            contract_code: The Solidity contract source code to analyze
            
        Returns:
            Dict containing:
                - score (int): The calculated security score
                - has_access_control (bool): Whether access control was detected
        """
        lower_code = contract_code.lower()
        has_access_control = any(p.lower() in lower_code 
                               for p in self.ACCESS_CONTROL_PATTERNS)
        return {
            "score": (HEURISTIC_POINTS['access_control_present'] 
                     if has_access_control 
                     else HEURISTIC_POINTS['access_control_absent']),
            "has_access_control": has_access_control
        }

    def _check_safe_math(self, contract_code: str) -> Dict[str, Union[bool, int]]:
        """Check for SafeMath usage.
        
        Args:
            contract_code: The Solidity contract source code to analyze
            
        Returns:
            Dict containing:
                - score (int): The calculated security score
                - uses_safemath (bool): Whether SafeMath usage was detected
        """
        # Single scan through the string for better performance
        has_safemath = "SafeMath" in contract_code
        if not has_safemath:
            has_safemath = "using SafeMath" in contract_code
            
        return {
            "score": (HEURISTIC_POINTS.get('safe_math_present', 5) 
                     if has_safemath 
                     else HEURISTIC_POINTS.get('safe_math_absent', -5)),
            "uses_safemath": has_safemath
        }

    def _check_compiler_version(self, slither_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check compiler version for known vulnerabilities."""
        compiler_version = slither_results.get("compiler_version", "")
        if not compiler_version:
            return {"score": 0, "version": "unknown"}
            
        try:
            from packaging import version
            is_modern = version.parse(compiler_version) >= version.parse("0.8.0")
            return {
                "score": 10 if is_modern else -5,
                "version": compiler_version,
                "is_modern": is_modern
            }
        except Exception as e:
            self.logger.warning(f"Error parsing compiler version: {e}")
            return {"score": 0, "version": compiler_version}

    def _calculate_score(self, heuristics: Dict[str, Any], ml_results: Dict[str, Any]) -> float:
        """Calculate an overall security score.
        
        Args:
            heuristics: Dictionary of heuristic scores
            ml_results: Dictionary of ML analysis results
            
        Returns:
            Normalized score between 0 and 100
        """
        # Start with a base score
        score = 50.0
        
        # Add heuristic scores
        for heuristic in heuristics.values():
            if isinstance(heuristic, dict) and "score" in heuristic:
                score += heuristic["score"]
        
        # Add ML-based scores if available
        if ml_results:
            # Adjust score based on ML confidence in vulnerabilities
            for vuln_type, confidence in ml_results.items():
                if confidence > 0.7:  # High confidence in vulnerability
                    score -= 20 * confidence
        
        # Normalize to 0-100 range
        return max(0, min(100, score))