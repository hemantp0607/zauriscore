"""Base analyzer class for all contract analysis modules."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import handling with graceful fallback
try:
    from ..utils.logger import setup_logger
    from ..config import settings as config
except ImportError:
    import logging
    
    def setup_logger(name: str):
        """Fallback logger setup"""
        return logging.getLogger(name)
    
    # Fallback config object
    class Config:
        def get(self, key: str, default=None):
            return default
    
    config = Config()

class BaseAnalyzer(ABC):
    """Base class for all contract analyzers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analyzer with a logger and configuration.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.logger = setup_logger(f"analyzer.{self.__class__.__name__}")
        self.config = config or globals().get('config', {})
        self._validate_config()
    
    @abstractmethod
    def analyze(self, contract_path: str, **kwargs) -> Dict[str, Any]:
        """Analyze a smart contract.
        
        Args:
            contract_path: Path to the contract file or directory
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def validate_contract(self, contract_path: str) -> bool:
        """Validate if the contract is suitable for analysis.
        
        Args:
            contract_path: Path to the contract file or directory
            
        Returns:
            bool: True if contract is valid, False otherwise
        """
        try:
            path = Path(contract_path)
            
            if not path.exists():
                self.logger.error(f"Contract path does not exist: {contract_path}")
                return False
            
            # Get max size from config (default 100KB)
            max_size_kb = self.config.get('MAX_CONTRACT_SIZE', 100)
            max_size_bytes = max_size_kb * 1024
            
            if path.is_file():
                file_size = path.stat().st_size
                if file_size > max_size_bytes:
                    self.logger.error(
                        f"Contract size ({file_size} bytes) exceeds maximum "
                        f"allowed size ({max_size_bytes} bytes): {path}"
                    )
                    return False
                    
            return True
            
        except (OSError, PermissionError) as e:
            self.logger.error(f"Error validating contract path {contract_path}: {e}")
            return False
    
    def format_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format analysis findings into a standardized format.
        
        Args:
            findings: List of raw findings
            
        Returns:
            Formatted findings dictionary
        """
        return {
            'severity_counts': self._count_severities(findings),
            'findings': findings,
            'summary': self._generate_summary(findings)
        }
    
    def _count_severities(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count findings by severity."""
        severities = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'informational': 0
        }
        
        for finding in findings:
            severity = finding.get('severity', 'informational').lower().strip()
            
            # Handle common severity variations
            severity_map = {
                'info': 'informational',
                'warning': 'medium',
                'error': 'high',
                'fatal': 'critical'
            }
            
            severity = severity_map.get(severity, severity)
            
            if severity in severities:
                severities[severity] += 1
            else:
                # Log unknown severity but don't crash
                self.logger.warning(f"Unknown severity level: {finding.get('severity')}")
                severities['informational'] += 1
                
        return severities
    
    def _generate_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Generate a human-readable summary of findings."""
        counts = self._count_severities(findings)
        total = sum(counts.values())
        
        if total == 0:
            return "No security issues found."
            
        summary = ["Security issues found:"]
        for severity, count in counts.items():
            if count > 0:
                summary.append(f"- {severity.title()}: {count}")
                
        return "\n".join(summary)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if hasattr(self.config, 'get'):
            max_size = self.config.get('MAX_CONTRACT_SIZE', 100)
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                self.logger.warning(
                    f"Invalid MAX_CONTRACT_SIZE: {max_size}. Using default: 100KB"
                )
                # Could set a corrected value here if needed