"""Base analyzer class for all contract analysis modules."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.logger import setup_logger
from ..config import config

class BaseAnalyzer(ABC):
    """Base class for all contract analyzers."""
    
    def __init__(self):
        """Initialize the analyzer with a logger and configuration."""
        self.logger = setup_logger(f"analyzer.{self.__class__.__name__}")
        self.config = config
    
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
        path = Path(contract_path)
        
        if not path.exists():
            self.logger.error(f"Contract path does not exist: {contract_path}")
            return False
            
        if path.is_file() and path.stat().st_size > self.config.MAX_CONTRACT_SIZE * 1024:  # Convert KB to bytes
            self.logger.error(f"Contract size exceeds maximum allowed size: {path}")
            return False
            
        return True
    
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
            severity = finding.get('severity', 'informational').lower()
            if severity in severities:
                severities[severity] += 1
                
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
