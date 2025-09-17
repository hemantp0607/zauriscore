"""
Test script for verifying report generation in different formats.
"""
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from zauriscore.utils.report_generator import generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data() -> dict:
    """Create sample analysis data for testing."""
    return {
        "metadata": {
            "tool_version": "1.2.0",
            "analysis_date": "2025-03-15T10:30:00Z",
            "contract_address": "0x1234...abcd"
        },
        "decision_summary": {
            "status": "Needs-Review",
            "reasons": ["Medium severity issues found"],
            "risk_score": 65,
            "risk_category": "Medium",
            "highlights": [
                "Potential reentrancy vulnerability in withdraw() function",
                "Missing zero-address validation in constructor"
            ]
        },
        "contract_details": {
            "ContractName": "SampleToken",
            "CompilerVersion": "0.8.20",
            "LicenseType": "MIT",
            "OptimizationUsed": True
        },
        "provenance": {
            "chain": {"chainid": 1, "network": "Ethereum Mainnet"},
            "compiler": {
                "requested_version": "0.8.20",
                "used_version": "0.8.20+commit.a1b79de6"
            },
            "tools": {
                "slither_version": "0.10.0",
                "mythril_version": "0.23.28",
                "detectors": ["reentrancy-eth", "tx-origin"]
            },
            "sources": {
                "etherscan_endpoint": "https://etherscan.io/address/0x1234...abcd#code",
                "response_hash": "a1b2c3...",
                "verification_status": "Verified",
                "source_type": "Solidity"
            },
            "runtime": {
                "started_at": "2025-03-15T10:20:00Z",
                "finished_at": "2025-03-15T10:25:30Z",
                "duration_seconds": 330
            }
        },
        "security_issues": [
            {
                "title": "Reentrancy in withdraw()",
                "severity": "Medium",
                "confidence": "High",
                "location": "contracts/SampleToken.sol:42",
                "description": "The withdraw() function makes an external call before updating state variables.",
                "recommendation": "Follow the checks-effects-interactions pattern and update state variables before making external calls."
            },
            {
                "title": "Missing zero-address validation",
                "severity": "Low",
                "confidence": "Medium",
                "location": "contracts/SampleToken.sol:15",
                "description": "The constructor does not validate that the owner address is not the zero address.",
                "recommendation": "Add a require statement to validate that the owner address is not the zero address."
            }
        ]
    }

def test_report_generation():
    """Test report generation in all formats."""
    output_dir = Path("test_reports")
    output_dir.mkdir(exist_ok=True)
    
    sample_data = create_sample_data()
    
    # Test JSON generation
    json_path = output_dir / "sample_report.json"
    try:
        result = generate_report(sample_data, str(json_path), 'json')
        logger.info(f"JSON report generated: {result}")
    except Exception as e:
        logger.error(f"Error generating JSON report: {e}")
    
    # Test Markdown generation
    md_path = output_dir / "sample_report.md"
    try:
        result = generate_report(sample_data, str(md_path), 'markdown')
        logger.info(f"Markdown report generated: {result}")
    except Exception as e:
        logger.error(f"Error generating Markdown report: {e}")
    
    # Test PDF generation (requires pandoc and LaTeX)
    pdf_path = output_dir / "sample_report.pdf"
    try:
        result = generate_report(sample_data, str(pdf_path), 'pdf')
        logger.info(f"PDF report generated: {result}")
    except Exception as e:
        logger.warning(f"PDF generation failed (requires pandoc and LaTeX): {e}")
    
    logger.info("Report generation tests completed")

if __name__ == "__main__":
    test_report_generation()
