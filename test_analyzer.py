"""
Test script for the ZauriScore heuristic analyzer.
"""
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zauriscore.analyzers.heuristic_analyzer import HeuristicAnalyzer

def main():
    # Initialize the analyzer
    analyzer = HeuristicAnalyzer()
    
    # Test with a sample contract (you can replace this with an actual contract path)
    sample_contract = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract SimpleStorage {
        uint256 public storedData;
        
        function set(uint256 x) public {
            storedData = x;
        }
        
        function get() public view returns (uint256) {
            return storedData;
        }
    }
    """
    
    # Save the sample contract to a temporary file
    contract_path = "test_contract.sol"
    with open(contract_path, "w") as f:
        f.write(sample_contract)
    
    try:
        # Run the analysis
        print("Running analysis on sample contract...")
        result = analyzer.analyze(contract_path)
        
        # Print the results
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Contract: {contract_path}")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Vulnerabilities found: {len(result.get('vulnerabilities', []))}")
        
        if 'vulnerabilities' in result and result['vulnerabilities']:
            print("\nVulnerabilities:")
            for vuln in result['vulnerabilities']:
                print(f"- {vuln.get('title', 'Unknown')} ({vuln.get('severity', 'unknown')}): {vuln.get('description', 'No description')}")
        
        print("\nFull Results:")
        print("-" * 50)
        import json
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(contract_path):
            os.remove(contract_path)

if __name__ == "__main__":
    main()
