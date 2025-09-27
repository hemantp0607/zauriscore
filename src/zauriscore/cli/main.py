"""
ZauriScore CLI - Command Line Interface for Smart Contract Analysis
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from zauriscore.core.ml_pipeline import ZauriScoreMLPipeline
from zauriscore.analyzers.comprehensive_contract_analysis import ComprehensiveContractAnalyzer
from zauriscore.utils.report_generator import generate_report

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def analyze_contract(contract_path: str, output_path: Optional[str] = None, 
                   format: str = 'json', etherscan_api_key: Optional[str] = None) -> None:
    """Analyze a smart contract and generate a report."""
    try:
        logger.info(f"Analyzing contract: {contract_path}")
        
        if etherscan_api_key:
            os.environ['ETHERSCAN_API_KEY'] = etherscan_api_key
            
        analyzer = ComprehensiveContractAnalyzer()
        
        if os.path.isfile(contract_path):
            with open(contract_path, 'r', encoding='utf-8-sig') as f:
                source_code = f.read().strip()
                
            if not source_code:
                raise ValueError("Contract file is empty")
                
            result = analyzer.analyze_contract(source_code=source_code)
        else:
            result = analyzer.analyze_contract(contract_address=contract_path)
        
        if output_path:
            output = generate_report(result, output_path, format)
            logger.info(f"Report generated: {output}")
        else:
            if format == 'json':
                import json
                print(json.dumps(result, indent=2))
            else:
                print(generate_report(result, format=format))
                
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}")
        sys.exit(1)

def train_model(dataset_path: str, model_output: str) -> None:
    """Train the ML model on the specified dataset."""
    try:
        logger.info(f"Training model on dataset: {dataset_path}")
        pipeline = ZauriScoreMLPipeline()
        pipeline.train(dataset_path, model_output)
        logger.info(f"Model saved to: {model_output}")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

def main() -> None:
    """Main entry point for the ZauriScore CLI."""
    try:
        parser = argparse.ArgumentParser(
            description='ZauriScore - Smart Contract Security Analysis Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Global arguments
        parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze a smart contract')
        analyze_parser.add_argument('contract_path', help='Path to the smart contract file or Ethereum address')
        analyze_parser.add_argument('-o', '--output', help='Output file path for the report (default: print to stdout)')
        analyze_parser.add_argument('--format', choices=['json', 'markdown', 'pdf'], default='json',
                                  help='Output format of the report')
        analyze_parser.add_argument('--etherscan-key', help='Etherscan API key for contract verification')

        # Train command
        train_parser = subparsers.add_parser('train', help='Train the ML model')
        train_parser.add_argument('dataset_path', help='Path to the training dataset')
        train_parser.add_argument('model_output', help='Output path for the trained model')

        args = parser.parse_args()
        
        # Set up logging with verbosity
        setup_logging(verbose=args.verbose)
        
        # Enable debug logging for all zauriscore modules if verbose
        if args.verbose:
            logging.getLogger('zauriscore').setLevel(logging.DEBUG)
            logging.getLogger('urllib3').setLevel(logging.INFO)  # Reduce noise from urllib3
        
        if args.command == 'analyze':
            analyze_contract(
                contract_path=args.contract_path,
                output_path=args.output,
                format=args.format,
                etherscan_api_key=args.etherscan_key
            )
        elif args.command == 'train':
            train_model(
                dataset_path=args.dataset_path,
                model_output=args.model_output
            )
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if 'args' in locals() and hasattr(args, 'verbose') and args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)

if __name__ == '__main__':
    main()