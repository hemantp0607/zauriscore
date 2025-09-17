import sys
import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from slither import Slither
from slither.analyses.data_dependency.data_dependency import is_dependent
from slither.core.declarations import Function, Contract, FunctionContract
from slither.slithir.operations import HighLevelCall, LowLevelCall, InternalCall
from slither.slithir.variables import Constant
from slither.core.variables.state_variable import StateVariable
from slither.core.cfg.node import Node
from slither.core.expressions.expression import Expression
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from dotenv import load_dotenv
import re
from datetime import datetime
import requests

# Import our analyzers
from .code_similarity import CodeSimilarityAnalyzer
from .gas_optimizer import GasOptimizationAnalyzer

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('zauriscore.log'),
        logging.StreamHandler()
    ]
)

class ComprehensiveContractAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._codebert_tokenizer = None
        self._codebert_model = None
        self.temp_dir = None
        self.temp_file = None
        
        # Load environment variables
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        
        # Load environment variables from .env file
        try:
            load_dotenv()
            self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
            if self.etherscan_api_key:
                # Validate API key format if present
                # Etherscan API keys can vary in length over time; accept 32-64 alphanumeric chars
                if not re.fullmatch(r'[A-Za-z0-9]{32,64}', self.etherscan_api_key):
                    raise ValueError("Invalid Etherscan API key format")
                self.logger.info("Etherscan API key loaded successfully")
            else:
                self.logger.warning("ETHERSCAN_API_KEY not set - API key dependent features will be unavailable")
        except Exception as e:
            self.logger.error(f"Error loading or validating API key: {e}")
            self.etherscan_api_key = None

        # Initialize our analyzers
        self.gas_optimizer = GasOptimizationAnalyzer()
        self._load_codebert()

    def _load_codebert(self):
        try:
            if self._codebert_tokenizer is None or self._codebert_model is None:
                self._codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
                self._codebert_model = AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base')
                self.logger.info("CodeBERT model and tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load CodeBERT model/tokenizer: {e}")
            self._codebert_tokenizer = None
            self._codebert_model = None

    def _run_codebert_analysis(self, source_code: str) -> Dict[str, Any]:
        """Run CodeBERT analysis on the source code."""
        if not self._codebert_model:
            self._load_codebert()

        try:
            # Preprocess the code
            source_code = source_code.strip()
            if not source_code:
                return {'error': 'Empty source code'}

            # Tokenize and encode
            inputs = self._codebert_tokenizer(
                source_code,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Get model predictions
            with torch.no_grad():
                outputs = self._codebert_model(**inputs)
                logits = outputs.logits
                
            # Get predicted class and confidence scores
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence_scores = torch.softmax(logits, dim=1).tolist()[0]

            # Return analysis results
            return {
                'predicted_class': predicted_class,
                'confidence_scores': confidence_scores,
                'analysis_timestamp': datetime.now().isoformat(),
                'source_code_length': len(source_code),
                'token_count': len(self._codebert_tokenizer.tokenize(source_code))
            }

        except Exception as e:
            self.logger.error(f"Error in CodeBERT analysis: {e}")
            return {'error': str(e)}

    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate a summary of the analysis results."""
        # Calculate total issues
        results['summary']['total_issues'] = sum(
            results['static_analysis']['summary'].values()
        )

        # Determine security risk level
        high_issues = results['static_analysis']['summary']['high']
        medium_issues = results['static_analysis']['summary']['medium']
        
        if high_issues > 0:
            results['summary']['security_risk'] = 'High'
        elif medium_issues > 0:
            results['summary']['security_risk'] = 'Medium'
        else:
            results['summary']['security_risk'] = 'Low'

        # Determine gas efficiency
        if results['gas_optimization']['estimated_savings'] > 100000:
            results['summary']['gas_efficiency'] = 'Poor'
        elif results['gas_optimization']['estimated_savings'] > 50000:
            results['summary']['gas_efficiency'] = 'Fair'
        else:
            results['summary']['gas_efficiency'] = 'Good'

        # Generate recommendations
        recommendations = []
        
        # Add security recommendations
        if high_issues > 0:
            recommendations.append("Critical security issues detected. Immediate review required.")
        if medium_issues > 0:
            recommendations.append("Medium severity issues found. Review and fix recommended.")
        
        # Add gas optimization recommendations
        if results['gas_optimization']['estimated_savings'] > 0:
            recommendations.append(f"Gas optimization opportunities found. Estimated savings: ~{results['gas_optimization']['estimated_savings']} gas")
        
        # Add ML analysis recommendations
        if results['ml_analysis'].get('predicted_class') == 1:
            confidence = results['ml_analysis'].get('confidence_scores', [0, 0])[1]
            if confidence > 0.7:
                recommendations.append("ML model detected potential vulnerabilities. Review ML analysis results.")
        
        results['summary']['recommendations'] = recommendations

    def analyze_contract(self, contract_address: str = None, source_code: str = None) -> Dict[str, Any]:
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'contract_address': contract_address,
            'contract_name': 'Unknown',
            'compiler_version': 'Unknown',
            'optimization_used': 'Unknown',
            'runs': 0,
            'static_analysis': {
                'detectors': [],
                'summary': {
                    'high': 0,
                    'medium': 0,
                    'low': 0,
                    'informational': 0
                }
            },
            'pattern_analysis': {
                'patterns': [],
                'matches': 0
            },
            'ml_analysis': {
                'predicted_class': None,
                'confidence_scores': [],
                'source_code_length': 0,
                'token_count': 0
            },
            'gas_optimization': {
                'opportunities': [],
                'estimated_savings': 0
            },
            'summary': {
                'total_issues': 0,
                'security_risk': 'Unknown',
                'gas_efficiency': 'Unknown',
                'recommendations': []
            }
        }

        # Prepare source code and filesystem layout
        try:
            if source_code is None and contract_address:
                # Get source code from Etherscan
                contract_data = self.get_contract_source(contract_address)
                if contract_data.get('status') == '1':
                    item = contract_data['result'][0]
                    source_code = item.get('SourceCode', '')
                    # Add contract metadata to results
                    results.update({
                        'contract_name': item.get('ContractName', 'Unknown'),
                        'compiler_version': item.get('CompilerVersion', 'Unknown'),
                        'optimization_used': item.get('OptimizationUsed', 'Unknown'),
                        'runs': item.get('Runs', 0)
                    })
            elif source_code:
                # For direct source code analysis
                results.update({
                    'contract_name': 'TestContract',
                    'compiler_version': 'Unknown',
                    'optimization_used': 'Unknown',
                    'runs': 0
                })

            if not source_code:
                self.logger.error("No source code available for analysis")
                return results

            # Directory to place temporary contract files
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_contracts')
            os.makedirs(temp_dir, exist_ok=True)

            # Default main file path
            import time
            main_file = os.path.abspath(os.path.join(temp_dir, f'Contract_{int(time.time())}.sol'))

            # Normalize Etherscan quirks: double-braced JSON and escaped newlines
            sc = source_code.strip()
            # If JSON wrapped in extra braces {{ ... }}, strip one layer
            if (sc.startswith("{{") and sc.endswith("}}")) or (sc.startswith("{\n{") and sc.rstrip().endswith("}\n}")):
                sc = sc[1:-1]

            # Attempt to parse as JSON (multi-file format)
            try:
                parsed = json.loads(sc)
                if isinstance(parsed, dict) and 'sources' in parsed:
                    # Write each source file
                    for rel_path, file_data in parsed['sources'].items():
                        norm_path = os.path.normpath(rel_path)
                        full_path = os.path.join(temp_dir, norm_path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(file_data.get('content', ''))
                    # Set source_path to directory for Slither to discover contracts
                    source_path = temp_dir
                else:
                    # Single-file JSON with 'content'
                    with open(main_file, 'w', encoding='utf-8') as f:
                        f.write(parsed.get('content', source_code))
                    source_path = main_file
            except json.JSONDecodeError:
                # Not JSON; treat as raw solidity source but unescape if needed
                try:
                    # If the string includes escaped newlines like \r\n or \n, unescape them
                    if "\\n" in sc or "\\r\\n" in sc:
                        sc = bytes(sc, 'utf-8').decode('unicode_escape')
                except Exception:
                    pass
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(sc)
                source_path = main_file

        except Exception as e:
            self.logger.error(f"Error preparing source code: {e}")
            return {'error': f'Error preparing source code: {str(e)}'}
            
        # Initialize Slither with the source code
        try:
            slither = Slither(source_path)
            
            # Update contract metadata
            if slither.contracts:
                contract = slither.contracts[0]
                results['contract_name'] = contract.name
                try:
                    # Try to get compiler information
                    results['compiler_version'] = contract.compilation_unit.compiler_version
                    results['optimization_used'] = contract.compilation_unit.compiler_optimization
                    results['runs'] = contract.compilation_unit.compiler_runs
                except AttributeError:
                    # Fallback if newer Slither API
                    results['compiler_version'] = 'Unknown'
                    results['optimization_used'] = 'Unknown'
                    results['runs'] = 0
                
                # Analyze gas optimization
                try:
                    gas_results = self.gas_optimizer.analyze(contract)
                    results['gas_optimization']['opportunities'] = gas_results
                    # Estimate gas savings
                    results['gas_optimization']['estimated_savings'] = sum(
                        2000 for _ in gas_results  # Estimate 2000 gas per optimization
                    )
                except Exception as e:
                    self.logger.error(f"Error in gas optimization analysis: {e}")
                    results['gas_optimization']['opportunities'] = []
                    results['gas_optimization']['estimated_savings'] = 0

                # Run Slither detectors
                for detector in slither.detectors:
                    try:
                        issues = detector.detect()
                        if issues:
                            severity = detector.IMPACT.title()
                            results['static_analysis']['summary'][severity.lower()] += len(issues)
                            results['static_analysis']['detectors'].extend([
                                {
                                    'title': issue.title,
                                    'description': issue.description,
                                    'severity': severity,
                                    'impact': detector.IMPACT.title(),
                                    'confidence': detector.CONFIDENCE.title(),
                                    'lines': issue.lines
                                } for issue in issues
                            ])
                    except Exception as e:
                        self.logger.error(f"Error running detector {detector.__class__.__name__}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error initializing Slither or analyzing contract: {e}")
            return {'error': f'Error analyzing contract: {str(e)}'}

        # Run CodeBERT analysis

    def get_contract_source(self, address: str) -> dict:
        if not address.startswith('0x') or len(address) != 42:
            raise ValueError("Invalid Ethereum address")

        if not self.etherscan_api_key:
            raise ValueError("ETHERSCAN_API_KEY not configured")

        try:
            # Mask API key in logs
            masked = f"{self.etherscan_api_key[:4]}***{self.etherscan_api_key[-4:]}"
            # V2 API pattern uses /v2/api with module/action and requires chainid
            # Default to Ethereum mainnet (chainid=1)
            log_url = (
                f"https://api.etherscan.io/v2/api?module=contract&action=getsourcecode"
                f"&address={address}&chainid=1&apikey={masked}"
            )
            self.logger.info(f"Request URL: {log_url}")

            # Actual request with full key
            url = (
                f"https://api.etherscan.io/v2/api?module=contract&action=getsourcecode"
                f"&address={address}&chainid=1&apikey={self.etherscan_api_key}"
            )
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Normalize to V1-like structure expected by existing callers
            if str(data.get('status')) == '1' and str(data.get('message')).upper() == 'OK' and isinstance(data.get('result'), dict):
                result = data['result']
                # Wrap into a list to mimic V1 shape
                return {'status': '1', 'message': 'OK', 'result': [result]}

            # Handle NOTOK
            if str(data.get('message')).upper() == 'NOTOK':
                raise Exception(f"Etherscan V2 API error: {data.get('result', 'Unknown error')}\nURL: {url}")

            raise Exception(f"Unexpected Etherscan V2 response: {data}\nURL: {url}")

        except Exception as e:
            self.logger.error(f"Error fetching contract source (V2): {e}")
            raise
