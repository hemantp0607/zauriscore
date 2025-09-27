import sys
import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from slither import Slither
from datetime import datetime
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from dotenv import load_dotenv
import re
import shutil

# Import our analyzers with graceful fallback
try:
    from .code_similarity import CodeSimilarityAnalyzer
    from .gas_optimizer import GasOptimizationAnalyzer
except ImportError:
    import logging
    logging.warning("Some analyzer imports not available. Features may be limited.")
    
    class CodeSimilarityAnalyzer:
        """Placeholder for missing import"""
        def analyze(self, *args, **kwargs):
            return []
    
    class GasOptimizationAnalyzer:
        """Placeholder for missing import"""
        def analyze(self, *args, **kwargs):
            return []

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
        
        # Load and validate API key
        try:
            load_dotenv()
            self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
            if self.etherscan_api_key:
                # Validate API key format if present
                # Etherscan API keys can vary in length over time; accept 22-64 alphanumeric chars
                if not re.fullmatch(r'[A-Za-z0-9]{22,64}', self.etherscan_api_key):
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

    def _cleanup_temp_files(self, temp_dir: str, source_path: str) -> None:
        """Clean up temporary files and directories"""
        try:
            if source_path and os.path.exists(source_path):
                if os.path.isfile(source_path):
                    os.unlink(source_path)
                else:
                    shutil.rmtree(source_path, ignore_errors=True)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.debug("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def analyze_contract(self, contract_address: str = None, source_code: str = None, chainid: int = 1) -> Dict[str, Any]:
        import os  # Import os at the method level to ensure it's available
        import tempfile
        import shutil
        import time
        
        self.logger.info(f"Starting contract analysis for address: {contract_address}")
        
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
                'token_count': 0,
                'error': None
            },
            'gas_optimization': {
                'opportunities': [],
                'estimated_savings': 0,
                'error': None
            },
            'summary': {
                'total_issues': 0,
                'security_risk': 'Unknown',
                'gas_efficiency': 'Unknown',
                'recommendations': []
            },
            'errors': []
        }

        # Create a unique temporary directory for this analysis
        temp_dir = tempfile.mkdtemp(prefix='zauriscore_')
        self.logger.debug(f"Created temporary directory: {temp_dir}")
        
        source_path = None

        try:
            # Prepare source code and filesystem layout
            if source_code is None and contract_address:
                # Get source code from Etherscan
                contract_data = self.get_contract_source(contract_address, chainid)
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
            
            # If we have source code (either from file or Etherscan)
            if source_code:
                # For direct source code analysis
                if 'contract_name' not in results:  # Only update if not set from Etherscan
                    results.update({
                        'contract_name': 'TestContract',
                        'compiler_version': 'Unknown',
                        'optimization_used': 'Unknown',
                        'runs': 0
                    })
            if not source_code:
                self.logger.error("No source code available for analysis")
                return results

            # Use the temporary directory created at the start of the method
            # Default main file path
            import time
            safe_contract_name = ''.join(c if c.isalnum() else '_' for c in results['contract_name'])
            main_file = os.path.join(temp_dir, f"{safe_contract_name}_{int(time.time())}.sol")
            self.logger.debug(f"Using main file: {main_file}")

            # Clean up the source code
            sc = source_code.strip()
            self.logger.debug(f"Source code length: {len(sc)} characters")
            
            # Remove any byte order mark (BOM) if present
            if sc.startswith('\ufeff'):
                sc = sc[1:].strip()
            
            # Log the first 200 characters of the source code for debugging
            self.logger.debug(f"Source code preview: {sc[:200]}...")
                
            # If the source looks like a JSON object (for multi-file projects)
            if sc.startswith('{') and sc.endswith('}'):
                self.logger.debug("Source appears to be in JSON format")
                
                # If JSON wrapped in extra braces {{ ... }}, strip one layer
                if (sc.startswith("{{") and sc.endswith("}}")) or (sc.startswith("{\n{") and sc.rstrip().endswith("}\n}")):
                    self.logger.debug("Source is wrapped in extra braces, stripping one layer")
                    sc = sc[1:-1].strip()
                
                # Attempt to parse as JSON (multi-file format)
                try:
                    parsed = json.loads(sc)
                    self.logger.debug(f"Successfully parsed source as JSON. Keys: {list(parsed.keys())}")
                    
                    sources = parsed.get('sources', parsed.get('source', {}))
                    if isinstance(sources, dict):
                        # Write each source file
                        for rel_path, file_data in sources.items():
                            try:
                                norm_path = os.path.normpath(rel_path)
                                full_path = os.path.join(temp_dir, norm_path)
                                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                                
                                content = file_data.get('content', '')
                                if not content and isinstance(file_data, str):
                                    content = file_data  # Handle case where file_data is directly the content
                                    
                                if content.startswith('\ufeff'):
                                    content = content[1:]
                                    
                                with open(full_path, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                    
                                self.logger.debug(f"Wrote source file: {full_path} ({len(content)} bytes)")
                                
                            except Exception as e:
                                self.logger.error(f"Error writing source file {rel_path}: {e}")
                                raise
                        
                        # Set source_path to directory for Slither to discover contracts
                        source_path = temp_dir
                        self.logger.debug(f"Source path set to directory: {source_path}")
                    else:
                        # Single-file JSON with 'content'
                        self.logger.debug("No 'sources'/'source' key found, treating as single file with 'content' key")
                        content = parsed.get('content', sc)  # Fallback to original source if no content key
                        
                        if content.startswith('\ufeff'):
                            content = content[1:]
                            
                        with open(main_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                            
                        source_path = main_file
                        self.logger.debug(f"Source path set to single file: {source_path} ({len(content)} bytes)")
                        
                except json.JSONDecodeError as e:
                    # Not valid JSON, treat as raw Solidity code
                    self.logger.warning(f"Failed to parse source as JSON: {e}. Treating as raw Solidity code")
                    with open(main_file, 'w', encoding='utf-8') as f:
                        f.write(sc)
                    source_path = main_file
                    self.logger.debug(f"Source path set to raw Solidity file: {source_path}")
                    
            else:
                # Not JSON, treat as raw Solidity code
                self.logger.debug("Source does not appear to be JSON, treating as raw Solidity code")
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(sc)
                source_path = main_file
                self.logger.debug(f"Source path set to raw Solidity file: {source_path}")
                
            # Verify the source file/directory exists
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source path does not exist: {source_path}")
                
            if os.path.isdir(source_path):
                self.logger.debug(f"Source is a directory. Contents: {os.listdir(source_path)}")
                # If it's a directory, find the first .sol file
                sol_files = [f for f in os.listdir(source_path) if f.endswith('.sol')]
                if sol_files:
                    source_path = os.path.join(source_path, sol_files[0])
                    self.logger.debug(f"Using Solidity file: {source_path}")
                else:
                    raise ValueError("No .sol files found in the source directory")
            
            # Set the solc binary path to the one installed by py-solc-x
            import solcx
            from pathlib import Path
            
            # Get the path to the solc binary installed by py-solc-x
            solc_versions = solcx.get_installed_solc_versions()
            if not solc_versions:
                raise RuntimeError("No Solidity compiler found. Please install solc using 'solcx.install_solc('0.8.0')'")
            
            # Use the first installed version
            solc_version = solc_versions[0]
            solc_path = solcx.install.get_executable(solc_version)
            
            # Set the SOLC environment variable to the solc binary path
            os.environ["SOLC_PATH"] = str(solc_path)
            
            # Log the source path and its type
            self.logger.debug(f"Source path: {source_path}")
            if os.path.isdir(source_path):
                self.logger.debug(f"Source path is a directory. Contents: {os.listdir(source_path)}")
                # If it's a directory, find the first .sol file
                sol_files = [f for f in os.listdir(source_path) if f.endswith('.sol')]
                if sol_files:
                    source_path = os.path.join(source_path, sol_files[0])
                    self.logger.debug(f"Using Solidity file: {source_path}")
                else:
                    raise ValueError("No .sol files found in the source directory")
            
            # Initialize Slither with the source code and explicit solc path
            self.logger.debug(f"Initializing Slither with source: {source_path}")
            slither = Slither(source_path, solc=str(solc_path))
            
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

            # Run ML analysis
            try:
                ml_results = self._run_codebert_analysis(source_code)
                results['ml_analysis'].update(ml_results)
            except Exception as e:
                self.logger.error(f"Error in ML analysis: {e}")
                results['ml_analysis']['error'] = str(e)

            # Generate summary
            self._generate_summary(results)

        except Exception as e:
            self.logger.error(f"Error preparing source code or analyzing contract: {e}")
            results['error'] = f'Error analyzing contract: {str(e)}'
            
        finally:
            # Always cleanup temporary files
            self._cleanup_temp_files(temp_dir, source_path)

        return results

    def get_contract_source(self, address: str, chainid: int = 1) -> dict:
        """
        Fetch contract source code from Etherscan API.
        
        Args:
            address: Ethereum contract address (0x...)
            chainid: Blockchain ID (1=Mainnet)
            
        Returns:
            dict: Contract source code and metadata if successful, None otherwise
        """
        if not address or not isinstance(address, str) or not address.startswith('0x') or len(address) != 42:
            error_msg = f"Invalid Ethereum address: {address}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not self.etherscan_api_key or not isinstance(self.etherscan_api_key, str):
            error_msg = "Etherscan API key is not properly configured"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.info(f"[Etherscan V2] Fetching source for contract: {address} on chain {chainid}")
        self.logger.debug(f"[Etherscan V2] Using API key: {self.etherscan_api_key[:4]}...{self.etherscan_api_key[-4:]}")

        base_params = {
            'module': 'contract',
            'action': 'getsourcecode',
            'address': address,
            'apikey': self.etherscan_api_key,
            'chainid': str(chainid)
        }
        
        api_url = "https://api.etherscan.io/v2/api"
        
        try:
            response = requests.get(api_url, params=base_params, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            try:
                data = response.json()
                self.logger.debug(f"[V2 API] Raw response: {json.dumps(data, indent=2)}")
                
                # Check if the response is valid
                if not isinstance(data, dict):
                    self.logger.warning("[V2 API] Invalid response format")
                    return {'status': '0', 'message': 'Invalid response', 'result': None}
                    
                # Check status code
                status = str(data.get('status', '0'))
                if status != '1':
                    error_msg = data.get('message', 'Unknown error')
                    self.logger.warning(f"[V2 API] Request failed: {error_msg}")
                    return {'status': '0', 'message': error_msg, 'result': data.get('result')}
                
                # Get the first result
                result = data.get('result')
                if not result or not isinstance(result, list) or len(result) == 0:
                    self.logger.warning("[V2 API] No results returned")
                    return {'status': '0', 'message': 'No results', 'result': None}
                    
                contract_data = result[0] if isinstance(result, list) else result
                
                # Log the contract data structure for debugging
                self.logger.debug(f"Contract data keys: {list(contract_data.keys())}")
                
                # Check if source code is available
                source_code = contract_data.get('SourceCode')
                if not source_code:
                    self.logger.warning("[V2 API] No source code in response")
                    self.logger.debug(f"Contract data: {json.dumps(contract_data, indent=2)}")
                    return {'status': '0', 'message': 'No source code', 'result': [contract_data]}
                
                # Log the source code format for debugging
                source_type = 'JSON' if source_code.strip().startswith('{') else 'Solidity'
                self.logger.info(f"Source code type: {source_type}, length: {len(source_code)} characters")
                
                # Log the first 500 characters of the source code for debugging
                self.logger.debug(f"Source code preview: {source_code[:500]}...")
                
                # Log success
                contract_name = contract_data.get('ContractName', 'Unknown')
                self.logger.info(
                    f"Successfully retrieved source code for {contract_name} "
                    f"({len(source_code)} chars) from V2 API"
                )
                
                # Return the complete contract data
                return {
                    'status': '1',
                    'message': 'OK',
                    'result': [contract_data]
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"[V2 API] Failed to parse JSON response: {e}")
                self.logger.debug(f"Response text: {response.text[:500]}...")
                return {'status': '0', 'message': 'Parse error', 'result': None}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[V2 API] Request failed: {str(e)}")
            return {'status': '0', 'message': str(e), 'result': None}
        except Exception as e:
            self.logger.error(f"[V2 API] Unexpected error: {str(e)}", exc_info=True)
            return {'status': '0', 'message': str(e), 'result': None}