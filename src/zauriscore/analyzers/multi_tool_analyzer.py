import os
import json
import logging
import subprocess
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Local imports with graceful fallback
try:
    from ..data import ContractSourceRetriever
    from ..utils import generate_report, check_slither_installation
except ImportError:
    # Handle missing local imports gracefully
    import logging
    logging.warning("Local imports not available. Some features may be limited.")
    
    class ContractSourceRetriever:
        """Placeholder for missing import"""
        pass
    
    def generate_report(*args, **kwargs):
        """Placeholder for missing import"""
        return {}
    
    def check_slither_installation():
        """Placeholder for missing import"""
        return True

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class MultiToolAnalyzer:
    def __init__(self, etherscan_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """Initialize MultiToolAnalyzer with API keys"""
        # Set up API keys
        self.api_key = self._resolve_etherscan_api_key(etherscan_api_key)
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY', '')
        
        # Set up logging
        self._setup_logging()
        
        # Validate configuration
        if not self.openai_api_key:
            self.logger.warning("No OpenAI API key provided. Some features may be limited.")

    def _validate_api_key(self, key: str) -> bool:
        """Validate Etherscan API key format and basic integrity"""
        if not key or len(key) < 10:
            return False
        return key.isupper() and all(c.isalnum() for c in key)

    def _resolve_etherscan_api_key(self, etherscan_api_key: Optional[str]) -> str:
        """Resolve Etherscan API key from multiple sources"""
        # Priority 1: Explicitly passed key
        if etherscan_api_key and self._validate_api_key(etherscan_api_key.strip()):
            return etherscan_api_key.strip()
        
        # Priority 2: Environment variable
        env_key = os.getenv('ETHERSCAN_API_KEY', '').strip()
        if env_key and self._validate_api_key(env_key):
            return env_key
        
        # Fallback: Return placeholder (log error)
        logger.error("No valid Etherscan API key found. Please provide a valid key.")
        return 'INVALID_API_KEY'

    def _setup_logging(self) -> None:
        """Configure logging with enhanced verbosity"""
        self.logger = logging.getLogger('MultiToolAnalyzer')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Log configuration (without sensitive data)
        self.logger.debug("Etherscan API Key configured: [REDACTED]")
    
    def _validate_network_connectivity(self) -> bool:
        """Simple network connectivity check"""
        try:
            response = requests.get('https://api.etherscan.io', timeout=5)
            return response.status_code == 200
        except Exception:
            return True  # Don't block on network issues, let the actual request handle it
    
    def retrieve_source_code(self, contract_address: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Retrieve contract source code using multiple methods:
        1. Local Cache
        2. Etherscan API
        3. Web3.py Fallback
        
        Args:
            contract_address (str): Ethereum contract address
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        
        Returns:
            Dict with source code and metadata
        """
        import time
        import random
        import os
        import hashlib
        import socket
        import ssl
        import certifi
        import urllib3
        import requests
        import json
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Cache directory for contract sources
        cache_dir = os.path.join(os.path.dirname(__file__), 'contract_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        def get_cache_path(contract_address: str) -> str:
            """Generate a unique cache file path for a contract address"""
            cache_filename = hashlib.md5(contract_address.encode()).hexdigest() + '.json'
            return os.path.join(cache_dir, cache_filename)
        
        def save_to_cache(contract_address: str, source_data: Dict[str, Any]) -> None:
            """Save contract source to local cache"""
            try:
                cache_path = get_cache_path(contract_address)
                with open(cache_path, 'w') as f:
                    json.dump(source_data, f, indent=2)
                self.logger.info(f"Cached source code for {contract_address}")
            except Exception as e:
                self.logger.warning(f"Failed to cache source code: {e}")
        
        def load_from_cache(contract_address: str) -> Dict[str, Any]:
            """Load contract source from local cache"""
            try:
                cache_path = get_cache_path(contract_address)
                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    self.logger.info(f"Loaded source code from cache for {contract_address}")
                    return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cached source code: {e}")
            return {}
        
        # Method 1: Check local cache first
        cached_source = load_from_cache(contract_address)
        if cached_source:
            return cached_source
        
        # Method 2: Etherscan API
        self.logger.debug(f"Using Etherscan API for contract {contract_address}")
        
        # Construct URL with explicit API key
        url = "https://api.etherscan.io/api"
        
        # Validate API key before making request
        if not self.api_key or self.api_key == 'INVALID_API_KEY':
            raise ValueError("Cannot make Etherscan API request with invalid API key")
        
        # Explicit API key sanitization
        sanitized_api_key = ''.join(c for c in self.api_key if c.isalnum())
        
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address,
            "apikey": sanitized_api_key
        }
        
        # Log the request details (without sensitive data)
        self.logger.debug(f"API Request URL: {url}")
        self.logger.debug(f"API Request for contract: {contract_address}")
        
        # Detailed request headers for debugging
        headers = {
            'User-Agent': 'MultiToolAnalyzer/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        # Verify network before making requests
        if not self._validate_network_connectivity():
            self.logger.warning("Network connectivity issues detected")
        
        for attempt in range(max_retries):
            try:
                # Use a session for potential performance and connection reuse
                with requests.Session() as session:
                    response = session.get(
                        url, 
                        params=params,
                        headers=headers, 
                        timeout=(5, 10),  # (connect timeout, read timeout)
                        verify=True  # Use system CA certificates
                    )
                
                # Log response details for debugging
                self.logger.debug(f"Response Status Code: {response.status_code}")
                
                # Raise exception for bad HTTP status
                response.raise_for_status()
                
                data = response.json()
                
                # Detailed logging for debugging
                self.logger.debug(f"Etherscan API Response Status: {data.get('status')}")
                self.logger.debug(f"Etherscan API Response Message: {data.get('message', 'No message')}")
                
                # Comprehensive status and result checking
                if data.get('status') == '1' and data.get('result'):
                    source_info = data['result'][0]
                    source_code = source_info.get('SourceCode', '')
                    
                    # Extensive logging for source code retrieval
                    self.logger.debug(f"Raw Source Code Length: {len(source_code)}")
                    self.logger.debug(f"Contract Name: {source_info.get('ContractName', 'Unknown')}")
                    
                    # Handle JSON-like source code formats
                    if source_code.startswith('{{') and source_code.endswith('}}'):
                        try:
                            source_json = json.loads(source_code[1:-1])
                            # Extract main contract source if multiple sources exist
                            if isinstance(source_json.get('sources'), dict):
                                source_code = next(iter(source_json['sources'].values())).get('content', '')
                        except json.JSONDecodeError:
                            self.logger.warning("Could not parse complex source code format")
                    
                    source_data = {
                        'source_code': source_code,
                        'contract_name': source_info.get('ContractName', 'Unknown'),
                        'compiler_version': source_info.get('CompilerVersion', ''),
                        'abi': source_info.get('ABI', ''),
                        'raw_response': source_info
                    }
                    
                    # Save to cache
                    save_to_cache(contract_address, source_data)
                    return source_data
                else:
                    error_msg = data.get('message', 'Unknown error')
                    self.logger.error(f"Etherscan API error: {error_msg}")
                    
                    # Additional error context logging
                    if 'rate limit' in error_msg.lower():
                        self.logger.warning("Rate limit detected. Backing off...")
            
            except (requests.RequestException, ValueError) as e:
                self.logger.error(f"Etherscan API retrieval error (Attempt {attempt + 1}/{max_retries}): {e}")
                
                # Exponential backoff with jitter for connection errors
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Connection error. Waiting {wait_time:.2f} seconds before retry.")
                    time.sleep(wait_time)
        
        # Method 3: Web3.py Fallback
        try:
            from web3 import Web3
            from web3.auto import w3
            
            # Try to get source code via web3
            source_code = w3.eth.contract(address=Web3.toChecksumAddress(contract_address)).source_code
            
            if source_code:
                source_data = {
                    'source_code': source_code,
                    'contract_name': 'Unknown (Web3 Retrieval)',
                    'retrieval_method': 'web3'
                }
                
                # Save to cache
                save_to_cache(contract_address, source_data)
                return source_data
        except Exception as e:
            self.logger.warning(f"Web3.py source code retrieval failed: {e}")
        
        # If all methods fail
        self.logger.error(f"Failed to retrieve source code for {contract_address} after all attempts")
        return {}
    
    def run_slither_analysis(self, source_code: str, contract_name: str) -> Dict[str, Any]:
        """Run Slither static analysis"""
        import tempfile
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as temp_file:
                temp_file.write(source_code)
                temp_file_path = temp_file.name
            
            cmd = ['slither', temp_file_path, '--json', '-', '--print', 'detectors']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            vulnerabilities = []
            try:
                slither_json = json.loads(result.stdout) if result.stdout else {}
                vulnerabilities = slither_json.get('results', {}).get('detectors', [])
            except json.JSONDecodeError:
                self.logger.warning("Could not parse Slither JSON output")
            
            return {
                'success': result.returncode == 0,
                'vulnerabilities': vulnerabilities,
                'raw_output': result.stdout,
                'errors': result.stderr
            }
        
        except subprocess.TimeoutExpired:
            self.logger.error("Slither analysis timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"Slither analysis error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
    
    def run_mythril_analysis(self, source_code: str) -> Dict[str, Any]:
        """Run Mythril static analysis"""
        import tempfile
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as temp_file:
                temp_file.write(source_code)
                temp_file_path = temp_file.name
            
            cmd = ['mythril', 'analyze', temp_file_path, '--json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            try:
                mythril_json = json.loads(result.stdout) if result.stdout else {}
            except json.JSONDecodeError:
                mythril_json = {}
            
            return {
                'success': result.returncode == 0,
                'vulnerabilities': mythril_json.get('issues', []),
                'raw_output': result.stdout,
                'errors': result.stderr
            }
        
        except subprocess.TimeoutExpired:
            self.logger.error("Mythril analysis timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"Mythril analysis error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
    
    def generate_comprehensive_report(self, contract_address: str) -> Dict[str, Any]:
        """
        Generate comprehensive security analysis report
        
        Args:
            contract_address (str): Ethereum contract address
        
        Returns:
            Comprehensive security analysis report
        """
        # 1. Retrieve Source Code
        source_info = self.retrieve_source_code(contract_address)
        
        if not source_info.get('source_code'):
            return {
                'status': 'error',
                'message': 'Could not retrieve source code'
            }
        
        # 2. Run Multiple Analyses
        slither_results = self.run_slither_analysis(
            source_info['source_code'], 
            source_info['contract_name']
        )
        
        mythril_results = self.run_mythril_analysis(source_info['source_code'])
        
        # 3. Consolidate Vulnerabilities
        consolidated_vulnerabilities = self._consolidate_vulnerabilities(
            slither_results.get('vulnerabilities', []),
            mythril_results.get('vulnerabilities', [])
        )
        
        # 4. Calculate Risk Score
        risk_score = self._calculate_risk_score(
            slither_results, 
            mythril_results
        )
        
        # 5. Generate Comprehensive Report
        return {
            'contract_address': contract_address,
            'contract_name': source_info['contract_name'],
            'compiler_version': source_info['compiler_version'],
            'risk_score': risk_score,
            'risk_category': self._categorize_risk(risk_score),
            'vulnerabilities': consolidated_vulnerabilities,
            'tool_results': {
                'slither': slither_results,
                'mythril': mythril_results
            }
        }
    
    def _consolidate_vulnerabilities(self, slither_vulns: list, mythril_vulns: list) -> list:
        """
        Consolidate vulnerabilities from multiple tools
        
        Args:
            slither_vulns (list): Vulnerabilities from Slither
            mythril_vulns (list): Vulnerabilities from Mythril
        
        Returns:
            Consolidated list of unique vulnerabilities
        """
        consolidated = {}
        
        # Process Slither vulnerabilities
        for vuln in slither_vulns:
            key = vuln.get('check', vuln.get('title', 'Unknown'))
            if key not in consolidated:
                consolidated[key] = {
                    'type': key,
                    'severity': 'medium',
                    'detection_tools': ['Slither'],
                    'details': vuln
                }
        
        # Process Mythril vulnerabilities
        for vuln in mythril_vulns:
            key = vuln.get('title', 'Unknown')
            if key not in consolidated:
                consolidated[key] = {
                    'type': key,
                    'severity': 'medium',
                    'detection_tools': ['Mythril'],
                    'details': vuln
                }
            elif 'Mythril' not in consolidated[key]['detection_tools']:
                consolidated[key]['detection_tools'].append('Mythril')
        
        return list(consolidated.values())
    
    def _calculate_risk_score(self, slither_results: Dict, mythril_results: Dict) -> float:
        """
        Calculate comprehensive risk score
        
        Args:
            slither_results (Dict): Slither analysis results
            mythril_results (Dict): Mythril analysis results
        
        Returns:
            Calculated risk score (0-100)
        """
        base_score = 50.0  # Neutral starting point
        
        # Adjust score based on Slither findings
        if slither_results.get('vulnerabilities'):
            base_score -= len(slither_results['vulnerabilities']) * 5
        
        # Further adjust based on Mythril findings
        if mythril_results.get('vulnerabilities'):
            base_score -= len(mythril_results['vulnerabilities']) * 5
        
        return max(0, min(100, base_score))
    
    def _categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk based on score
        
        Args:
            risk_score (float): Calculated risk score
        
        Returns:
            Risk category string
        """
        if risk_score <= 20:
            return 'Low Risk'
        elif risk_score <= 50:
            return 'Moderate Risk'
        elif risk_score <= 80:
            return 'High Risk'
        else:
            return 'Critical Risk'

# Example usage
if __name__ == '__main__':
    analyzer = MultiToolAnalyzer()
    report = analyzer.generate_comprehensive_report('0x2e6E871Ec3D8112b37Ac5D34b0bc33F4069CF611')
    print(json.dumps(report, indent=2))