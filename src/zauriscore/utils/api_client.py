"""HTTP client with retry and timeout support."""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, Callable

import requests
from requests.adapters import HTTPAdapter, Retry

from ..config import config
from ..utils.logger import setup_logger

class APIClient:
    """HTTP client with retry and timeout support."""
    
    def __init__(
        self,
        base_url: str = "",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: int = 30,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for all requests
            api_key: API key for authentication
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            timeout: Request timeout in seconds
            cache_dir: Directory to cache responses (if None, caching is disabled)
        """
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.api_key = api_key
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = setup_logger("api.client")
        
        # Set up session with retry strategy
        self.session = self._create_session(max_retries, backoff_factor)
    
    def _create_session(self, max_retries: int, backoff_factor: float) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        # Mount the retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'ZauriScore/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Add API key if provided
        if self.api_key:
            session.headers['Authorization'] = f'Bearer {self.api_key}'
        
        return session
    
    def _get_cache_path(self, url: str, params: Optional[Dict] = None) -> Optional[Path]:
        """Get the cache path for a request."""
        if not self.cache_dir:
            return None
            
        # Create a cache key from the URL and params
        cache_key = f"{url}_{json.dumps(params, sort_keys=True) if params else ''}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        
        return self.cache_dir / f"{cache_key}.json"
    
    def _get_cached_response(self, cache_path: Path) -> Optional[Dict]:
        """Get a cached response if it exists and is fresh."""
        if not cache_path.exists():
            return None
            
        # Check if cache is fresh (24 hours)
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > 24 * 3600:  # 24 hours
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None
    
    def _cache_response(self, cache_path: Path, data: Dict) -> None:
        """Cache a response."""
        if not self.cache_dir:
            return
            
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (IOError, TypeError) as e:
            self.logger.warning(f"Failed to cache response to {cache_path}: {e}")
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """Send an HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            json_data: JSON request body
            headers: Additional headers
            use_cache: Whether to use cached responses if available
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        cache_path = self._get_cache_path(url, params) if use_cache else None
        
        # Try to get cached response
        if use_cache and cache_path:
            cached = self._get_cached_response(cache_path)
            if cached is not None:
                self.logger.debug(f"Using cached response for {url}")
                return cached
        
        # Prepare request
        req_headers = self.session.headers.copy()
        if headers:
            req_headers.update(headers)
        
        self.logger.debug(f"Sending {method} request to {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=req_headers,
                timeout=self.timeout
            )
            
            # Raise an exception for 4XX/5XX responses
            response.raise_for_status()
            
            # Parse JSON response
            try:
                result = response.json()
            except ValueError:
                result = {'data': response.text}
            
            # Cache the response if successful
            if cache_path and response.status_code == 200:
                self._cache_response(cache_path, result)
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """Send a GET request."""
        return self.request('GET', endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """Send a POST request."""
        return self.request('POST', endpoint, json_data=json_data, **kwargs)
    
    def put(self, endpoint: str, json_data: Optional[Dict] = None, **kwargs) -> Dict:
        """Send a PUT request."""
        return self.request('PUT', endpoint, json_data=json_data, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Dict:
        """Send a DELETE request."""
        return self.request('DELETE', endpoint, **kwargs)
    
    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
