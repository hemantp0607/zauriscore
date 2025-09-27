"""
Asynchronous HTTP client with retry, timeout, and security features.

This module provides an async HTTP client built on aiohttp with features like:
- Automatic retry with exponential backoff
- Request/response logging
- Security protections (SSRF, rate limiting)
- Type hints and async/await support
"""

import asyncio
import ipaddress
import json
import logging
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union, List, Type, TypeVar, Tuple, cast
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout

from ..config import settings
from ..exceptions import (
    APIClientError,
    APITimeoutError,
    APIConnectionError,
    APISecurityError,
    APIResponseError,
    ValidationError
)
from .logger import get_logger
from .error_handling import async_handle_errors, resource_manager

# Type variable for generic response handling
T = TypeVar('T')

# Re-export exceptions for backward compatibility
AsyncAPIClientError = APIClientError

class AsyncAPIClient:
    """Asynchronous HTTP client with retry and security features."""
    
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        rate_limit: Optional[int] = None,
        rate_window: int = 60,
        blacklisted_ips: Optional[Set[str]] = None,
        allowed_domains: Optional[Set[str]] = None,
        verify_ssl: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the async HTTP client.
        
        Args:
            base_url: Base URL for all requests
            timeout: Default timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
            rate_limit: Max requests per rate window (None for no limit)
            rate_window: Rate limit window in seconds
            blacklisted_ips: Set of blacklisted IP addresses
            allowed_domains: Set of allowed domains (if None, all domains allowed)
            verify_ssl: Whether to verify SSL certificates
            logger: Custom logger instance
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            self.base_url = base_url.rstrip('/')
            self.timeout = ClientTimeout(total=timeout)
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.backoff_factor = backoff_factor
            self.rate_limit = rate_limit
            self.rate_window = rate_window
            self.verify_ssl = verify_ssl
            self._session = None
            self._request_timestamps = []
            self._blacklisted_ips = set(blacklisted_ips) if blacklisted_ips else set()
            self._allowed_domains = set(allowed_domains) if allowed_domains else None
            self.logger = logger or get_logger('api.client')
            
            # Validate configuration
            self._validate_config()
            
            # Initialize rate limiting
            self._rate_semaphore = asyncio.Semaphore(rate_limit if rate_limit else 1000)
            self._rate_cleanup_task = None
            
        except Exception as e:
            raise ValidationError(f"Invalid client configuration: {str(e)}") from e
    
    def _validate_config(self) -> None:
        """Validate the client configuration.
        
        Raises:
            ValidationError: If any configuration is invalid
        """
        if self.max_retries < 0:
            raise ValidationError("max_retries must be >= 0")
        if self.retry_delay <= 0:
            raise ValidationError("retry_delay must be > 0")
        if self.backoff_factor < 1.0:
            raise ValidationError("backoff_factor must be >= 1.0")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValidationError("rate_limit must be > 0 or None")
        if self.rate_window <= 0:
            raise ValidationError("rate_window must be > 0")
            
        # Validate blacklisted IPs
        for ip in self._blacklisted_ips:
            try:
                ipaddress.ip_address(ip)
            except ValueError as e:
                raise ValidationError(f"Invalid IP address in blacklist: {ip}") from e
                
        self._blacklisted_networks = [ipaddress.ip_network(ip) for ip in self._blacklisted_ips]
        

    async def __aenter__(self) -> 'AsyncAPIClient':
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=self.timeout,
            json_serialize=json.dumps,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _is_ip_allowed(self, ip: str) -> bool:
        """Check if an IP address is allowed."""
        try:
            addr = ipaddress.ip_address(ip)
            return not any(addr in net for net in self._blacklisted_networks)
        except ValueError:
            return False

    async def _resolve_host(self, hostname: str) -> List[str]:
        """Resolve a hostname to IP addresses."""
        try:
            return [ip[4][0] for ip in await asyncio.get_event_loop().getaddrinfo(
                hostname, None, proto=socket.IPPROTO_TCP
            )]
        except (socket.gaierror, OSError) as e:
            raise APIConnectionError(f"Failed to resolve {hostname}: {e}")

    @async_handle_errors(
        error_mapping={
            socket.gaierror: lambda e: APISecurityError(f"Failed to resolve hostname: {str(e)}"),
            Exception: lambda e: APISecurityError(f"Security check failed: {str(e)}")
        }
    )
    async def _check_url_security(self, url: str) -> None:
        """Check if a URL is allowed based on security rules.
        
        Args:
            url: The URL to check
            
        Raises:
            APISecurityError: If the URL violates security rules
        """
        parsed = urlparse(url)
        
        # Check if domain is allowed
        if self._allowed_domains and parsed.hostname not in self._allowed_domains:
            raise APISecurityError(f"Access to domain {parsed.hostname} is not allowed")
            
        # Check for SSRF attempts
        if parsed.hostname in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
            raise APISecurityError("Access to localhost is not allowed")
            
        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private:
                raise APISecurityError(f"Access to private IP {ip} is not allowed")
            if any(ip in network for network in self._blacklisted_networks):
                raise APISecurityError(f"Access to blacklisted IP {ip} is not allowed")
        except ValueError:
            # Not an IP address, resolve hostname
            ips = await self._resolve_host(parsed.hostname)
            for ip in ips:
                if ip.is_private:
                    raise APISecurityError(f"Hostname resolves to private IP: {ip}")
                if any(ip in network for network in self._blacklisted_networks):
                    raise APISecurityError(f"Hostname resolves to blacklisted IP: {ip}")

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> ClientResponse:
        """Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request (can be relative to base_url)
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.request()
            
        Returns:
            ClientResponse: The response object
            
        Raises:
            APITimeoutError: If the request times out
            APIConnectionError: If there's a connection error
            APIResponseError: If the API returns an error response
            APISecurityError: If a security check fails
        """
        if not url.startswith(('http://', 'https://')):
            url = f"{self.base_url}/{url.lstrip('/')}"
        
        # Check URL security
        await self._check_url_security(url)
        
        # Apply rate limiting
        await self._check_rate_limit()
        
        # Set default headers if not provided
        headers = kwargs.pop('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = f"ZauriScore/1.0"
        if 'Accept' not in headers:
            headers['Accept'] = 'application/json'
            
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        # Configure SSL verification
        if 'ssl' not in kwargs:
            kwargs['ssl'] = self.verify_ssl
        
        # Track request timestamp for rate limiting
        self._request_timestamps.append(asyncio.get_event_loop().time())
        
        # Make the request
        response = await self.session.request(method, url, headers=headers, **kwargs)
        
        # Log the request
        self._log_request(method, url, response.status)
        
        # Check for rate limiting
        if response.status == 429:
            retry_after = self._get_retry_after(response)
            self.logger.warning("Rate limited, retrying after %d seconds", retry_after)
            await asyncio.sleep(retry_after)
            return await self._make_request(method, url, **kwargs)
            
        # Check for server errors
        if response.status >= 500:
            error_data = await self._get_error_data(response)
            raise APIResponseError(
                status=response.status,
                message=error_data.get('message', response.reason),
                response_data=error_data
            )
            
        # Check for client errors
        if response.status >= 400:
            error_data = await self._get_error_data(response)
            raise APIResponseError(
                status=response.status,
                message=error_data.get('message', response.reason),
                response_data=error_data
            )
            
        return response

    @async_handle_errors(
        error_mapping={
            ValueError: lambda e: APIResponseError("Invalid response format", status=500),
            aiohttp.ContentTypeError: lambda e: APIResponseError("Invalid content type in response", status=500)
        }
    )
    async def _handle_response(
        self, 
        response: ClientResponse, 
        require_ok: bool = True
    ) -> Dict[str, Any]:
        """Handle the API response and return parsed JSON.
        
        Args:
            response: The response object from aiohttp
            require_ok: If True, raises an exception for non-2xx responses
            
        Returns:
            Dict containing the parsed JSON response
            
        Raises:
            APIResponseError: If require_ok is True and the response is not successful
            ValueError: If the response cannot be parsed as JSON
            aiohttp.ContentTypeError: If the response content type is not JSON
        """
        if not response.content_length:
            return {}
            
        response_data = await response.json()
        
        if require_ok and not response.ok:
            raise APIResponseError(
                status=response.status,
                message=response_data.get("message", response.reason),
                response_data=response_data
            )
            
        return response_data

    # Convenience methods
    async def get(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        require_ok: bool = True,
    ) -> Dict[str, Any]:
        """Make a GET request."""
        return await self._make_request(
            "GET", endpoint, params=params, headers=headers, require_ok=require_ok
        )

    async def post(
        self,
        endpoint: str,
        *,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        require_ok: bool = True,
    ) -> Dict[str, Any]:
        """Make a POST request."""
        return await self._make_request(
            "POST", endpoint, json_data=json_data, headers=headers, require_ok=require_ok
        )

    async def put(
        self,
        endpoint: str,
        *,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        require_ok: bool = True,
    ) -> Dict[str, Any]:
        """Make a PUT request."""
        return await self._make_request(
            "PUT", endpoint, json_data=json_data, headers=headers, require_ok=require_ok
        )

    async def delete(
        self,
        endpoint: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        require_ok: bool = True,
    ) -> Dict[str, Any]:
        """Make a DELETE request."""
        return await self._make_request(
            "DELETE", endpoint, headers=headers, require_ok=require_ok
        )

    async def get_paginated(
        self,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data_key: str = "data",
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Make a paginated GET request, automatically fetching all pages.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            data_key: Key in the response that contains the data array
            page_size: Number of items per page
            
        Returns:
            Combined response with all pages of data
        """
        params = params or {}
        params["page"] = 1
        params["page_size"] = page_size
        
        first_page = await self.get(endpoint, params=params, require_ok=True)
        results = first_page.get(data_key, [])
        total_pages = first_page.get("total_pages", 1)
        
        if total_pages > 1:
            tasks = []
            for page in range(2, total_pages + 1):
                page_params = params.copy()
                page_params["page"] = page
                tasks.append(self.get(endpoint, params=page_params, require_ok=True))
            
            # Fetch all pages concurrently
            pages = await asyncio.gather(*tasks)
            for page in pages:
                results.extend(page.get(data_key, []))
        
        return {
            **first_page,
            data_key: results,
            "total": len(results),
        }

# Example usage:
"""
async def example():
    async with AsyncAPIClient("https://api.example.com") as client:
        try:
            # Make a GET request
            data = await client.get("/endpoint", params={"key": "value"})
            
            # Make a POST request
            result = await client.post(
                "/submit",
                json_data={"key": "value"},
                headers={"Authorization": "Bearer token"}
            )
            print(result)
            
            # Get paginated data
            all_items = await client.get_paginated("/items", page_size=50)
            print(f"Fetched {len(all_items['data'])} items")
            
        except AsyncAPIClientError as e:
            print(f"API error: {e}")
"""
