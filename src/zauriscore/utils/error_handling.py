"""
Error handling utilities for ZauriScore.

This module provides decorators and context managers for consistent error handling.
"""

import functools
import logging
from typing import Any, Callable, Optional, Type, TypeVar, Tuple, Dict, cast
from contextlib import contextmanager

from ..exceptions import ZauriScoreError

T = TypeVar('T')
E = TypeVar('E', bound=Exception)


def handle_errors(
    exceptions: Tuple[Type[E], ...] = (Exception,),
    default: Any = None,
    log_level: int = logging.ERROR,
    raise_custom: Optional[Type[ZauriScoreError]] = None,
    **custom_kwargs: Any
) -> Callable:
    """
    Decorator to standardize error handling across the codebase.
    
    Args:
        exceptions: Exception types to catch
        default: Default value to return on error
        log_level: Logging level for errors
        raise_custom: Custom exception to raise
        **custom_kwargs: Additional kwargs for custom exception
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger = logging.getLogger(func.__module__)
                logger.log(
                    log_level,
                    f"Error in {func.__qualname__}: {str(e)}",
                    exc_info=log_level <= logging.DEBUG
                )
                
                if raise_custom:
                    raise raise_custom(**custom_kwargs) from e
                if default is not None:
                    return cast(T, default)
                raise
        return wrapper
    return decorator


def async_handle_errors(
    exceptions: Tuple[Type[E], ...] = (Exception,),
    default: Any = None,
    log_level: int = logging.ERROR,
    raise_custom: Optional[Type[ZauriScoreError]] = None,
    **custom_kwargs: Any
) -> Callable:
    """
    Async version of handle_errors decorator.
    
    Args:
        exceptions: Exception types to catch
        default: Default value to return on error
        log_level: Logging level for errors
        raise_custom: Custom exception to raise
        **custom_kwargs: Additional kwargs for custom exception
        
    Returns:
        Decorated async function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger = logging.getLogger(func.__module__)
                logger.log(
                    log_level,
                    f"Error in {func.__qualname__}: {str(e)}",
                    exc_info=log_level <= logging.DEBUG
                )
                
                if raise_custom:
                    raise raise_custom(**custom_kwargs) from e
                if default is not None:
                    return cast(T, default)
                raise
        return wrapper
    return decorator


@contextmanager
def resource_manager(
    resource: Optional[T] = None,
    cleanup: Optional[Callable[[], None]] = None,
    suppress_errors: bool = False
) -> T:
    """
    Generic context manager for resources that need cleanup.
    
    Args:
        resource: The resource to manage (optional)
        cleanup: Function to call on cleanup
        suppress_errors: If True, suppress errors during cleanup
        
    Yields:
        The managed resource
    """
    try:
        yield resource
    finally:
        if cleanup:
            try:
                cleanup()
            except Exception as e:
                if not suppress_errors:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning("Error during cleanup: %s", e, exc_info=True)


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases after each retry
        exceptions: Exception types to catch and retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs",
                            attempt, max_retries, func.__qualname__, delay
                        )
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                        
                    time.sleep(min(delay, max_delay))
                    delay *= backoff_factor
            
            raise last_exception or Exception("Unknown error in retry decorator")
        return wrapper
    return decorator
