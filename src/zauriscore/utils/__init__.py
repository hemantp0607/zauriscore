"""
ZauriScore Utils Package

This package contains utility functions and helper modules:
- Report Generator: Generate analysis reports
- Environment Checker: Verify tool dependencies
"""

from .report_generator import generate_report
from .check_slither_env import check_slither_installation

__all__ = ['generate_report', 'check_slither_installation']
