"""
Helper functions for contSet operations

This module provides helper functions for working with contSet objects,
including block operations and set operations.
"""

from .block_mtimes import block_mtimes
from .block_operation import block_operation
from .enclose import enclose

__all__ = ['block_mtimes', 'block_operation', 'enclose'] 