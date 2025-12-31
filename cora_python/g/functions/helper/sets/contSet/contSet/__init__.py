"""
Helper functions for contSet operations

This module provides helper functions for working with contSet objects,
including block operations and set operations.
"""

from .block_mtimes import block_mtimes
from .block_operation import block_operation
from .enclose import enclose
from .reorder_numeric import reorder_numeric
from .lin_error2dAB import lin_error2dAB

__all__ = ['block_mtimes', 'block_operation', 'enclose', 'reorder_numeric', 'lin_error2dAB'] 