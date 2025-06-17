"""
Specification module for CORA

This module contains classes for temporal logic specifications
and their verification in reachability analysis.
"""

from .specification import Specification, create_safety_specification, create_invariant_specification, create_unsafe_specification
from .add import add
from .check import check
from .eq import eq
from .isequal import isequal
from .inverse import inverse
from .isempty import isempty
from .ne import ne
from .project import project
from .splitLogic import splitLogic

__all__ = [
    'Specification', 'create_safety_specification', 'create_invariant_specification', 
    'create_unsafe_specification', 'add', 'check', 'eq', 'isequal', 'inverse', 
    'isempty', 'ne', 'project', 'splitLogic'
] 