"""
converter - MATLAB converter functions

This module contains functions for converting between different data formats
and calling optimization solvers, translated from MATLAB to Python.

Authors: MATLAB original authors
         Python translation by AI Assistant
"""

from .mat2vec import mat2vec
from .vec2mat import vec2mat
from .CORAlinprog import CORAlinprog

__all__ = [
    'mat2vec',
    'vec2mat',
    'CORAlinprog'
] 