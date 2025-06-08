"""
contSet package - Continuous set representations for CORA

This package provides various continuous set representations including
intervals, zonotopes, polytopes, and other geometric objects.
"""

from .contSet import contSet
from .interval import interval
from .zonotope import zonotope

__all__ = ['contSet', 'interval', 'zonotope'] 