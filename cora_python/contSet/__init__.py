"""
contSet package - Continuous set representations for CORA

This package provides various continuous set representations including
intervals, zonotopes, polytopes, and other geometric objects.
"""

from .contSet import ContSet
from .interval import Interval
from .zonotope import Zonotope

__all__ = ['ContSet', 'Interval', 'Zonotope'] 