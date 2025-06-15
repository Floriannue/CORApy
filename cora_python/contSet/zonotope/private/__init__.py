"""
Private functions for zonotope operations

This package contains private helper functions used internally by zonotope methods.
These functions mirror the structure of the MATLAB private functions.
"""

from .priv_zonotopeContainment_pointContainment import priv_zonotopeContainment_pointContainment
from .priv_norm_exact import priv_norm_exact
from .priv_norm_ub import priv_norm_ub

__all__ = ['priv_zonotopeContainment_pointContainment', 'priv_norm_exact', 'priv_norm_ub'] 