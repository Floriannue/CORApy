"""
empty - instantiates an empty zonotope bundle
"""

def empty(n: int = 0):
    """Instantiates an empty zonotope bundle of dimension n"""
    from .zonoBundle import ZonoBundle
    from cora_python.contSet.zonotope.zonotope import Zonotope
    if n < 0:
        raise ValueError('Dimension must be nonnegative')
    # For zero dimension, return an empty list of zonotopes (MATLAB-compatible)
    if n == 0:
        return ZonoBundle([])
    # Otherwise, represent empty bundle by a single empty zonotope to carry dimension
    Z_empty = Zonotope.empty(n)
    return ZonoBundle([Z_empty])