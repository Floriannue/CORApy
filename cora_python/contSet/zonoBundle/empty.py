"""
empty - instantiates an empty zonotope bundle
"""

def empty(n: int = 0):
    """Instantiates an empty zonotope bundle"""
    from .zonoBundle import ZonoBundle
    return ZonoBundle([]) 