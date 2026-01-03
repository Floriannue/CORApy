"""
abstractReset - AbstractReset package for hybrid automata

This package contains the AbstractReset class and all its methods.
Each method is implemented in its own file following the MATLAB structure.
"""

from .abstractReset import AbstractReset
from .isequal import isequal
from .eq import eq
from .ne import ne

# Attach methods to the AbstractReset class
AbstractReset.isequal = isequal
AbstractReset.eq = eq
AbstractReset.ne = ne

# Map to Python operators
def __eq__(self, other):
    """Python == operator"""
    return eq(self, other) if isinstance(other, AbstractReset) else False

def __ne__(self, other):
    """Python != operator"""
    return ne(self, other) if isinstance(other, AbstractReset) else True

AbstractReset.__eq__ = __eq__
AbstractReset.__ne__ = __ne__

__all__ = ['AbstractReset', 'isequal', 'eq', 'ne']

