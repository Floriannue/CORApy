"""
empty - instantiates an empty level set

Syntax:
    ls = levelSet.empty(n)

Inputs:
    n - dimension

Outputs:
    ls - empty level set

Example: 
    ls = levelSet.empty(2);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       09-January-2024
Last update:   15-January-2024 (TL, parse input)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Import symbolic computation capabilities
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

def empty(n: int = 0):
    """
    Instantiates an empty level set
    
    Args:
        n: Dimension of the empty level set (default: 0)
        
    Returns:
        LevelSet: Empty level set object
        
    Example:
        >>> ls = LevelSet.empty(2)
    """
    # Parse input
    if n is None:
        n = 0
    
    # Input validation
    inputArgsCheck([(n, 'att', 'numeric', ['scalar', 'nonnegative'])])
    
    if not SYMPY_AVAILABLE:
        raise CORAerror('CORA:specialError', 'LevelSet.empty requires sympy to be installed.')
    
    # Import here to avoid circular import
    from .levelSet import LevelSet
    
    # Create symbolic variables x of dimension n
    # MATLAB: vars = sym('x',[n,1]);
    if n == 0:
        vars_ = []
    elif n == 1:
        vars_ = [sp.Symbol('x0')]
    else:
        vars_ = [sp.Symbol(f'x{i}') for i in range(n)]
    
    # Create equation eq = 1 (always true)
    # MATLAB: eq = sym(1);
    eq = sp.Integer(1)
    
    # Create levelSet with eq <= 0 (1 <= 0 is always false, so set is empty)
    # MATLAB: ls = levelSet(eq,vars,{"<="});
    return LevelSet(eq, vars_, '<=')

