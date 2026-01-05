"""
display - Displays the properties of a levelSet object on the command
    window

Syntax:
    display(ls)

Inputs:
    ls - levelSet object

Outputs:
    ---

Example: 
    syms x y
    eq = x^2 + y^2 - 4;
    ls = levelSet(eq,[x;y],'==');
    display(ls);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       09-June-2020
Last update:   03-March-2022 (MP, levelSets with multiple equations)
               15-May-2023 (MW, fix printed string)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from cora_python.contSet.contSet.display import display_ as contSet_display

if TYPE_CHECKING:
    from .levelSet import LevelSet


def display_(ls: 'LevelSet', var_name: str = None) -> str:
    """
    Displays the properties of a levelSet object (internal function that returns string)
    
    Args:
        ls: levelSet object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Display input variable
    lines.append("")
    if var_name is None:
        var_name = 'ls'
    lines.append(f"{var_name} =")
    lines.append("")
    
    # Display dimension
    contSet_str = contSet_display(ls)
    lines.append(contSet_str)
    lines.append("")
    
    # String of variables in level set
    if hasattr(ls, 'vars') and ls.vars is not None:
        if isinstance(ls.vars, (list, tuple, np.ndarray)):
            varsPrintStr = ",".join(str(v) for v in ls.vars)
        else:
            varsPrintStr = str(ls.vars)
    else:
        varsPrintStr = ""
    
    # Number of concatenated level sets
    if hasattr(ls, 'eq'):
        if isinstance(ls.eq, (list, tuple, np.ndarray)):
            numSets = len(ls.eq)
        else:
            numSets = 1
    else:
        numSets = 0
    
    if numSets == 1:
        eq_str = str(ls.eq) if hasattr(ls, 'eq') else ""
        compOp = ls.compOp if hasattr(ls, 'compOp') else "=="
        lines.append(f"  f({varsPrintStr}): {eq_str} {compOp} 0")
        lines.append("")
    elif numSets > 1:
        # Loop over all level sets
        for i in range(numSets):
            # Only use & sign for all but last equation
            andStr = ""
            if i != numSets - 1:
                andStr = " & "
            
            # Display i-th equation
            eq_i = ls.eq[i] if isinstance(ls.eq, (list, tuple, np.ndarray)) else ls.eq
            compOp_i = ls.compOp[i] if isinstance(ls.compOp, (list, tuple, np.ndarray)) else ls.compOp
            
            # Try to format with limited precision (like MATLAB's vpa)
            try:
                import sympy as sp
                if hasattr(eq_i, 'evalf'):
                    eq_str = str(sp.N(eq_i, 3))
                else:
                    eq_str = str(eq_i)
            except:
                eq_str = str(eq_i)
            
            lines.append(f"  f{i+1}({varsPrintStr}): {eq_str} {compOp_i} 0{andStr}")
        lines.append("")
    
    return "\n".join(lines)


def display(ls: 'LevelSet', var_name: str = None) -> None:
    """
    Displays the properties of a levelSet object (prints to stdout)
    
    Args:
        ls: levelSet object
        var_name: Optional variable name
    """
    print(display_(ls, var_name), end='')

