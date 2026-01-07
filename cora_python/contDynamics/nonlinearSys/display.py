"""
display - displays a nonlinearSys object on the command window

Syntax:
    display(nlnsys)

Inputs:
    nlnsys - nonlinearSys object

Outputs:
    ---

Example:
    f = lambda x, u: np.array([x[1], (1-x[0]**2)*x[1]-x[0]])
    nlnsys = NonlinearSys('vanDerPol', f)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger
Written:       17-October-2007
Last update:   19-June-2022
               23-November-2022 (TL, dispInput)
Last revision: ---
Python translation: 2025
"""

import numpy as np
import sympy as sp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nonlinearSys import NonlinearSys

from cora_python.contDynamics.contDynamics.display import display_ as contDynamics_display
from cora_python.contDynamics.contDynamics.symVariables import symVariables
from cora_python.g.functions.verbose.display.display_matrix_vector import display_matrix_vector


def display_(nlnsys: 'NonlinearSys', var_name: str = None) -> str:
    """
    Displays a nonlinearSys object (internal function that returns string)
    
    Args:
        nlnsys: NonlinearSys object
        var_name: Optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    # Display parent object
    parent_str = contDynamics_display(nlnsys)
    lines.append(parent_str)
    lines.append("")
    
    # Display type
    lines.append("Type: Nonlinear continuous-time system")
    lines.append("")
    
    # Create symbolic variables
    vars_dict, _ = symVariables(nlnsys)
    
    # Insert symbolic variables into the system equations
    try:
        f = nlnsys.mFile(vars_dict['x'], vars_dict['u'])
        
        # Convert to list if single element
        if isinstance(f, (sp.Matrix, np.ndarray)):
            if f.size == 1:
                f = [f]
            else:
                f = [f[i] for i in range(len(f))]
        elif not isinstance(f, (list, tuple)):
            f = [f]
        
        # Display state space equations
        lines.append("State-space equations:")
        for i, fi in enumerate(f):
            # Convert symbolic expression to string
            if isinstance(fi, sp.Basic):
                fi_str = str(fi)
            else:
                fi_str = str(fi)
            lines.append(f"  f({i+1}) = {fi_str}")
        lines.append("")
    except Exception as e:
        # If symbolic evaluation fails, show function handle info
        lines.append("State-space equations:")
        if hasattr(nlnsys.mFile, '__name__'):
            lines.append(f"  Dynamic function: {nlnsys.mFile.__name__}")
        else:
            lines.append("  Dynamic function: anonymous")
        lines.append("")
    
    # Insert symbolic variables into the output equations
    try:
        y = nlnsys.out_mFile(vars_dict['x'], vars_dict['u'])
        
        # Convert to list if single element
        if isinstance(y, (sp.Matrix, np.ndarray)):
            if y.size == 1:
                y = [y]
            else:
                y = [y[i] for i in range(len(y))]
        elif not isinstance(y, (list, tuple)):
            y = [y]
        
        # Display output equations
        lines.append("Output equations:")
        for i, yi in enumerate(y):
            # Convert symbolic expression to string
            if isinstance(yi, sp.Basic):
                yi_str = str(yi)
            else:
                yi_str = str(yi)
            lines.append(f"  y({i+1}) = {yi_str}")
        lines.append("")
    except Exception as e:
        # If symbolic evaluation fails, show function handle info
        lines.append("Output equations:")
        if hasattr(nlnsys.out_mFile, '__name__'):
            lines.append(f"  Output function: {nlnsys.out_mFile.__name__}")
        else:
            lines.append("  Output function: anonymous")
        lines.append("")
    
    return "\n".join(lines)


def display(nlnsys: 'NonlinearSys', var_name: str = None) -> None:
    """
    Displays a nonlinearSys object (prints to stdout)
    
    Args:
        nlnsys: NonlinearSys object
        var_name: Optional variable name
    """
    print(display_(nlnsys, var_name), end='')

