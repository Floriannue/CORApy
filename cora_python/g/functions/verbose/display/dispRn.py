"""
dispRn - Displays a fullspace object on the command window

Syntax:
    dispRn(fs, varName)

Inputs:
    fs - fullspace object
    varName - optional variable name

Outputs:
    (to console)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/display

Authors:       Mark Wetzlinger
Written:       02-May-2020
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Optional, Any


def dispRn(fs: Any, varName: Optional[str] = None) -> str:
    """
    Displays a fullspace object (returns string)
    
    Args:
        fs: fullspace object
        varName: optional variable name
        
    Returns:
        str: String representation
    """
    lines = []
    
    lines.append("")
    if varName:
        lines.append(f"{varName} =")
        lines.append("")
    
    dim_val = fs.dim() if hasattr(fs, 'dim') and callable(fs.dim) else (fs.dimension if hasattr(fs, 'dimension') else 0)
    lines.append(f"fullspace:")
    lines.append(f"- dimension: {dim_val}")
    lines.append(f"- {dim_val}-dimensional full space")
    
    return "\n".join(lines)

