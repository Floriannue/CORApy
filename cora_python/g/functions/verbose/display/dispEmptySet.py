"""
dispEmptySet - displays text for empty objects

Syntax:
    dispEmptySet(S)
    dispEmptySet(S, argname)

Inputs:
    S - contSet object
    argname - name of obj in workspace (optional)

Outputs:
    -

Example:
    I = Interval.empty(2)
    dispEmptySet(I, "I")

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: display

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       10-January-2024 (MATLAB)
Python translation: 2025
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet


def dispEmptySet(S: 'ContSet', argname: Optional[str] = None) -> str:
    """
    Displays text for empty objects
    
    Args:
        S: contSet object
        argname: name of obj in workspace (optional)
        
    Returns:
        str: formatted display string
    """
    result = ""
    
    if argname:
        result += f"\n{argname} =\n\n"
    
    # Display empty set information
    class_name = S.__class__.__name__
    dimension = S.dim()
    result += f"  {dimension}-dimensional empty set (represented as {class_name})\n"
    
    return result 