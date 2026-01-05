"""
display - Displays the properties of a contSet object (dimension, location
    of center, ...)

Syntax:
    display(S)

Inputs:
    S - contSet object

Outputs:
    -

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff, Mark Wetzlinger
Written:       30-September-2006
Last update:   02-May-2020 (MW, standardized to other classes)
               14-December-2022 (TL, property check in subclass)
               25-November-2022 (MW, property check in subclass)
Last revision: ---
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.contSet.contSet import ContSet

def display_(S: 'ContSet') -> str:
    """
    Displays the properties of a contSet object (dimension, location of center, ...)
    (Internal function that returns string)
    
    Args:
        S: contSet object
        
    Returns:
        str: string representation of the object
    """
    
    # Check if object is empty
    if hasattr(S, 'isemptyobject') and S.isemptyobject():
        return f"{S.__class__.__name__}(empty)"
    
    # Dimension
    n = S.dim()
    
    # Basic information
    result = f"{S.__class__.__name__}({n}D)\n"
    
    # Display center if available
    if hasattr(S, 'center'):
        try:
            c = S.center()
            if c is not None:
                result += f"  center: {c.flatten()}\n"
        except:
            pass
    
    # Display additional properties based on the type
    class_name = S.__class__.__name__.lower()
    
    if class_name == 'interval':
        if hasattr(S, 'inf') and hasattr(S, 'sup'):
            result += f"  infimum: {S.inf()}\n"
            result += f"  supremum: {S.sup()}\n"
    elif class_name == 'zonotope':
        if hasattr(S, 'generators'):
            result += f"  generators: {S.generators().shape[1]} columns\n"
    elif class_name == 'polytope':
        if hasattr(S, 'V') and S.V is not None:
            result += f"  vertices: {S.V.shape[1]} columns\n"
        if hasattr(S, 'A') and S.A is not None:
            result += f"  constraints: {S.A.shape[0]} rows\n"
    elif class_name == 'ellipsoid':
        if hasattr(S, 'Q'):
            result += f"  shape matrix: {S.Q.shape}\n"
    
    return result.rstrip()


def display(S: 'ContSet') -> None:
    """
    Displays the properties of a contSet object (prints to stdout)
    
    Args:
        S: contSet object
    """
    print(display_(S), end='') 