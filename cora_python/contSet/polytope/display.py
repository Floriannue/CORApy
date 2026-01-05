"""
display - Displays the properties of a polytope object (inequality and
   equality constraints) on the command window

Syntax:
   display(P)

Inputs:
   P - polytope object

Outputs:
   (to console)

Example: 
   A = np.array([[1, 2], [-1, 2], [-2, -2], [1, -2]])
   b = np.ones((4, 1))
   P = Polytope(A, b)
   display(P)

Authors:       Viktor Kotsev, Mark Wetzlinger
Written:       06-June-2022
Last update:   01-December-2022 (MW, adapt to other CORA display)
Last revision: ---
"""

from typing import TYPE_CHECKING
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

if TYPE_CHECKING:
    from .polytope import Polytope


def display_(P: 'Polytope', name: str = None) -> str:
    """
    Displays the properties of a polytope object, mirroring MATLAB behavior.
    Only shows computed representations without triggering conversions.
    (Internal function that returns string)
    
    Args:
        P: Polytope object
        name: Variable name to display
        
    Returns:
        String representation for display
    """
    
    def displayMatrixVector(mat, mat_name):
        """Helper function to format matrix/vector display"""
        lines = []
        if mat is None or mat.size == 0:
            return lines
            
        # Format similar to MATLAB
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
            
        lines.append(f"    {mat_name} =")
        
        # Simple formatting for matrices
        for i in range(mat.shape[0]):
            if mat.shape[1] == 1:
                lines.append(f"        {mat[i, 0]:8.4f}")
            else:
                row_str = "        " + "  ".join(f"{mat[i, j]:8.4f}" for j in range(mat.shape[1]))
                lines.append(row_str)
        
        return lines

    def aux_prop2string(prop_val):
        """Helper function to display three-valued logic in set properties"""
        if prop_val is None:
            return 'Unknown'
        elif prop_val:
            return 'true'
        else:
            return 'false'

    output_lines = []
    output_lines.append("")
    
    # If no name provided, try to get variable name (like MATLAB inputname)
    if name is None:
        name = "ans"  # Default like MATLAB
    
    output_lines.append(f"{name} =")
    output_lines.append("")
    
    # Display dimension (from base class)
    output_lines.append(f"Polytope object with dimension: {P.dim()}")
    output_lines.append("")

    # Display vertex representation - check flag first
    if P.isVRep:
        try:
            if P.V is not None:
                output_lines.append('Vertex representation:')
                output_lines.extend(displayMatrixVector(P.V, 'V'))
            else:
                output_lines.append('Vertex representation: (none)') # Should not happen if isVRep is true
        except CORAerror: # Catch CORAerror if V is not available for some reason
            output_lines.append('Vertex representation: (not computed)')
    else:
        output_lines.append('Vertex representation: (not computed)')
    output_lines.append("")

    # Display inequality constraints - check internal properties directly
    if P.isHRep:
        try:
            if P.A is None or P.A.size == 0:
                output_lines.append('Inequality constraints (A*x <= b): (none)')
            else:
                output_lines.append('Inequality constraints (A*x <= b):')
                output_lines.extend(displayMatrixVector(P.A, 'A'))
                output_lines.extend(displayMatrixVector(P.b, 'b'))
        except CORAerror: # Catch CORAerror if A/b are not available
            output_lines.append('Inequality constraints (A*x <= b): (not computed)')
    else:
        output_lines.append('Inequality constraints (A*x <= b): (not computed)')
    output_lines.append("")

    # Display equality constraints
    if P.isHRep:
        try:
            if P.Ae is None or P.Ae.size == 0:
                output_lines.append('Equality constraints (Ae*x = be): (none)')
            else:
                output_lines.append('Equality constraints (Ae*x = be):')
                output_lines.extend(displayMatrixVector(P.Ae, 'Ae'))
                output_lines.extend(displayMatrixVector(P.be, 'be'))
        except CORAerror: # Catch CORAerror if Ae/be are not available
            output_lines.append('Equality constraints (Ae*x = be): (not computed)')
    else:
        output_lines.append('Equality constraints (Ae*x = be): (not computed)')
    output_lines.append("")
    
    # Display set properties (if available)
    output_lines.append("Set properties:")
    
    # Use method interface for computed properties (like MATLAB)
    bounded_result = P.isBounded()
    output_lines.append(f"Bounded?                          {aux_prop2string(bounded_result)}")

    empty_result = P.isemptyobject()
    output_lines.append(f"Empty set?                        {aux_prop2string(empty_result)}")

    fullDim_result = P.isFullDim()
    output_lines.append(f"Full-dimensional set?             {aux_prop2string(fullDim_result)}")

    # These are cached properties, not computed methods
    output_lines.append(f"Minimal halfspace representation? {aux_prop2string(P.minHRep)}")
    output_lines.append(f"Minimal vertex representation?    {aux_prop2string(P.minVRep)}")
    output_lines.append("")
    
    return "\n".join(output_lines)


def display(P: 'Polytope', name: str = None) -> None:
    """
    Displays the properties of a polytope object (prints to stdout)
    
    Args:
        P: Polytope object
        name: Variable name to display
    """
    print(display_(P, name), end='') 