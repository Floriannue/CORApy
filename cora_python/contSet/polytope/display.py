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

if TYPE_CHECKING:
    from .polytope import Polytope


def display(P: 'Polytope', name: str = None) -> str:
    """
    Displays the properties of a polytope object, mirroring MATLAB behavior.
    Only shows computed representations without triggering conversions.
    
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
    if P._isVRep and P._V is not None:
        output_lines.append('Vertex representation:')
        output_lines.extend(displayMatrixVector(P._V, 'V'))
    else:
        output_lines.append('Vertex representation: (not computed)')
    output_lines.append("")

    # Display inequality constraints - check internal properties directly
    if P._A is None or P._A.size == 0:
        output_lines.append('Inequality constraints (A*x <= b): (none)')
    else:
        output_lines.append('Inequality constraints (A*x <= b):')
        output_lines.extend(displayMatrixVector(P._A, 'A'))
        output_lines.extend(displayMatrixVector(P._b, 'b'))
    output_lines.append("")

    # Display equality constraints
    if P._Ae is None or P._Ae.size == 0:
        output_lines.append('Equality constraints (Ae*x = be): (none)')
    else:
        output_lines.append('Equality constraints (Ae*x = be):')
        output_lines.extend(displayMatrixVector(P._Ae, 'Ae'))
        output_lines.extend(displayMatrixVector(P._be, 'be'))
    output_lines.append("")
    
    # Display set properties (if available)
    output_lines.append("Set properties:")
    output_lines.append(f"Bounded?                          {aux_prop2string(P.bounded)}")
    output_lines.append(f"Empty set?                        {aux_prop2string(P.emptySet)}") 
    output_lines.append(f"Full-dimensional set?             {aux_prop2string(P.fullDim)}")
    output_lines.append(f"Minimal halfspace representation? {aux_prop2string(P.minHRep)}")
    output_lines.append(f"Minimal vertex representation?    {aux_prop2string(P.minVRep)}")
    output_lines.append("")
    
    return "\n".join(output_lines) 