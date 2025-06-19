from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .polytope import Polytope

def display(P: 'Polytope', name: str = "P") -> str:
    """
    Creates a string representation of a polytope object's properties,
    mirroring the format of the MATLAB CORA library's display function.
    """
    
    def displayMatrixVector(mat, mat_name):
        lines = []
        header = f"  {mat_name} ="
        lines.append(header)
        
        if mat is None or mat.size == 0:
            lines.append("     []")
            return lines

        # Define a custom formatter to match MATLAB's output style
        float_formatter = lambda x: f"{x: >1.4f}".rstrip('0').rstrip('.') if x != 0 else " 0."
        
        # Use np.array2string with the custom formatter
        mat_str = np.array2string(
            mat.astype(float),
            precision=4,
            suppress_small=True,
            prefix='     ',
            formatter={'float_kind': float_formatter}
        )

        lines.append("     " + mat_str)
        return lines

    output_lines = [f"\n{name} =", ""]
    output_lines.append(f"Polytope object with dimension: {P.dim()}\n")

    # Display vertex representation
    try:
        V = P._V
        if P._has_v_rep:
            output_lines.append('Vertex representation:')
            output_lines.extend(displayMatrixVector(V, 'V'))
        else:
            output_lines.append('Vertex representation: (not computed)')
    except AttributeError:
        output_lines.append('Vertex representation: (not computed)')
    output_lines.append("")

    # Display inequality constraints
    try:
        A, b = P.A, P.b
        if A is None or A.size == 0:
            output_lines.append('Inequality constraints (A*x <= b): (none)')
        else:
            output_lines.append('Inequality constraints (A*x <= b):')
            output_lines.extend(displayMatrixVector(A, 'A'))
            output_lines.extend(displayMatrixVector(b, 'b'))
    except (NotImplementedError, AttributeError):
        output_lines.append('Inequality constraints (A*x <= b): (not computed)')
    output_lines.append("")

    # Display equality constraints
    try:
        Ae, be = P.Ae, P.be
        if Ae is None or Ae.size == 0:
            output_lines.append('Equality constraints (Ae*x = be): (none)')
        else:
            output_lines.append('Equality constraints (Ae*x = be):')
            output_lines.extend(displayMatrixVector(Ae, 'Ae'))
            output_lines.extend(displayMatrixVector(be, 'be'))
    except (NotImplementedError, AttributeError):
        output_lines.append('Equality constraints (Ae*x = be): (not computed)')
    output_lines.append("")
    
    return "\n".join(output_lines) 