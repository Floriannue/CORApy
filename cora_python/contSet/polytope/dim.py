"""
dim - dimension of a polytope

Syntax:
    n = dim(P)

Inputs:
    P - polytope object

Outputs:
    n - dimension of the polytope

Example:
    A = [1 0; 0 1; -1 0; 0 -1];
    b = [1; 1; 1; 1];
    P = polytope(A, b);
    n = dim(P)

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 30-September-2006 (MATLAB)
Last update: 22-March-2007 (MATLAB)
Python translation: 2025
"""

def dim(P) -> int:
    """
    Compute the dimension of a polytope
    
    Args:
        P: Polytope object
        
    Returns:
        int: Dimension of the polytope
    """
    
    # Check if polytope is empty
    if P.isemptyobject():
        return 0
    
    # For halfspace representation: A*x <= b
    if P.A is not None and P.A.size > 0:
        return P.A.shape[1]
    
    # For vertex representation: V
    if hasattr(P, 'V') and P.V is not None and P.V.size > 0:
        return P.V.shape[0]
    
    # Default case - empty polytope
    return 0 