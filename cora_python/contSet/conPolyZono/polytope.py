"""
polytope - Encloses a constrained polynomial zonotope by a polytope

Syntax:
    P = polytope(cPZ)
    P = polytope(cPZ,type)
    P = polytope(cPZ,type,method)

Inputs:
    cPZ - conPolyZono object
    type - algorithm used to compute polytope enclosure ('linearize',
           'extend', or 'all')
    method - algorithm used for contraction ('forwardBackward',
             'linearize', 'polynomial', 'interval', 'all', or 'none')

Outputs:
    P - polytope object

Example: 
    A = 1/8 * np.array([[-10, 2, 2, 3, 3]])
    b = -3/8
    EC = np.array([[1, 0, 1, 2, 0], [0, 1, 1, 0, 2], [0, 0, 0, 0, 0]])
    c = np.array([[0], [0]])
    G = np.array([[1, 0, 1, -1/4], [0, 1, -1, 1/4]])
    E = np.array([[1, 0, 2, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    cPZ = conPolyZono(c, G, E, A, b, EC)

    P = polytope(cPZ)

Other m-files required: reduce
Subfunctions: none
MAT-files required: none

See also: polytope, conPolyZono/conZonotope

Authors:       Niklas Kochdumper (MATLAB)
               Python translation by AI Assistant
Written:       06-December-2021 (MATLAB)
Last update:   ---
Python translation: 2025
"""

def polytope(cPZ, *varargin):
    """
    Encloses a constrained polynomial zonotope by a polytope
    
    Args:
        cPZ: conPolyZono object
        *varargin: type and method arguments (optional)
        
    Returns:
        P: polytope object
    """
    # Input arguments are checked in conPolyZono/conZonotope function!
    
    # Convert to constrained zonotope first, then to polytope
    from cora_python.contSet.conZonotope import ConZonotope
    from cora_python.contSet.polytope import Polytope
    
    cZ = ConZonotope(cPZ, *varargin)
    P = Polytope(cZ)
    
    # Set properties
    P._bounded_val = True
    
    return P
