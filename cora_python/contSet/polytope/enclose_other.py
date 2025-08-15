"""
enclose - computes a polytope that encloses a polytope and other set types

This method extends the basic enclose functionality to handle other set types
beyond just polytopes.

Syntax:
    P = enclose(P1, S)
    P = enclose(P1, M, S)

Inputs:
    P1 - polytope object
    S - another set object
    M - transformation matrix

Outputs:
    P - enclosing polytope

Example: 
    P1 = polytope([1 0; -1 0; 0 1; 0 -1], [1; 1; 1; 1])
    S = zonotope([0; 0], [0.5 0; 0 0.5])
    P = enclose(P1, S)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polytope.enclose

Authors:       Python translation by AI Assistant
Written:       2025
"""

def enclose_other(P1, *varargin):
    """
    Computes a polytope that encloses a polytope and other set types
    
    Args:
        P1: polytope object
        *varargin: other arguments (set S or matrix M and set S)
        
    Returns:
        P: enclosing polytope
    """
    from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
    from .polytope import Polytope
    
    if not isinstance(P1, Polytope):
        raise CORAerror('CORA:wrongInput', 'First argument must be a polytope')
    
    if len(varargin) == 1:
        S = varargin[0]
        # Convert S to polytope if possible, then use convex hull
        try:
            P2 = S.polytope()
            return P1.convHull_(P2)
        except:
            raise CORAerror('CORA:notSupported', 'Cannot convert set to polytope for enclosure')
    elif len(varargin) == 2:
        M, S = varargin
        # Apply transformation and then enclose
        try:
            P2 = S.polytope()
            P2_transformed = M @ P2
            return P1.convHull_(P2_transformed)
        except:
            raise CORAerror('CORA:notSupported', 'Cannot convert set to polytope for enclosure')
    else:
        raise CORAerror('CORA:wrongInput', 'Wrong number of input arguments')
