"""
and_ - overloads '&' operator, computes the intersection of a
   full-dimensional space and another set or numerical vector;
   case R^0: can only intersect with R^0, 0 (not representable in
   MATLAB), or the empty set, resulting in R^0, R^0, and the empty set,
   respectively

Syntax:
   S_out = and_(fs,S)
   S_out = and_(fs,S,method)

Inputs:
   fs - fullspace object
   S - contSet object or numerical vector
   method - (optional) approximation method

Outputs:
   S_out - intersection

Example: 
   fs = fullspace(2);
   S = zonotope([1;1],[2 1; -3 1]);
   S_out = fs & S;

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/and

Authors:       Mark Wetzlinger
Written:       22-March-2023
Last update:   05-April-2023 (rename and_)
               28-September-2024 (MW, use precedence)
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

def and_(fs, S, *args):
    """
    Overloads '&' operator, computes the intersection of a
    full-dimensional space and another set or numerical vector;
    case R^0: can only intersect with R^0, 0 (not representable in
    MATLAB), or the empty set, resulting in R^0, R^0, and the empty set,
    respectively
    
    Args:
        fs: fullspace object
        S: contSet object or numerical vector
        *args: additional arguments (unused)
        
    Returns:
        S_out: intersection
    """
    # intersection is always the other set, no need to re-direct to lower
    # precedence
    if hasattr(S, 'copy') and callable(getattr(S, 'copy')):
        S_out = S.copy()
    else:
        S_out = S
    
    return S_out

# ------------------------------ END OF CODE ------------------------------ 