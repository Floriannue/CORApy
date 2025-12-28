"""
dim - returns the dimension of the ambient space of a level set; this
   corresponds to the number of different variables in its equation

Syntax:
    n = dim(ls)

Inputs:
    ls - levelSet object

Outputs:
    n - dimension of the ambient space

Example: 
   syms x y
   eq = x^2 + y^2 - 4;
   ls = levelSet(eq,[x;y],'==');

   dim(ls)

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: zonotope/dim, interval/dim, levelSet

Authors:       Niklas Kochdumper
Written:       10-December-2021 
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

def dim(ls):
    """
    Returns the dimension of the ambient space of a level set
    
    Args:
        ls: levelSet object
        
    Returns:
        n: dimension of the ambient space
    """
    return ls._dim_val if hasattr(ls, '_dim_val') else 0

