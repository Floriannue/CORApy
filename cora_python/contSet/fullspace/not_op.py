"""
not - overloads '~' operator to compute the complement of a
   full-dimensional space, resulting in an empty set

Syntax:
   O = ~fs;
   O = not(fs);

Inputs:
   fs - fullspace object

Outputs:
   O - emptySet object

Example: 
   fs = fullspace(2);
   O = ~fs;

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       07-May-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025

----------------------------- BEGIN CODE ------------------------------
"""

def not_op(fs):
    """
    Overloads '~' operator to compute the complement of a
    full-dimensional space, resulting in an empty set
    
    Args:
        fs: fullspace object
        
    Returns:
        O: emptySet object
    """
    # Import here to avoid circular imports
    from cora_python.contSet.emptySet import EmptySet
    
    # complement is an empty set of same dimension
    O = EmptySet(fs.dimension)
    
    return O

# ------------------------------ END OF CODE ------------------------------ 