"""
not - overloads '~' operator to compute the complement of an empty set,
   resulting in a full-dimensional space

Syntax:
   fs = ~O;
   fs = not(O);

Inputs:
   O - emptySet object

Outputs:
   fs - fullspace object

Example: 
   O = emptySet(2);
   fs = ~O;

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Mark Wetzlinger
Written:       07-May-2023
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

def not_op(self):
    """
    Overloads '~' operator to compute the complement of an empty set,
    resulting in a full-dimensional space
    
    Returns:
        fs: fullspace object of same dimension
    """
    # import classes that could import the class of this method
    from cora_python.contSet.fullspace import Fullspace
    
    # complement is a fullspace of same dimension
    return Fullspace(self.dimension) 