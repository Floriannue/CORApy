"""
polytope - converts a emptySet to a polytope object

Syntax:
   P = polytope(O)

Inputs:
   O - emptySet object

Outputs:
   P - polytope object

Example: 
   O = emptySet(2);
   P = polytope(O);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: polytope/empty

Authors:       Tobias Ladner
Written:       25-February-2025
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

def polytope(self):
    """
    Converts a emptySet to a polytope object
    
    Returns:
        P: polytope object representing empty set
    """
    # import classes that could import the class of this method
    from cora_python.contSet.polytope import Polytope
    
    return Polytope.empty(self.dimension) 