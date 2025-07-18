"""
interval - conversion to interval objects

Syntax:
   I = interval(O)

Inputs:
   O - emptySet object

Outputs:
   I - interval object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: interval/empty

Authors:       Tobias Ladner
Written:       14-October-2024
Last update:   ---
Last revision: ---
Automatic python translation: Florian Nüssel BA 2025
"""

def interval(self, *args):
    """
    Conversion to interval objects
    
    Args:
        *args: optional arguments (not used)
        
    Returns:
        I: interval object representing empty set
    """
    # import classes that could import the class of this method
    from cora_python.contSet.interval import Interval
    
    return Interval.empty(self.dim()) 