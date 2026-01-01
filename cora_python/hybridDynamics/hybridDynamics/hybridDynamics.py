"""
hybridDynamics - abstract superclass for hybrid dynamics

Syntax:
    obj = hybridDynamics()

Inputs:
    -

Outputs:
    HD - generated hybridDynamics object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       19-October-2023
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from abc import ABC


class HybridDynamics(ABC):
    """
    Abstract superclass for hybrid dynamics
    
    This is an abstract base class that serves as the foundation
    for hybrid automata and parallel hybrid automata.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Constructor for hybridDynamics
        
        This is an abstract class, so direct instantiation is not allowed.
        """
        pass

