"""
getNumMessagePassingSteps - returns the number of message passing steps

Syntax:
    numMPsteps = getNumMessagePassingSteps(obj)

Inputs:
    obj - object of class (graph) neuralNetwork

Outputs:
    numMPsteps - number of message passing steps

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       21-March-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Any


def getNumMessagePassingSteps(obj: Any) -> int:
    """
    Return the number of message passing steps.
    
    Args:
        obj: NeuralNetwork object
        
    Returns:
        Number of message passing steps
    """
    # Count the number of GCN layers
    numMPsteps = sum(1 for layer in obj.layers 
                     if hasattr(layer, '__class__') and 'nnGCNLayer' in str(layer.__class__))
    
    return numMPsteps
