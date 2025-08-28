"""
getNumNeurons - returns the number of input neurons per layer and the
   number of output neurons

Syntax:
    pattern = getNumNeurons(obj)

Inputs:
    obj - object of class neuralNetwork

Outputs:
    numNeurons - array of number of neurons per layer, or nan if unknown

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       26-August-2022
Last update:   ---
Last revision: 02-October-2023
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import List, Union
from .neuralNetwork import NeuralNetwork


def getNumNeurons(obj: NeuralNetwork) -> List[Union[int, float]]:
    """
    Returns the number of input neurons per layer and the number of output neurons
    
    Args:
        obj: NeuralNetwork object
        
    Returns:
        numNeurons: array of number of neurons per layer, or nan if unknown
    """
    numNeurons = np.zeros(len(obj.layers) + 1)
    
    # iterate over all layers
    for i in range(len(obj.layers)):
        if not obj.layers[i].inputSize:
            obj.setInputSize()

        
        if not hasattr(obj.layers[i], 'inputSize') or obj.layers[i].inputSize is None:
            # if still empty, use nan
            numNeurons[i] = np.nan
        else:
            numNeurons[i] = np.prod(obj.layers[i].inputSize)
    
    # output neurons
    if not hasattr(obj, 'neurons_out') or obj.neurons_out is None:
        numNeurons[-1] = np.nan
    else:
        numNeurons[-1] = obj.neurons_out
    
    return numNeurons
