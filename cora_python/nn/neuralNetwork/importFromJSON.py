"""
importFromJSON - imports a network from json

Syntax:
    nn = neuralNetwork.importFromJSON(jsonstr)

Inputs:
    jsonstr - json string

Outputs:
    nn - neuralNetwork

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/exportAsJSON

Authors:       Tobias Ladner
Written:       10-November-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import json
from typing import Any
from .neuralNetwork import NeuralNetwork


def importFromJSON(jsonstr: str) -> NeuralNetwork:
    """
    Imports a network from json
    
    Args:
        jsonstr: json string
        
    Returns:
        nn: neuralNetwork
    """
    # decode json - equivalent to jsondecode(jsonstr) in MATLAB
    nnStruct = json.loads(jsonstr)
    
    # import network - equivalent to neuralNetwork.importFromStruct(nnStruct) in MATLAB
    nn = NeuralNetwork.importFromStruct(nnStruct)
    
    return nn
