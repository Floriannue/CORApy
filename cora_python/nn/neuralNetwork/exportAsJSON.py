"""
exportAsJSON - exports this network in JSON format

Syntax:
    nn = nn.exportAsJSON()
    nn = nn.exportAsJSON(file_path)

Inputs:
    nn - neuralNetwork
    file_path - (optional) str, file path to store the network

Outputs:
    json - json string

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/importFromJSON

Authors:       Tobias Ladner
Written:       10-November-2023
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import json
from typing import Any, Optional
from .neuralNetwork import NeuralNetwork


def exportAsJSON(nn: NeuralNetwork, file_path: Optional[str] = None) -> str:
    """
    Exports this network in JSON format
    
    Args:
        nn: neuralNetwork object
        file_path: (optional) str, file path to store the network
        
    Returns:
        json: json string
    """
    # convert to struct
    nnStruct = nn.exportAsStruct()
    
    # convert to json
    jsonstr = json.dumps(nnStruct, indent=2)
    
    # save to file if path given
    if file_path is not None:
        try:
            with open(file_path, 'w') as f:
                f.write(jsonstr)
        except Exception as e:
            print(f'Unable to write to file: {file_path}')
            print(f'Error: {e}')
    
    return jsonstr
