"""
readNetwork - reads and converts a network according to file ending

Syntax:
    res = neuralNetwork.readNetwork(file_path)

Inputs:
    file_path: path to file
    varargin: further input parameter

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: -

Authors:       Tobias Ladner
Written:       30-March-2022
Last update:   30-November-2022 (inputArgsCheck)
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import os
from typing import Any, List, Optional
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def readNetwork(file_path: str, *args) -> NeuralNetwork:
    """
    Reads and converts a network according to file ending
    
    Args:
        file_path: path to file
        *args: further input parameters
        
    Returns:
        obj: generated neuralNetwork object
    """
    # validate input
    if not os.path.isfile(file_path):
        # test if file is already on path
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if spec is None:
                raise CORAerror('CORA:fileNotFound', file_path)
        except:
            raise CORAerror('CORA:fileNotFound', file_path)
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    possible_extensions = ['.onnx', '.nnet', '.yml', '.sherlock', '.json']
    
    if ext not in possible_extensions:
        raise CORAerror('CORA:wrongValue', 'first',
                       f'has to end in {", ".join(possible_extensions)}')
    
    # redirect to specific read function
    if ext == '.onnx':
        obj = NeuralNetwork.readONNXNetwork(file_path, *args)
    elif ext == '.nnet':
        # TODO: Implement readNNetNetwork
        raise CORAerror('CORA:notSupported', 'readNNetNetwork not implemented yet')
    elif ext == '.yml':
        # TODO: Implement readYMLNetwork
        raise CORAerror('CORA:notSupported', 'readYMLNetwork not implemented yet')
    elif ext == '.sherlock':
        # TODO: Implement readSherlockNetwork
        raise CORAerror('CORA:notSupported', 'readSherlockNetwork not implemented yet')
    elif ext == '.json':
        # TODO: Implement readJSONNetwork
        raise CORAerror('CORA:notSupported', 'readJSONNetwork not implemented yet')
    else:
        # This should not happen due to validation above
        raise CORAerror('CORA:wrongValue', 'first',
                       f'has to end in {", ".join(possible_extensions)}')
    
    return obj
