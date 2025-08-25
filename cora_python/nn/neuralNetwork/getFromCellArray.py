"""
getFromCellArray - instantiates a neuralNetwork from the given weight,
   bias, and activation function stored in cell arrays
   (legacy function, move to the neuralNetwork/read... function or 
    instantiate layers directly)

Syntax:
    res = neuralNetwork.getFromCellArray(W, b, {actFun1, actFun2, ...})
    res = neuralNetwork.getFromCellArray(W, b, actFun)

Inputs:
    W - cell array holding the weight matrix per linear layer
    b - cell array holding the bias vectors per linear layer
    actFun:
       - cell array holding the actFun string per nonlinear layer
       - actFun string for all nonlinear layer
       (see possible strings in nnActivationLayer/instantiateFromString)

Outputs:
    res - generated network

Example:
  W = {rand(3, 2), rand(3, 3), rand(3, 2)};
  b = {rand(3, 1), rand(3, 1), rand(3, 1)};
  actFun = {'relu', 'relu', 'identity'};
  nn = neuralNetwork.getFromCellArray(W, b, actFun);

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork, nnActivationLayer/instantiateFromString

Authors:       Tobias Ladner
Written:       30-November-2022
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import List, Union
import numpy as np
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def getFromCellArray(W: List[np.ndarray], b: List[np.ndarray], 
                    actFun: Union[str, List[str]]) -> NeuralNetwork:
    """
    Instantiate a neuralNetwork from the given weight, bias, and activation function.
    
    Args:
        W: List holding the weight matrix per linear layer
        b: List holding the bias vectors per linear layer
        actFun: Either a string for all layers or a list of strings per layer
        
    Returns:
        NeuralNetwork object
        
    Raises:
        CORAerror: If inputs are invalid or dimensions don't match
    """
    # pre-process second input possibility
    K = len(W)
    if not isinstance(actFun, list):
        actFunStr = actFun
        actFun = [actFunStr] * K
    
    # validate input
    if not isinstance(W, list):
        raise CORAerror('CORA:wrongInputInConstructor', 'W must be a list')
    if not isinstance(b, list):
        raise CORAerror('CORA:wrongInputInConstructor', 'b must be a list')
    if not isinstance(actFun, list):
        raise CORAerror('CORA:wrongInputInConstructor', 'actFun must be a list or string')
    
    if len(b) != K:
        raise CORAerror('CORA:dimensionMismatch', 'W and b must have the same length')
    if len(actFun) != K:
        raise CORAerror('CORA:dimensionMismatch', 'W and actFun must have the same length')
    
    # instantiate layers
    layers = [None] * (2 * K)
    for k in range(K):
        # Create linear layer
        from .layers.linear.nnLinearLayer import nnLinearLayer
        layers[2 * k] = nnLinearLayer(W[k], b[k])
        
        # Create activation layer
        from .layers.nonlinear.nnActivationLayer import nnActivationLayer
        layers[2 * k + 1] = nnActivationLayer.instantiateFromString(actFun[k])
    
    obj = NeuralNetwork(layers)
    
    return obj
