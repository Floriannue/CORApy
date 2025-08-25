"""
readSherlockNetwork - reads and converts a network saved in sherlock format

Syntax:
    res = neuralNetwork.readSherlockNetwork(file_path)

Inputs:
    file_path - path to file
    actFun - string of activation function

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/getFromCellArray

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       12-November-2021
Last update:   30-March-2022
                30-November-2022 (removed neuralNetworkOld)
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

from typing import Optional
import numpy as np
from .neuralNetwork import NeuralNetwork


def readSherlockNetwork(file_path: str, actFun: Optional[str] = None) -> NeuralNetwork:
    """
    Read and convert a network saved in sherlock format.
    
    Args:
        file_path: Path to file
        actFun: String of activation function (defaults to 'ReLU')
        
    Returns:
        NeuralNetwork object
    """
    # actFun will be checked in neuralNetwork/getFromCellArray at the end
    if actFun is None:
        # support legacy
        actFun = 'ReLU'
    
    # read text from file
    with open(file_path, 'r') as f:
        text = f.read()
    lines = text.split('\n')
    
    # get network properties
    nrInputs = int(lines[0].strip())
    nrOutputs = int(lines[1].strip())
    hiddenLayers = int(lines[2].strip())
    
    # get number of neurons in each layer
    nrNeurons = [None] * (hiddenLayers + 2)
    nrNeurons[0] = nrInputs
    nrNeurons[-1] = nrOutputs
    
    for i in range(hiddenLayers):
        nrNeurons[i + 1] = int(lines[3 + i].strip())
    
    # initialization
    cnt = 3 + hiddenLayers
    W = [None] * (hiddenLayers + 1)
    b = [None] * (hiddenLayers + 1)
    
    # loop over all layers
    for i in range(len(nrNeurons) - 1):
        
        # initialization
        temp = np.zeros((nrNeurons[i + 1], nrNeurons[i] + 1))
       
        # read data
        for k in range(nrNeurons[i + 1]):
            offset = k * (nrNeurons[i] + 1)
            for j in range(nrNeurons[i] + 1):
                temp[k, j] = float(lines[cnt + offset + j].strip())
        cnt = cnt + (nrNeurons[i] + 1) * nrNeurons[i + 1]
        
        # get weight matrix and bias vector
        W[i] = temp[:, :-1]
        b[i] = temp[:, -1]
    
    # construct neural network
    obj = NeuralNetwork.getFromCellArray(W, b, actFun)
    
    return obj
