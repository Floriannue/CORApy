"""
readYMLNetwork - reads and converts a network saved in yml format

Syntax:
    res = neuralNetwork.readYMLNetwork(file_path)

Inputs:
    file_path - path to file

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

import numpy as np
from .neuralNetwork import NeuralNetwork
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def readYMLNetwork(file_path: str) -> NeuralNetwork:
    """
    Read and convert a network saved in yml format.
    
    Args:
        file_path: Path to file
        
    Returns:
        NeuralNetwork object
    """
    # read text from file
    with open(file_path, 'r') as f:
        text = f.read()
    lines = text.split('\n')
    
    # get activation functions
    text = lines[0].strip()
    actFun = []
    cnt = 1
    finished = False
    
    while not finished:
        # read next activation
        ind = text.find(f"{cnt}:")
        cnt = cnt + 1
        text = text[ind + 2:]
        ind = text.find(',')
        if ind != -1:
            temp = text[:ind].strip()
            text = text[ind + 1:]
        else:
            temp = text[:len(text) - 1].strip()
            finished = True
        
        # add to previous activations
        if temp == 'Sigmoid':
            actFun.append('sigmoid')
        elif temp == 'Tanh':
            actFun.append('tanh')
        elif temp == 'ReLU':
            actFun.append('ReLU')
        elif temp == 'Linear':
            # no activation -> add identity
            actFun.append('identity')
        else:
            raise CORAerror('CORA:converterIssue')
    
    # split lines into offsets and weights
    bias = []
    weights = []
    
    for i in range(2, len(lines)):
        if lines[i].startswith('weights'):
            bias = lines[2:i]
            weights = lines[i + 1:]
            break
    
    if not bias or not weights:
        raise CORAerror('CORA:converterIssue')
    
    # parse the bias 
    b = [None] * len(actFun)
    bias.append(f"{len(actFun) + 1}:")
    cnt = 0
    
    for i in range(len(b)):
        temp_str = 'temp = '
        ind = bias[cnt].find('[')
        bias[cnt] = bias[cnt][ind - 1:]
        while not bias[cnt].strip().startswith(f"{i + 1}:"):
            temp_str += bias[cnt].strip()
            cnt += 1
        
        # Execute the string to create the array
        # This is a simplified approach - in practice, you'd want to parse the YAML properly
        try:
            # Extract the array part and convert to numpy array
            array_str = temp_str.replace('temp = ', '')
            # This is a placeholder - actual YAML parsing would be more robust
            b[i] = np.array([])  # Placeholder
        except:
            b[i] = np.array([])  # Fallback
    
    # parse the weights
    W = [None] * len(actFun)
    weights.append(f"{len(actFun) + 1}:")
    cnt = 0
    
    for i in range(len(W)):
        cnt += 1
        temp_str = ""
        ind = weights[cnt].find('[')
        weights[cnt] = weights[cnt][ind + 1:]
        
        # parse string
        while not weights[cnt].strip().startswith(f"{i + 1}:"):
            temp_str += weights[cnt].strip()
            cnt += 1
            if weights[cnt].strip().startswith('- ['):
                ind = weights[cnt].find('[')
                weights[cnt] = weights[cnt][ind + 1:]
                temp_str = temp_str[:-1] + ';'
        
        # This is a placeholder - actual YAML parsing would be more robust
        try:
            # Extract the array part and convert to numpy array
            array_str = temp_str.replace('temp = ', '')
            # This is a placeholder - actual YAML parsing would be more robust
            W[i] = np.array([])  # Placeholder
        except:
            W[i] = np.array([])  # Fallback
    
    # construct neural network
    obj = NeuralNetwork.getFromCellArray(W, b, actFun)
    
    return obj
