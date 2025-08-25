"""
initWeights - initialize the weight of the neural network

Syntax:
    nn.initWeights('glorot')

Inputs:
    method - {'glorot' (default), 'shi'}
    seed - 'default' or nonnegative integer
    idxLayer - indices of layers that should be initialized

Outputs:
    - 

References:
    [1] Glorot, X., Bengio, Y. "Understanding the difficulty of training 
       deep feedforward neural networks". PLMR, 2010.
    [2] Z. Shi, Y. Wang, H. Zhang, J. Yi, and C. Hsieh, "Fast certified 
       robust training with short warmup". NeurIPS, 2021.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork/train

Authors:       Lukas Koller
Written:       12-February-2024
Last update:   ---
Last revision: ---
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, List, Optional, Union
from .neuralNetwork import NeuralNetwork


def initWeights(nn: NeuralNetwork, method: str = 'glorot', seed: Union[str, int] = 'default', 
               idxLayer: Optional[List[int]] = None) -> None:
    """
    Initialize the weight of the neural network
    
    Args:
        nn: NeuralNetwork object
        method: initialization method {'glorot' (default), 'shi'}
        seed: 'default' or nonnegative integer
        idxLayer: indices of layers that should be initialized (defaults to all layers)
    """
    # validate parameters
    if idxLayer is None:
        # 1-based indexing like MATLAB
        idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    # Set random seed for reproducibility
    if seed != 'default':
        np.random.seed(seed)
    
    for i in idxLayer:
        layeri = nn.layers[i]  # Convert to 0-based indexing
        
        # Check if layer is a linear or convolutional layer
        if (hasattr(layeri, '__class__') and 
            ('nnLinearLayer' in str(layeri.__class__) or 'nnConv2dLayer' in str(layeri.__class__))):
            
            nin, nout = layeri.getNumNeurons()
            
            if method == 'glorot':
                # uniform between [-a,a] where a = 1/sqrt(nin)
                a = 1 / np.sqrt(nin)
                layeri.W = np.random.uniform(-a, a, (nout, nin))
                # init bias with 0
                layeri.b = np.zeros((nout, 1))
                
            elif method == 'shi':
                # normal distributed with mu = 0, sigma = sqrt(2*pi)/nin
                sigma = np.sqrt(2 * np.pi) / nin
                layeri.W = np.random.normal(0, sigma, (nout, nin))
                # init bias with 0
                layeri.b = np.zeros((nout, 1))
                
            else:
                raise ValueError(f"Unknown initialization method: {method}")
