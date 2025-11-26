"""
nnAvgPool2DLayer - class for average pooling 2D layers

Syntax:
    obj = nnAvgPool2DLayer(poolSize, padding, stride, dilation, name)

Inputs:
    poolSize - dimensions of the pooling region
    padding - padding [left, top, right, bottom]
    stride - step size per dimension
    dilation - step size per dimension
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

References:
    [1] T. Gehr, et al. "AI2: Safety and Robustness Certification of
        Neural Networks with Abstract Interpretation," 2018
    [2] Practical Course SoSe '22 - Report Martina Hinz

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nnConv2DLayer

Authors:       Martina Hinz, Tobias Ladner
Written:       17-June-2022
Last update:   01-December-2022 (combine with nnConv2DLayer)
Last revision: 17-August-2022

Translated to Python by: Florian Nüssel
Translation date: 2025-11-25
"""

import numpy as np
from typing import List, Optional
from .nnConv2DLayer import nnConv2DLayer


class nnAvgPool2DLayer(nnConv2DLayer):
    """
    Average pooling 2D layer
    
    This layer performs average pooling with a quadratic pooling region.
    It inherits from nnConv2DLayer and implements pooling as a special
    case of convolution.
    """
    
    def __init__(self, poolSize, padding=None, stride=None, dilation=None, name=None):
        """
        Constructor for nnAvgPool2DLayer
        
        Args:
            poolSize: Dimensions of the pooling region [height, width]
            padding: Padding [left, top, right, bottom] (default: [0, 0, 0, 0])
            stride: Step size per dimension (default: poolSize)
            dilation: Step size per dimension (default: [1, 1])
            name: Name of the layer (default: None)
        """
        # Validate poolSize
        poolSize = np.array(poolSize)
        if poolSize.size == 1:
            poolSize = np.array([poolSize, poolSize])
        
        # Construct dummy filter for average pooling
        # Average pooling is implemented as convolution with uniform weights
        # For nnConv2DLayer, we need a 4D weight matrix: [H, W, in_channels, out_channels]
        # Initially, we create a simple 2D filter and will update it in getOutputSize
        # when we know the number of channels
        W = np.ones(poolSize) / np.prod(poolSize)
        # Reshape to 4D: [H, W, 1, 1] for single channel
        W = W.reshape(poolSize[0], poolSize[1], 1, 1)
        b = 0
        
        # Set default values
        if padding is None:
            padding = np.array([0, 0, 0, 0])
        if stride is None:
            stride = poolSize  # Default stride equals poolSize
        if dilation is None:
            dilation = np.array([1, 1])
        
        # Call parent constructor
        super().__init__(W, b, padding, stride, dilation, name)
        
        # Store poolSize as property
        self.poolSize = poolSize
    
    def getOutputSize(self, imgSize):
        """
        Compute size of output feature map
        
        Args:
            imgSize: Input image size [height, width, channels]
            
        Returns:
            outputSize: Output size [height, width, channels]
        """
        in_c = imgSize[2]
        p_h = self.poolSize[0]
        p_w = self.poolSize[1]
        
        # Update filter weights for multi-channel input
        # Each channel is pooled independently
        # W should be [p_h, p_w, in_c, out_c] where W[:, :, i, i] = 1/(p_h*p_w) for all spatial positions
        # and W[:, :, i, j] = 0 for i != j
        # Initialize with zeros
        self.W = np.zeros((p_h, p_w, in_c, in_c), dtype=np.float64)
        # Set diagonal blocks: for each channel i, set all spatial positions to 1/(p_h*p_w)
        pool_value = 1.0 / (p_h * p_w)
        for i in range(in_c):
            self.W[:, :, i, i] = pool_value
        
        # Update bias to match number of output channels (in_c == out_c for AvgPool)
        # conv2d expects b to have size matching W.shape[3] (number of output channels)
        self.b = np.zeros(in_c, dtype=np.float64)
        
        # Compute output size using parent method
        outputSize = super().getOutputSize(imgSize)
        outputSize[2] = imgSize[2]  # Number of channels remain the same
        
        return outputSize


# ------------------------------ END OF CODE ------------------------------

