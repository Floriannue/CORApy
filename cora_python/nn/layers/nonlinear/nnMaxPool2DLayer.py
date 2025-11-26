"""
nnMaxPool2DLayer - class for max pooling 2D layers

Syntax:
    obj = nnMaxPool2DLayer(poolSize, stride, name)

Inputs:
    poolSize - size of pooling area, column vector
    stride - step size, column vector

Outputs:
    obj - generated object

References:
    [1] T. Gehr, et al. "AI2: Safety and Robustness Certification of
        Neural Networks with Abstract Interpretation," 2018
    [2] Practical Course SoSe '22 - Report Lukas Koller

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: NeuralNetwork

Authors:       Lukas Koller, Tobias Ladner
Written:       05-June-2022
Last update:   02-December-2022 (better permutation computation)
Last revision: 17-August-2022
               02-December-2022 (clean up)

Translated to Python by: Florian Nüssel
Translation date: 2025-11-25
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..nnLayer import nnLayer


class nnMaxPool2DLayer(nnLayer):
    """
    Max pooling 2D layer
    
    This layer performs max pooling over 2D spatial data.
    """
    
    is_refinable = False
    
    def __init__(self, poolSize, stride=None, name=None):
        """
        Constructor for nnMaxPool2DLayer
        
        Args:
            poolSize: Size of pooling area [height, width]
            stride: Step size [height_stride, width_stride] (default: poolSize)
            name: Layer name (default: None)
        """
        # Call parent constructor
        super().__init__(name)
        
        # Set poolSize
        self.poolSize = np.array(poolSize)
        if self.poolSize.size == 1:
            self.poolSize = np.array([poolSize, poolSize])
        
        # Set stride (default to poolSize)
        if stride is None:
            self.stride = self.poolSize.copy()
        else:
            self.stride = np.array(stride)
            if self.stride.size == 1:
                self.stride = np.array([stride, stride])
    
    def getNumNeurons(self):
        """
        Get number of input and output neurons
        
        Returns:
            nin: Number of input neurons
            nout: Number of output neurons
        """
        if self.inputSize is None or len(self.inputSize) == 0:
            return None, None
        
        # We can only compute the number of neurons if the input size was set
        nin = self.inputSize[0] * self.inputSize[1] * self.inputSize[2]
        outputSize = self.getOutputSize(self.inputSize)
        nout = np.prod(outputSize)
        
        return nin, nout
    
    def getOutputSize(self, imgSize):
        """
        Compute output size
        
        Args:
            imgSize: Input image size [height, width, channels]
            
        Returns:
            outputSize: Output size [height, width, channels]
        """
        in_h = imgSize[0]
        in_w = imgSize[1]
        pool_h = self.poolSize[0]
        pool_w = self.poolSize[1]
        
        # If pool width or height do not divide image width or height,
        # the remaining pixels are ignored
        out_h = int(np.floor((in_h - pool_h) / self.stride[0])) + 1
        out_w = int(np.floor((in_w - pool_w) / self.stride[1])) + 1
        out_c = imgSize[2]  # Preserve number of channels
        
        outputSize = np.array([out_h, out_w, out_c])
        return outputSize
    
    def evaluateNumeric(self, input_data, options=None):
        """
        Evaluate layer numerically
        
        Args:
            input_data: Input data (column vector, all dimensions (h,w,c) flattened)
            options: Evaluation options
            
        Returns:
            r: Output after max pooling
        """
        num_samples = input_data.shape[1] if input_data.ndim > 1 else 1
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        
        # Move adjacent pixels next to each other
        id_mpp = self._aux_computePermutationMatrix()
        
        # Rearrange
        input_rearranged = input_data[id_mpp, :]
        input_rearranged = input_rearranged.reshape(np.prod(self.poolSize), -1, num_samples)
        
        # Compute max
        r = np.max(input_rearranged, axis=0)
        
        # Reshape to match input format
        r = r.reshape(-1, num_samples)
        
        return r
    
    def evaluateSensitivity(self, S, x, options):
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix
            x: Input data
            options: Evaluation options
            
        Returns:
            S: Updated sensitivity matrix
        """
        # This is a simplified implementation
        # Full implementation would require backpropagation storage
        raise NotImplementedError("Sensitivity evaluation for MaxPool2D not yet fully implemented")
    
    def evaluatePolyZonotope(self, c, G, GI, E, id_, id_2, ind, ind_2, options):
        """
        Evaluate polynomial zonotope
        
        Args:
            c: Center
            G: Generators
            GI: Independent generators
            E: Exponent matrix
            id_: ID vector
            id_2: ID vector 2
            ind: Index vector
            ind_2: Index vector 2
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        # This is a complex operation that requires full zonotope support
        raise NotImplementedError("PolyZonotope evaluation for MaxPool2D not yet fully implemented")
    
    # Internal functions
    
    def _aux_computePermutationMatrixForChannel(self):
        """
        Compute permutation matrix to permute an input vector such that
        pooled elements are adjacent. Only computes permutation matrix
        for a single channel.
        
        Returns:
            Wmp: Permutation matrix for single channel
        """
        # Read out properties
        img_h = self.inputSize[0]
        img_w = self.inputSize[1]
        pool_h = self.poolSize[0]
        pool_w = self.poolSize[1]
        
        # Compute number of pooling operations
        num_pools_h = int(np.floor(img_h / pool_h))
        num_pools_w = int(np.floor(img_w / pool_w))
        
        pools_h = np.eye(num_pools_h)
        
        def unitvector(i, n):
            """Create unit vector"""
            return ((np.arange(n) == i) * 1.0).reshape(-1, 1)
        
        pools_hw = []
        # Go along width and construct pooling
        for j in range(pool_w):
            pools_hj = np.kron(pools_h, unitvector(j, pool_w))
            entries_hj = np.kron(pools_hj, np.eye(pool_h))
            entries_hj = np.hstack([
                entries_hj,
                np.zeros((entries_hj.shape[0], img_h - (pool_h * num_pools_h)))
            ])
            pools_hw.append(entries_hj)
        
        # Gather
        pools_hw = np.hstack(pools_hw)
        
        # Compute pooling
        Wmp = np.kron(np.eye(num_pools_w), pools_hw)
        Wmp = np.hstack([Wmp, np.zeros((Wmp.shape[0], img_h * (img_w - (pool_w * num_pools_w))))])
        
        return Wmp
    
    def _aux_computePermutationMatrix(self):
        """
        Compute permutation matrix for entire input vector
        
        Returns:
            id_mpp: Permutation indices
        """
        # Compute permutation matrix for single channel
        Wmp = self._aux_computePermutationMatrixForChannel()
        
        # Construct permutation matrix for entire input
        c_in = self.inputSize[2]
        id_mpp = np.arange(np.prod(self.inputSize))
        id_mpp = id_mpp.reshape(-1, c_in)
        id_mpp = Wmp @ id_mpp
        id_mpp = id_mpp.reshape(-1)
        
        return id_mpp.astype(int)


# ------------------------------ END OF CODE ------------------------------

