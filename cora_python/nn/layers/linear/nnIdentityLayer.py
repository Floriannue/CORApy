"""
nnIdentityLayer - class for identity layer
   This layer is usually not necessary but can sometimes be helpful

Syntax:
    obj = nnIdentityLayer(name)

Inputs:
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: nnLayer

Authors:       Tobias Ladner
Written:       28-March-2022
Last update:   ---
Last revision: 10-August-2022 (renamed)
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from ..nnLayer import nnLayer


class nnIdentityLayer(nnLayer):
    """
    Identity layer class for neural networks
    
    This layer simply passes through the input without any transformation.
    It's usually not necessary but can sometimes be helpful.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Constructor for nnIdentityLayer
        
        Args:
            name: Name of the layer, defaults to type
        """
        # call super class constructor
        super().__init__(name)
        
        # whether the layer is refinable
        self.is_refinable = False
    
    def getNumNeurons(self) -> Tuple[List[int], List[int]]:
        """
        Get number of neurons for this layer
        
        Returns:
            nin: input neuron count (empty for this layer type)
            nout: output neuron count (empty for this layer type)
        """
        return [], []
    
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Get output size for given input size
        
        Args:
            inputSize: input size
            
        Returns:
            outputSize: output size (same as input for this layer)
        """
        return inputSize
    
    def evaluateNumeric(self, input_data: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate layer numerically - return identity
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: input data (torch tensor)
            options: evaluation options
            
        Returns:
            output: same as input (identity transformation, torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        return input_data
    
    def evaluateSensitivity(self, S: torch.Tensor, x: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate sensitivity - return identity
        Internal to nn - S and x are always torch tensors
        
        Args:
            S: sensitivity matrix (torch tensor)
            x: input data (torch tensor)
            options: evaluation options
            
        Returns:
            S: same as input (identity transformation, torch tensor)
        """
        # Internal to nn - S and x are always torch tensors
        return S
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id: np.ndarray, id_: np.ndarray, 
                            ind: np.ndarray, ind_: np.ndarray, 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate layer with polynomial zonotope - return identity
        
        Args:
            c: center
            G: generators
            GI: independent generators
            E: exponents
            id: identifiers
            id_: independent identifiers
            ind: indices
            ind_: independent indices
            options: evaluation options
            
        Returns:
            c, G, GI, E, id, id_, ind, ind_: same as input (identity transformation)
        """
        return c, G, GI, E, id, id_, ind, ind_
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer with Taylor model - return identity
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            output: same as input (identity transformation)
        """
        return input_data
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           j: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """
        Evaluate layer with constrained zonotope - return identity
        
        Args:
            c: center
            G: generators
            C: constraint matrix
            d: constraint vector
            l: lower bounds
            u: upper bounds
            j: index
            options: evaluation options
            
        Returns:
            c, G, C, d, l, u: same as input (identity transformation)
        """
        return c, G, C, d, l, u
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            layerStruct: layer structure
        """
        layerStruct = {
            'type': 'nnIdentityLayer',
            'name': self.name,
            'fields': {}
        }
        return layerStruct
