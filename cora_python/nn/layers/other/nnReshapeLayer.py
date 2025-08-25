"""
nnReshapeLayer - class for reshape layers

Syntax:
    obj = nnReshapeLayer(idx_out)
    obj = nnReshapeLayer(idx_out, name)

Inputs:
    idx_out - output indices
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
from typing import Any, Dict, List, Optional, Tuple
from ..nnLayer import nnLayer


class nnReshapeLayer(nnLayer):
    """
    Reshape layer class for neural networks
    
    This layer reshapes the input according to specified output indices.
    """
    
    def __init__(self, idx_out: List[int], name: Optional[str] = None):
        """
        Constructor for nnReshapeLayer
        
        Args:
            idx_out: Output indices
            name: Name of the layer, defaults to type
        """
        # call super class constructor
        super().__init__(name)
        
        self.idx_out = idx_out
        self.inputSize = None  # Will be set when input size is known
        
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
            outputSize: output size based on idx_out
        """
        self.inputSize = inputSize
        return self.idx_out
    
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer numerically
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            output: reshaped input data
        """
        # Reshape input according to idx_out
        return input_data.reshape(self.idx_out)
    
    def evaluateSensitivity(self, S: np.ndarray, x: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate sensitivity
        
        Args:
            S: sensitivity matrix
            x: input data
            options: evaluation options
            
        Returns:
            S: reshaped sensitivity matrix
        """
        # Reshape sensitivity according to idx_out
        return S.reshape(self.idx_out)
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id: np.ndarray, id_: np.ndarray, 
                            ind: np.ndarray, ind_: np.ndarray, 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate layer with polynomial zonotope
        
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
            c, G, GI, E, id, id_, ind, ind_: reshaped polynomial zonotope
        """
        # Reshape all components according to idx_out
        c = c.reshape(self.idx_out)
        G = G.reshape(self.idx_out + list(G.shape[1:]))
        GI = GI.reshape(self.idx_out + list(GI.shape[1:]))
        
        return c, G, GI, E, id, id_, ind, ind_
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer with Taylor model
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            output: reshaped input data
        """
        return input_data.reshape(self.idx_out)
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           j: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """
        Evaluate layer with constrained zonotope
        
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
            c, G, C, d, l, u: reshaped constrained zonotope
        """
        # Reshape center and generators according to idx_out
        c = c.reshape(self.idx_out)
        G = G.reshape(self.idx_out + list(G.shape[1:]))
        l = l.reshape(self.idx_out)
        u = u.reshape(self.idx_out)
        
        return c, G, C, d, l, u
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            layerStruct: layer structure
        """
        layerStruct = {
            'type': 'nnReshapeLayer',
            'name': self.name,
            'fields': {
                'idx_out': self.idx_out
            }
        }
        return layerStruct
