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
        Get output size for given input size (MATLAB parity)
        - For flatten (idx_out contains -1): return [prod(inputSize), 1]
        - For index-based reshape: return [num_indices, 1]
        """
        self.inputSize = inputSize
        if isinstance(self.idx_out, list) and (-1 in self.idx_out):
            return [int(np.prod(inputSize)), 1]
        idx = np.array(self.idx_out)
        num = int(idx.size)
        return [num, 1]
    
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer numerically
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            output: reshaped input data
        """
        # Debug output
        if hasattr(options, 'debug') and options.get('debug', False):
            print(f"DEBUG ReshapeLayer: input_data shape: {input_data.shape}, idx_out: {self.idx_out}")
        
        # Handle special case of -1 in idx_out (flatten to 1D)
        if -1 in self.idx_out:
            # Calculate the size for the -1 dimension
            total_size = input_data.size
            target_shape = []
            for dim in self.idx_out:
                if dim == -1:
                    # Calculate this dimension based on total size and other dimensions
                    other_dims = [d for d in self.idx_out if d != -1]
                    if other_dims:
                        other_size = np.prod(other_dims)
                        if other_size > 0:
                            target_shape.append(total_size // other_size)
                        else:
                            target_shape.append(1)
                    else:
                        target_shape.append(total_size)
                else:
                    target_shape.append(dim)
            
            # Ensure the reshape is valid
            if np.prod(target_shape) != total_size:
                # If the calculated shape doesn't match, fall back to flattening
                target_shape = [total_size]
            
            return input_data.reshape(target_shape)
        else:
            # Use idx_out as indices to reorder input (like MATLAB)
            # MATLAB: r = input(idx_vec, :, :);
            # Convert MATLAB's column-major idx_out to Python's row-major representation
            # MATLAB idx_out(:) gives column-major order, but we want to maintain C-order in Python
            # So we need to transpose idx_out first, then flatten to get the correct order
            idx_out_transposed = np.array(self.idx_out).T  # Transpose to convert to row-major
            idx_vec = idx_out_transposed.flatten()  # Now flatten in C-order
            # Convert to 0-based indexing for Python
            idx_vec = idx_vec - 1
            

            # Handle multi-dimensional input
            if input_data.ndim > 1:
                # For multi-dimensional input, apply indexing to first dimension
                # and preserve other dimensions
                result = input_data[idx_vec]
            else:
                # For 1D input, just apply indexing
                result = input_data[idx_vec]
            
            return result
    
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
        # Handle special case of -1 in idx_out (flatten to 1D)
        if -1 in self.idx_out:
            # Calculate the size for the -1 dimension
            total_size = S.size
            target_shape = []
            for dim in self.idx_out:
                if dim == -1:
                    # Calculate this dimension based on total size and other dimensions
                    other_dims = [d for d in self.idx_out if d != -1]
                    if other_dims:
                        other_size = np.prod(other_dims)
                        if other_size > 0:
                            target_shape.append(total_size // other_size)
                        else:
                            target_shape.append(1)
                    else:
                        target_shape.append(total_size)
                else:
                    target_shape.append(dim)
            
            # Ensure the reshape is valid
            if np.prod(target_shape) != total_size:
                # If the calculated shape doesn't match, fall back to flattening
                target_shape = [total_size]
            
            return S.reshape(target_shape)
        else:
            # Use idx_out as indices to reorder input (like MATLAB)
            # MATLAB: S = pagetranspose(obj.aux_reshape(pagetranspose(S)));
            idx_vec = np.array(self.idx_out).flatten()
            # Convert to 0-based indexing for Python
            idx_vec = idx_vec - 1
            
            # Handle multi-dimensional input
            if S.ndim > 1:
                # For multi-dimensional input, apply indexing to first dimension
                # and preserve other dimensions
                result = S[idx_vec]
            else:
                # For 1D input, just apply indexing
                result = S[idx_vec]
            
            return result
    
    def evaluateZonotopeBatch(self, c: np.ndarray, G: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate reshape for a batch of zonotopes (centers c and generators G).
        MATLAB behavior: for Flatten, pass through; for index-based reshape, reorder feature rows.
        Shapes: typically c in R^{n x 1 x b} (or n x 1), G in R^{n x q x b} (or n x q)
        """
        # Flatten case (dynamic -1): ACASXU uses this right after input; no feature reordering needed
        if isinstance(self.idx_out, list) and (-1 in self.idx_out):
            return c, G

        # Normalize shapes to 3D for uniform indexing (MATLAB uses n x 1 x b and n x q x b)
        if c.ndim == 2:
            c = c[:, :, np.newaxis]
        if G.ndim == 2:
            G = G[:, :, np.newaxis]

        # Index-based mapping: build 0-based flat index vector
        idx_vec = np.array(self.idx_out)
        # If idx_out is multi-dimensional, flatten in column-major like MATLAB
        if idx_vec.ndim > 1:
            idx_vec = idx_vec.T.reshape(-1, order='C')
        else:
            idx_vec = idx_vec.reshape(-1)
        # Convert to 0-based
        idx0 = idx_vec - 1

        # Bounds check (defensive)
        max_n = c.shape[0]
        idx0 = np.clip(idx0, 0, max_n - 1)

        # Apply to feature axis
        c = c[idx0, :, :]
        G = G[idx0, :, :]

        return c, G

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
