"""
nnElementwiseAffineLayer - class for elementwise affine layers

Syntax:
    obj = nnElementwiseAffineLayer(scale)
    obj = nnElementwiseAffineLayer(scale, offset)
    obj = nnElementwiseAffineLayer(scale, offset, name)

Inputs:
    scale - elementwise scale (scalar or matching dimension)
    offset - elementwise offset (scalar or matching dimension)
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner, Lukas Koller
Written:       30-March-2022
Last update:   14-December-2022 (variable input tests, inputArgsCheck)
                21-March-2024 (batchZonotope for training)
Last revision: 10-August-2022 (renamed)
                Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from ..nnLayer import nnLayer


class nnElementwiseAffineLayer(nnLayer):
    """
    Element-wise affine layer class for neural networks
    
    This class implements an element-wise affine transformation layer with scale and offset.
    The layer computes y = scale * x + offset for input x.
    """
    
    # whether the layer is refinable
    is_refinable = False
    
    def __init__(self, scale: np.ndarray = 1, offset: np.ndarray = 0, name: Optional[str] = None):
        """
        Constructor for nnElementwiseAffineLayer
        
        Args:
            scale: elementwise scale (scalar or matching dimension)
            offset: elementwise offset (scalar or matching dimension)
            name: name of the layer, defaults to type
        """
        # parse input - equivalent to setDefaultValues({1, 0, []}, varargin)
        if scale is None:
            scale = 1
        if offset is None:
            offset = 0
        
        # validate input - equivalent to inputArgsCheck
        if not isinstance(scale, (int, float, np.ndarray)):
            raise ValueError("scale must be numeric")
        if not isinstance(offset, (int, float, np.ndarray)):
            raise ValueError("offset must be numeric")
        
        # check dims - equivalent to MATLAB dimension checks
        if hasattr(scale, 'shape') and len(scale.shape) > 1 and scale.shape[1] > 1:
            raise ValueError("Scale should be a column vector or scalar")
        if hasattr(offset, 'shape') and len(offset.shape) > 1 and offset.shape[1] > 1:
            raise ValueError("Offset should be a column vector or scalar")
        
        if hasattr(scale, 'size') and hasattr(offset, 'size') and scale.size > 1 and offset.size > 1:
            if scale.size != offset.size:
                raise ValueError("The dimensions of scale and offset should match or be scalar values")
        
        # call super class constructor
        super().__init__(name)
        
        # Convert to torch tensors - all internal operations use torch
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        else:
            scale = scale.float()
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset, dtype=torch.float32)
        else:
            offset = offset.float()
        
        self.scale = scale
        self.offset = offset
    
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
    
    def castWeights(self, target_dtype):
        """
        Callback when data type of learnable parameters was changed
        
        Args:
            target_dtype: numpy dtype to cast parameters to
        """
        self.scale = self.scale.astype(target_dtype)
        self.offset = self.offset.astype(target_dtype)
    
    def evaluateNumeric(self, input_data: torch.Tensor, options: Dict[str, Any]) -> torch.Tensor:
        """
        Evaluate layer numerically
        Internal to nn - input_data is always torch tensor
        
        Args:
            input_data: input data (torch tensor)
            options: evaluation options
            
        Returns:
            r: scaled and offset input data (torch tensor)
        """
        # Internal to nn - input_data is always torch tensor
        
        device = input_data.device
        dtype = input_data.dtype
        
        # Move scale and offset to same device/dtype
        scale = self.scale.to(device=device, dtype=dtype)
        offset = self.offset.to(device=device, dtype=dtype)
        
        # Fix broadcasting: ensure scale and offset have same shape as input_data for element-wise operation
        # scale and offset should be (n, 1) to match input_data (n, batch_size)
        if input_data.ndim == 2 and scale.ndim == 1:
            scale = scale.reshape(-1, 1)
            offset = offset.reshape(-1, 1)
        
        r = scale * input_data + offset
        return r
    
    def evaluateSensitivity(self, S, x, options: Dict[str, Any]):
        """
        Evaluate sensitivity
        
        Args:
            S: sensitivity matrix (torch tensor)
            x: input data (torch tensor)
            options: evaluation options
            
        Returns:
            S: updated sensitivity matrix (torch tensor)
        """
        # Internal to nn - S and x are always torch tensors
        
        device = S.device
        dtype = S.dtype
        
        # Move scale to same device/dtype
        scale = self.scale.to(device=device, dtype=dtype)
        
        # MATLAB: S = scale(:)' .* S;
        # scale(:)' creates a row vector that broadcasts along the first dimension of S
        # This should work universally for any S shape and batch size
        
        # Convert scale to column vector first (like MATLAB's scale(:))
        scale_col = scale.reshape(-1, 1) if scale.ndim == 1 else scale
        
        # Then transpose to row vector (like MATLAB's scale(:)')
        # and reshape to broadcast properly with S along first dimension
        if S.ndim == 1:
            # S is 1D: (output_dim,)
            S = scale_col.flatten() * S
        elif S.ndim == 2:
            # S is 2D: (output_dim, input_dim) 
            scale_broadcast = scale_col.reshape(-1, 1)  # (output_dim, 1)
            S = scale_broadcast * S
        elif S.ndim == 3:
            # S is 3D: (output_dim, input_dim, batch_size)
            scale_broadcast = scale_col.reshape(-1, 1, 1)  # (output_dim, 1, 1)
            S = scale_broadcast * S
        else:
            # Higher dimensions - general case
            shape = [-1] + [1] * (S.ndim - 1)
            scale_broadcast = scale_col.reshape(shape)
            S = scale_broadcast * S
            
        return S
    
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
            c, G, GI, E, id, id_, ind, ind_: updated polynomial zonotope
        """
        c = self.scale * c + self.offset
        G = self.scale * G
        GI = self.scale * GI
        return c, G, GI, E, id, id_, ind, ind_
    
    def evaluateInterval(self, bounds: Any, options: Dict[str, Any]) -> Any:
        """
        Evaluate layer with interval
        
        Args:
            bounds: interval bounds
            options: evaluation options
            
        Returns:
            bounds: updated interval bounds
        """
        # TODO: Import Interval class when available
        # l = self.scale * bounds.inf + self.offset
        # u = self.scale * bounds.sup + self.offset
        # bounds = Interval(l, u)
        # For now, return bounds unchanged
        return bounds
    
    def evaluateZonotopeBatch(self, c: torch.Tensor, G: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate layer with zonotope batch (for training)
        Internal to nn - c and G are always torch tensors
        
        Args:
            c: center (torch tensor)
            G: generators (torch tensor)
            options: evaluation options
            
        Returns:
            c, G: updated zonotope batch (torch tensors)
        """
        # Internal to nn - c and G are always torch tensors
        
        device = c.device
        dtype = c.dtype
        scale = self.scale.to(device=device, dtype=dtype)
        offset = self.offset.to(device=device, dtype=dtype)
        
        # MATLAB: scale(:) and offset(:) are column vectors
        scale = scale.flatten()  # Ensure 1D
        offset = offset.flatten()  # Ensure 1D
        
        if options.get('nn', {}).get('interval_center', False):
            # MATLAB: mask = [(scale(:) < 0) ~(scale(:) < 0)];
            # Creates n x 2 logical matrix where first col is negative, second is positive
            # MATLAB: c_ = permute(c,[3 1 2]); c = permute(cat(3,c_(:,mask),c_(:,~mask)),[2 3 1]);
            # c has shape (n, 2, batch) for interval_center
            # c_ becomes (batch, n, 2) after permute
            # The mask reorders features: negative scale features first, then positive
            mask = scale < 0
            c_ = torch.permute(c, (2, 0, 1))  # (batch, n, 2)
            # Reorder features based on mask: negative first, then positive
            neg_indices = torch.where(mask)[0]
            pos_indices = torch.where(~mask)[0]
            reorder_idx = torch.cat([neg_indices, pos_indices])
            c_reordered = c_[:, reorder_idx, :]
            c = torch.permute(c_reordered, (1, 2, 0))  # Back to (n, 2, batch)
        
        # MATLAB: c = scale(:).*c + offset(:);
        # Reshape for broadcasting: scale and offset are (n,), c is (n, 1 or 2, batch)
        scale_reshaped = scale.reshape(-1, 1, 1) if c.ndim == 3 else scale.reshape(-1, 1)
        offset_reshaped = offset.reshape(-1, 1, 1) if c.ndim == 3 else offset.reshape(-1, 1)
        c = scale_reshaped * c + offset_reshaped
        
        if options.get('nn', {}).get('interval_center', False):
            # MATLAB: c = sort(c,2); - sort along dimension 2 (the bounds dimension)
            c, _ = torch.sort(c, dim=1)
        
        # MATLAB: G = scale(:).*G;
        # Scale the generators
        scale_G = scale.reshape(-1, 1, 1) if G.ndim == 3 else scale.reshape(-1, 1)
        G = scale_G * G
        return c, G
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate layer with Taylor model
        
        Args:
            input_data: input data
            options: evaluation options
            
        Returns:
            r: scaled and offset input data
        """
        r = self.scale * input_data + self.offset
        return r
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
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
            options: evaluation options
            
        Returns:
            c, G, C, d, l, u: updated constrained zonotope
        """
        c = self.scale * c + self.offset
        G = self.scale * G
        return c, G, C, d, l, u
    
    def backpropNumeric(self, input_data: np.ndarray, grad_out: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Backpropagation for numeric evaluation
        
        Args:
            input_data: input data
            grad_out: output gradient
            options: evaluation options
            
        Returns:
            grad_in: input gradient
        """
        grad_in = self.scale * grad_out
        return grad_in
    
    def backpropIntervalBatch(self, l: torch.Tensor, u: torch.Tensor, gl: torch.Tensor, gu: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backpropagation for interval batch
        
        Args:
            l: lower bounds
            u: upper bounds
            gl: lower bound gradients
            gu: upper bound gradients
            options: evaluation options
            
        Returns:
            gl, gu: updated gradients
        """
        gl = self.scale * gl
        gu = self.scale * gu
        return gl, gu
    
    def backpropZonotopeBatch(self, c: torch.Tensor, G: torch.Tensor, gc: torch.Tensor, gG: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backpropagation for zonotope batch
        
        Args:
            c: center
            G: generators
            gc: center gradients
            gG: generator gradients
            options: evaluation options
            
        Returns:
            gc, gG: updated gradients
        """
        gc = self.scale * gc
        gG = self.scale * gG
        return gc, gG
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            fieldStruct: field structure
        """
        fieldStruct = {
            'scale': self.scale,
            'offset': self.offset
        }
        return fieldStruct
    
    def _aux_getScaleAndOffset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read out scale and offsets as vectors
        
        Returns:
            scale, offset: scale and offset as vectors (torch tensors)
        """
        scale = self.scale.flatten()
        offset = self.offset.flatten()
        return scale, offset
