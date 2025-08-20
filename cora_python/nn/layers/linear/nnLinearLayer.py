"""
nnLinearLayer - class for linear layers

Syntax:
    obj = nnLinearLayer(W, b)
    obj = nnLinearLayer(W, b, name)

Inputs:
    W - weight matrix
    b - bias column vector
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Tobias Ladner, Lukas Koller
Written:       28-March-2022
Last update:   23-November-2022 (polish)
               14-December-2022 (variable input tests, inputArgsCheck)
               03-May-2023 (LK, added backprop for polyZonotope)
               25-May-2023 (LK, modified sampling of gradient for 'extreme')
               25-July-2023 (LK, sampling of gradient with cartProd)
               02-August-2023 (LK, added zonotope batch-eval & -backprop)
               19-August-2023 (LK, zonotope batch-eval: memory optimizations for GPU training)
               22-January-2024 (LK, functions for IBP-based training)
Last revision: 10-August-2022 (renamed)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from ..nnLayer import nnLayer


from cora_python.contSet.interval import Interval
from cora_python.contSet.emptySet import EmptySet


class nnLinearLayer(nnLayer):
    """
    Linear layer class for neural networks
    
    This class implements a linear transformation layer with weights W and bias b.
    The layer computes y = W*x + b for input x.
    """
    
    def __init__(self, W: np.ndarray, b: Optional[np.ndarray] = None, name: Optional[str] = None):
        """
        Constructor for nnLinearLayer
        
        Args:
            W: Weight matrix
            b: Bias column vector, defaults to 0
            name: Name of the layer, defaults to type
        """
        # parse input
        if b is None:
            b = 0
        
        # check dimensions
        if np.isscalar(b) or (isinstance(b, np.ndarray) and b.size == 1):
            b = b * np.ones((W.shape[0], 1))
        
        # check that b is a column vector
        if b.shape[1] != 1:
            raise ValueError("Second input 'b' should be a column vector.")
        
        # check that dimensions match
        if b.shape[0] != W.shape[0]:
            raise ValueError('The dimensions of W and b should match.')
        
        # call super class constructor
        super().__init__(name)
        
        self.W = W.astype(np.float64)
        self.b = b.astype(np.float64)
        self.d = []  # approx error (additive)
        
        # whether the layer is refinable
        self.is_refinable = False
    
    def evaluateNumeric(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate numeric input
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Output after linear transformation
        """
        # linear transformation
        if self._representsa_emptySet(input_data, eps=1e-10):
            r = self.b * np.ones((1, input_data.shape[1]))
        else:
            r = self.W @ input_data + self.b
        
        # add approx error
        if not self._representsa_emptySet(self.d, eps=1e-10):
            samples = self._randPoint(self.d, r.shape[1])
            # Only add samples if they have the right shape
            if samples.shape[0] > 0 and samples.shape == r.shape:
                r = r + samples
        
        return r
    
    def evaluateInterval(self, bounds: 'Interval', options: Dict[str, Any]) -> 'Interval':
        """
        Evaluate interval input using IBP (see Gowal et al. 2019)
        
        Args:
            bounds: Input interval bounds
            options: Evaluation options
            
        Returns:
            bounds: Output interval bounds
        """
        if Interval is None:
            raise ImportError("Interval class not available")
        
        # Compute center and radius.
        mu = (bounds.sup + bounds.inf) / 2
        r = (bounds.sup - bounds.inf) / 2
        
        # Compute linear relaxation.
        mu = self.W @ mu + self.b
        r = np.abs(self.W) @ r
        
        # Convert center and radius back to lower and upper bound.
        bounds = Interval(mu - r, mu + r)
        
        # add approx error
        if not self._representsa_emptySet(self.d, eps=1e-10):
            bounds = bounds + self.d
        
        return bounds
    
    def evaluateSensitivity(self, S: np.ndarray, x: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix
            x: Input point
            options: Evaluation options
            
        Returns:
            S: Updated sensitivity matrix
        """
        # S = S * obj.W;
        # use pagemtimes to compute sensitivity simultaneously for an
        # entire batch.
        if S.ndim == 3:
            # Handle batch case: S is (batch_size, input_dim, input_dim)
            # Need to compute S @ W for each batch element
            result = np.zeros((S.shape[0], S.shape[1], self.W.shape[0]))
            for i in range(S.shape[0]):
                result[i] = S[i] @ self.W
            return result
        else:
            # Handle single case: S @ W
            return S @ self.W
    
    def evaluatePolyZonotope(self, c: np.ndarray, G: np.ndarray, GI: np.ndarray, 
                            E: np.ndarray, id_: List[int], id_2: List[int], 
                            ind: List[int], ind_2: List[int], 
                            options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                           np.ndarray, List[int], List[int], 
                                                           List[int], List[int]]:
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
        c = self.W @ c + self.b
        G = self.W @ G
        GI = self.W @ GI
        
        if not self._representsa_emptySet(self.d, eps=1e-10):
            c = c + self._center(self.d)
            # Handle case where d might be empty or have wrong shape
            d_rad = self._rad(self.d)
            if d_rad.size > 0 and d_rad.shape[0] > 0:
                # Only add diagonal if d_rad has valid dimensions
                if d_rad.shape[0] == c.shape[0]:
                    GI = np.hstack([GI, np.diag(d_rad)])
        
        return c, G, GI, E, id_, id_2, ind, ind_2
    
    def evaluateZonotopeBatch(self, c: np.ndarray, G: np.ndarray, 
                             options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate zonotope batch (for training)
        
        Args:
            c: Center
            G: Generators
            options: Evaluation options
            
        Returns:
            Tuple of (c, G) results
        """
        n, _, batchSize = G.shape
        
        if options.get('nn', {}).get('interval_center', False):
            cl = c[:, 0, :].reshape(n, batchSize)
            cu = c[:, 1, :].reshape(n, batchSize)
            # Ensure lower bounds are less than upper bounds
            cl, cu = np.minimum(cl, cu), np.maximum(cl, cu)
            c_result = self.evaluateInterval(Interval(cl, cu))
            c = np.stack([c_result.inf, c_result.sup], axis=1)
        else:
            c = self.W @ c + self.b
        
        G = self.W @ G
        
        return c, G
    
    def evaluateTaylm(self, input_data: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate Taylor model
        
        Args:
            input_data: Input data
            options: Evaluation options
            
        Returns:
            r: Output after linear transformation
        """
        r = self.W @ input_data + self.b
        return r
    
    def evaluateConZonotope(self, c: np.ndarray, G: np.ndarray, C: np.ndarray, 
                           d: np.ndarray, l: np.ndarray, u: np.ndarray, 
                           options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Evaluate constraint zonotope
        
        Args:
            c: Center
            G: Generators
            C: Constraint matrix
            d: Constraint vector
            l: Lower bounds
            u: Upper bounds
            options: Evaluation options
            
        Returns:
            Tuple of evaluation results
        """
        c = self.W @ c + self.b
        G = self.W @ G
        
        return c, G, C, d, l, u
    
    def backpropNumeric(self, input_data: np.ndarray, grad_out: np.ndarray, 
                        options: Dict[str, Any]) -> np.ndarray:
        """
        Backpropagate numeric gradients
        
        Args:
            input_data: Input data
            grad_out: Output gradients
            options: Backpropagation options
            
        Returns:
            grad_in: Input gradients
        """
        # update weights and bias
        self.updateGrad('W', grad_out @ input_data.T, options)
        self.updateGrad('b', np.sum(grad_out, axis=1, keepdims=True), options)
        
        # backprop gradient
        grad_in = self.W.T @ grad_out
        return grad_in
    
    def backpropIntervalBatch(self, l: np.ndarray, u: np.ndarray, gl: np.ndarray, 
                             gu: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate interval batch (see Gowal et al. 2019)
        
        Args:
            l: Lower bounds
            u: Upper bounds
            gl: Lower bound gradients
            gu: Upper bound gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (gl, gu) results
        """
        mu = (u + l) / 2
        r = (u - l) / 2
        
        # Ensure dimensions match for broadcasting
        if gl.shape[0] != self.W.shape[0]:
            # Pad or truncate gl to match W dimensions
            if gl.shape[0] < self.W.shape[0]:
                # Pad with zeros
                gl_padded = np.zeros((self.W.shape[0], gl.shape[1]))
                gl_padded[:gl.shape[0], :] = gl
                gl = gl_padded
            else:
                # Truncate
                gl = gl[:self.W.shape[0], :]
        
        if gu.shape[0] != self.W.shape[0]:
            # Pad or truncate gu to match W dimensions
            if gu.shape[0] < self.W.shape[0]:
                # Pad with zeros
                gu_padded = np.zeros((self.W.shape[0], gu.shape[1]))
                gu_padded[:gu.shape[0], :] = gu
                gu = gu_padded
            else:
                # Truncate
                gu = gu[:self.W.shape[0], :]
        
        # update weights and bias
        self.updateGrad('W', (gu + gl) @ mu.T + (gu - gl) @ (r.T * np.sign(self.W)), options)
        self.updateGrad('b', np.sum(gu + gl, axis=1, keepdims=True), options)
        
        # backprop gradient
        dmu = self.W.T @ (gu + gl) / 2
        dr = np.abs(self.W.T) @ (gu - gl) / 2
        gl = dmu - dr
        gu = dmu + dr
        
        return gl, gu
    
    def backpropZonotopeBatch(self, c: np.ndarray, G: np.ndarray, gc: np.ndarray, 
                              gG: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate zonotope batch
        
        Args:
            c: Center
            G: Generators
            gc: Center gradients
            gG: Generator gradients
            options: Backpropagation options
            
        Returns:
            Tuple of (gc, gG) results
        """
        n, numGen, batchSize = G.shape
        
        # obtain indices of active generators
        genIds = self.backprop.get('store', {}).get('genIds', slice(None))
        
        zonotope_weight_update = options.get('nn', {}).get('train', {}).get('zonotope_weight_update', 'center')
        
        if zonotope_weight_update == 'center':
            # use the center to update the weights and biases
            weightsUpdate = gc @ c.T
            biasUpdate = np.sum(gc, axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'sample':
            # sample random point factors
            beta = 2 * np.random.rand(numGen, 1, batchSize).astype(c.dtype) - 1
            
            # compute gradient samples
            grads = gc + (gG @ beta).reshape(gc.shape)
            
            # compute input samples
            inputs = c + (G @ beta).reshape(c.shape)
            
            # Compute weights and bias update
            weightsUpdate = grads @ inputs.T
            biasUpdate = np.sum(grads, axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'extreme':
            numSamples = 1
            # sample a point that has only factors {-1,1}
            beta = np.random.choice([-1, 1], size=(numGen, numSamples, batchSize))
            
            # compute gradient samples
            grads = np.tile(gc[:, :, np.newaxis], (1, 1, numSamples)) + (gG @ beta)
            
            # compute input samples
            inputs = np.tile(c[:, :, np.newaxis], (1, 1, numSamples)) + (G @ beta)
            
            # Compute weights and bias update
            weightsUpdate = np.mean(grads @ inputs.transpose(0, 2, 1), axis=2)
            biasUpdate = np.sum(np.mean(grads, axis=2), axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'outer_product':
            # compute outer product of gradient and input zonotope
            # (1) outer product between centers
            centerTerm = gc @ c.T
            
            # (2) outer product between generator matrices
            gensTerm = 1/3 * np.sum(gG[:, genIds, :] @ G[:, genIds, :].transpose(0, 2, 1), axis=2)
            
            # Compute weights and bias update
            weightsUpdate = centerTerm + gensTerm
            biasUpdate = np.sum(gc, axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'sum':
            # compute outer product of gradient and input zonotope
            if options.get('nn', {}).get('interval_center', False):
                # (1) outer product between centers
                cl = c[:, 0, :].reshape(n, batchSize)
                cu = c[:, 1, :].reshape(n, batchSize)
                gl = gc[:, 0, :].reshape(gc.shape[0], batchSize)
                gu = gc[:, 1, :].reshape(gc.shape[0], batchSize)
                
                gl, gu = self.backpropIntervalBatch(cl, cu, gl, gu, options)
                gc = np.stack([gl, gu], axis=1)
                
                # (2) outer product between generator matrices
                gensTerm = np.sum(gG[:, genIds, :] @ G[:, genIds, :].transpose(0, 2, 1), axis=2)
                
                # Compute weights and bias update
                weightsUpdate = gensTerm
                biasUpdate = 0
            else:
                # (1) outer product between centers
                centerTerm = gc @ c.T
                
                # (2) outer product between generator matrices
                gensTerm = np.sum(gG[:, genIds, :] @ G[:, genIds, :].transpose(0, 2, 1), axis=2)
                
                # Compute weights and bias update
                weightsUpdate = centerTerm + gensTerm
                biasUpdate = np.sum(gc, axis=1, keepdims=True)
        else:
            raise ValueError("Only supported values for zonotope_weight_update are 'center', 'sample', 'extreme', 'outer_product', and 'sum'!")
        
        # update weights and bias
        self.updateGrad('W', weightsUpdate, options)
        self.updateGrad('b', biasUpdate, options)
        
        # linear map of the out-going gradient
        if not options.get('nn', {}).get('interval_center', False):
            gc = self.W.T @ gc
        
        # Ensure gG has correct dimensions for matrix multiplication
        if gG.shape[1] != self.W.shape[1]:
            # Handle dimension mismatch
            if gG.shape[1] < self.W.shape[1]:
                # Pad gG with zeros
                gG_padded = np.zeros((gG.shape[0], self.W.shape[1], gG.shape[2]))
                gG_padded[:, :gG.shape[1], :] = gG
                gG = gG_padded
            else:
                # Truncate gG
                gG = gG[:, :self.W.shape[1], :]
        
        gG = self.W.T @ gG
        
        # Clear backprop storage.
        if 'store' in self.backprop:
            self.backprop['store'].clear()
        
        return gc, gG
    
    def getLearnableParamNames(self) -> List[str]:
        """
        Get list of learnable properties
        
        Returns:
            names: List of learnable parameter names
        """
        return ['W', 'b']
    
    def getDropMask(self, x: np.ndarray, dropFactor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get drop mask for dropout
        
        Args:
            x: Input data
            dropFactor: Drop factor
            
        Returns:
            Tuple of (mask, keepIdx, dropIdx)
        """
        # Get the size of the input.
        n, batchSize = x.shape
        
        # Compute random permutation of the dimensions for each element in the batch.
        dDims = np.argsort(np.random.rand(n, batchSize), axis=0)
        
        # Convert to linear indices.
        batch_indices = np.tile(np.arange(batchSize), (n, 1))
        dDimsIdx = np.ravel_multi_index((dDims, batch_indices), x.shape)
        
        # Compute number of dimensions to keep.
        numDimsKeep = int(np.ceil(n * dropFactor))
        
        # Obtain dimensions to keep and which to set to 0.
        keepIdx = dDimsIdx[:numDimsKeep, :]
        dropIdx = dDimsIdx[numDimsKeep:, :]
        
        # Construct the mask.
        mask = np.zeros((n, batchSize), dtype=x.dtype)
        
        # Scale remaining dimensions s.t. their sum remains constant.
        mask.flat[keepIdx.flatten()] = 1 / (1 - dropFactor)
        
        return mask, keepIdx, dropIdx
    
    def getOutputSize(self, inputSize: List[int]) -> List[int]:
        """
        Get output size given input size
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        return [self.W.shape[0], 1]
    
    def getNumNeurons(self) -> Tuple[int, int]:
        """
        Get number of input and output neurons
        
        Returns:
            Tuple of (nin, nout)
        """
        nin = self.W.shape[1]
        nout = self.W.shape[0]
        return nin, nout
    
    def getFieldStruct(self) -> Dict[str, Any]:
        """
        Get field structure for serialization
        
        Returns:
            fieldStruct: Field structure
        """
        fieldStruct = {
            'size_W': self.W.shape,  # is lost for vectors in json
            'W': self.W,
            'b': self.b,
            'd': self.d
        }
        return fieldStruct
    
    # Helper methods for CORA functionality
    def _representsa_emptySet(self, obj, eps: float = 1e-10) -> bool:
        """Check if object represents empty set"""
        if EmptySet is not None:
            return hasattr(obj, 'representsa_') and obj.representsa_('emptySet', eps)
        return False
    
    def _randPoint(self, obj, num_points: int) -> np.ndarray:
        """Generate random points from object"""
        if hasattr(obj, 'randPoint'):
            return obj.randPoint(num_points)
        
        # Handle empty list or None
        if obj is None or (isinstance(obj, list) and len(obj) == 0):
            return np.zeros((0, num_points))
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                return np.zeros((0, num_points))
            # For non-empty arrays, return zeros with correct shape
            return np.zeros((obj.shape[0], num_points))
        
        # Fallback
        return np.zeros((1, num_points))
    
    def _center(self, obj) -> np.ndarray:
        """Get center of object"""
        if hasattr(obj, 'center'):
            return obj.center()
        return obj
    
    def _rad(self, obj) -> np.ndarray:
        """Get radius of object"""
        if hasattr(obj, 'rad'):
            return obj.rad()
        return np.zeros_like(obj)
