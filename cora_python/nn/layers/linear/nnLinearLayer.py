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

Note: Fixed tensor multiplication issues in backpropZonotopeBatch method by replacing
      @ operations with proper 3D tensor handling equivalent to MATLAB's pagemtimes.
"""

import numpy as np
import torch
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
        
        # Convert to torch tensors - all internal operations use torch
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32)
        else:
            W = W.float()
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        else:
            b = b.float()
        
        self.W = W
        self.b = b
        self.d = []  # approx error (additive)
        self.inputSize = []  # matches MATLAB property
        
        # whether the layer is refinable
        self.is_refinable = False
    
    def evaluateNumeric(self, input_data, options: Dict[str, Any]):
        """
        Evaluate numeric input
        
        Args:
            input_data: Input data (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            r: Output after linear transformation (torch tensor)
        """
        # Convert numpy input to torch if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        # Get device and dtype from input
        device = input_data.device
        dtype = input_data.dtype
        
        # Move weights and bias to same device/dtype as input
        W = self.W.to(device=device, dtype=dtype)
        b = self.b.to(device=device, dtype=dtype)
        
        # linear transformation
        if self._representsa_emptySet(input_data, eps=1e-10):
            r = b * torch.ones((1, input_data.shape[1]), dtype=dtype, device=device)
        else:
            r = W @ input_data + b
        
        # add approx error
        if not self._representsa_emptySet(self.d, eps=1e-10):
            samples = self._randPoint(self.d, r.shape[1])
            # Only add samples if they have the right shape
            if samples.shape[0] > 0 and samples.shape == r.shape:
                if isinstance(samples, np.ndarray):
                    samples = torch.tensor(samples, dtype=dtype, device=device)
                else:
                    samples = samples.to(device=device, dtype=dtype)
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
            if isinstance(self.d, list) and len(self.d) == 0:
                # Empty list, no error to add
                pass
            else:
                # Convert to numpy array if needed and add
                d_array = np.array(self.d) if isinstance(self.d, list) else self.d
                if d_array.size > 0:
                    bounds = bounds + d_array
        
        return bounds
    
    def evaluateSensitivity(self, S, x, options: Dict[str, Any]):
        """
        Evaluate sensitivity
        
        Args:
            S: Sensitivity matrix with shape (nK, output_dim, bSz) - receives from next layer (torch tensor)
            x: Input point (unused, kept for interface compatibility)
            options: Evaluation options
            
        Returns:
            S: Updated sensitivity matrix with shape (nK, input_dim, bSz) - passes to previous layer (torch tensor)
        """
        # Convert numpy to torch if needed
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32)
        
        # Get device and dtype from S
        device = S.device
        dtype = S.dtype
        
        # Move W to same device/dtype as S
        W = self.W.to(device=device, dtype=dtype)
        
        # MATLAB: S = pagemtimes(S,obj.W);
        # During backward propagation of sensitivity:
        # S has shape (nK, output_dim, bSz) where output_dim is this layer's output dimension
        # W has shape (output_dim, input_dim) 
        # pagemtimes(S, W) computes S[:,:,b] @ W for each batch b
        # Result: (nK, input_dim, bSz)
        
        if S.ndim == 3:
            # MATLAB: S = pagemtimes(S, obj.W) where:
            # S has shape (nK, output_dim, bSz) - sensitivity from next layer
            # W has shape (output_dim, input_dim)
            # Result: (nK, input_dim, bSz) - sensitivity to previous layer
            # For each batch b: S[:, :, b] @ W = (nK, output_dim) @ (output_dim, input_dim) = (nK, input_dim)
            # Use torch einsum: 'ijk,jl->ilk'
            result = torch.einsum('ijk,jl->ilk', S, W)
            return result
        else:
            # Handle 2D case: S @ W (no transpose needed)
            return S @ W
    
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
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - Input: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - W shape: {self.W.shape}, b shape: {self.b.shape}")
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - c content: {c}")
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - W content: {self.W}")
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - b content: {self.b}")
        
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - About to do: W @ c")
        temp_c = self.W @ c
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - W @ c result: {temp_c}, shape: {temp_c.shape}")
        
        c = temp_c + self.b
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - After adding bias: c = {c}, shape: {c.shape}")
        
        G = self.W @ G
        GI = self.W @ GI
        
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - After multiplication: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        
        if not self._representsa_emptySet(self.d, eps=1e-10) and self.d:
            # Only process if d is not empty and not representing empty set
            d_center = self._center(self.d)
            if hasattr(d_center, 'size') and d_center.size > 0:
                c = c + d_center
            # Handle case where d might be empty or have wrong shape
            d_rad = self._rad(self.d)
            if d_rad.size > 0 and d_rad.shape[0] > 0:
                # Only add diagonal if d_rad has valid dimensions
                if d_rad.shape[0] == c.shape[0]:
                    GI = np.hstack([GI, np.diag(d_rad)])
        
        print(f"DEBUG: nnLinearLayer.evaluatePolyZonotope - Final output: c shape: {c.shape}, G shape: {G.shape}, GI shape: {GI.shape}")
        
        return c, G, GI, E, id_, id_2, ind, ind_2
    
    def evaluateZonotopeBatch(self, c, G, 
                             options: Dict[str, Any]):
        """
        Evaluate zonotope batch (for training)
        
        Args:
            c: Center (numpy array or torch tensor) - converted to torch internally
            G: Generators (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            Tuple of (c, G) results (torch tensors)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        device = c.device
        dtype = c.dtype
        W = self.W.to(device=device, dtype=dtype)
        b = self.b.to(device=device, dtype=dtype)
        n, _, batchSize = G.shape
        
        if options.get('nn', {}).get('interval_center', False):
            cl = c[:, 0, :].reshape(n, batchSize)
            cu = c[:, 1, :].reshape(n, batchSize)
            # Ensure lower bounds are less than upper bounds
            cl, cu = torch.minimum(cl, cu), torch.maximum(cl, cu)
            # Convert to numpy for Interval (Interval class uses numpy)
            cl_np = cl.cpu().numpy()
            cu_np = cu.cpu().numpy()
            c_result = self.evaluateInterval(Interval(cl_np, cu_np), options)
            # Convert back to torch
            c = torch.stack([torch.tensor(c_result.inf, dtype=dtype, device=device), 
                            torch.tensor(c_result.sup, dtype=dtype, device=device)], dim=1)
        else:
            # MATLAB: c = pagemtimes(obj.W, c) + obj.b;
            # Use einsum for page-wise matrix multiplication: W @ c for each batch
            if c.ndim == 3:
                # c is (n_in, 1, batch), W is (n_out, n_in), result should be (n_out, 1, batch)
                # einsum 'ij,jkb->ikb' performs W @ c[:,:,b] for each b
                c = torch.einsum('ij,jkb->ikb', W, c) + b.reshape(b.shape[0], b.shape[1], 1)
            else:
                # c is (n_in, batch), W is (n_out, n_in), result should be (n_out, batch)
                # But we need to return (n_out, 1, batch) to match MATLAB format
                c = W @ c + b
                # Reshape to (n_out, 1, batch) to match expected 3D format
                if c.ndim == 2:
                    # Add middle dimension: (n_out, batch) -> (n_out, 1, batch)
                    c = c.reshape(c.shape[0], 1, c.shape[1])
        
        # MATLAB: G = pagemtimes(obj.W,G); (page-wise matrix multiplication)
        if G.ndim == 3:
            # G is (n_in, q, batch), W is (n_out, n_in), result should be (n_out, q, batch)
            # einsum 'ij,jkb->ikb' performs W @ G[:,:,b] for each b
            G = torch.einsum('ij,jkb->ikb', W, G)
        else:
            # G is (n_in, q), W is (n_out, n_in), result should be (n_out, q)
            # But we need to return (n_out, q, 1) to match expected 3D format
            G = W @ G
            # Reshape to (n_out, q, 1) to match expected 3D format
            if G.ndim == 2:
                # Add batch dimension: (n_out, q) -> (n_out, q, 1)
                G = G.reshape(G.shape[0], G.shape[1], 1)
        
        return c, G
    
    def evaluateTaylm(self, input_data: Any, options: Dict[str, Any]) -> Any:
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
        # Ensure mu and r have correct dimensions for matrix operations
        if mu.shape[0] != self.W.shape[1]:
            # Pad or truncate mu to match W input dimensions
            if mu.shape[0] < self.W.shape[1]:
                mu_padded = np.zeros((self.W.shape[1], mu.shape[1]))
                mu_padded[:mu.shape[0], :] = mu
                mu = mu_padded
            else:
                mu = mu[:self.W.shape[1], :]
        
        if r.shape[0] != self.W.shape[1]:
            # Pad or truncate r to match W input dimensions
            if r.shape[0] < self.W.shape[1]:
                r_padded = np.zeros((self.W.shape[1], r.shape[1]))
                r_padded[:r.shape[0], :] = r
                r = r_padded
            else:
                r = r[:self.W.shape[1], :]
        
        # MATLAB: obj.updateGrad('W', (gu + gl)*mu' + (gu - gl)*r'.*sign(obj.W), options);
        # First compute (gu - gl)*r' (matrix multiplication), then element-wise multiply with sign(W)
        # (gu - gl) is (output_dim, batch_size), r' is (batch_size, input_dim)
        # Result is (output_dim, input_dim)
        grad_W_term1 = (gu + gl) @ mu.T  # (output_dim, input_dim)
        grad_W_term2 = (gu - gl) @ r.T   # (output_dim, input_dim)
        grad_W_term2 = grad_W_term2 * np.sign(self.W)  # Element-wise multiplication
        self.updateGrad('W', grad_W_term1 + grad_W_term2, options)
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
            # MATLAB: grads = gc + reshape(pagemtimes(gG,beta),size(c));
            grads = gc + np.einsum('ijk,lkm->im', gG, beta).reshape(gc.shape)
            
            # compute input samples
            # MATLAB: inputs = inc + reshape(pagemtimes(G,beta),size(c));
            inputs = c + np.einsum('ijk,lkm->im', G, beta).reshape(c.shape)
            
            # Compute weights and bias update
            weightsUpdate = grads @ inputs.T
            biasUpdate = np.sum(grads, axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'extreme':
            numSamples = 1
            # sample a point that has only factors {-1,1}
            beta = np.random.choice([-1, 1], size=(numGen, numSamples, batchSize))
            
            # compute gradient samples
            # MATLAB: grads = permute(repmat(gc,1,1,numSamples),[1 3 2]) + pagemtimes(gG,beta);
            # This creates grads with shape (output_dim, numSamples, batchSize)
            grads = np.tile(gc[:, :, np.newaxis], (1, 1, numSamples)) + np.einsum('ijk,lkm->im', gG, beta).reshape(gc.shape[0], numSamples, gc.shape[1])
            
            # compute input samples
            # MATLAB: inputs = permute(repmat(c,1,1,numSamples),[1 3 2]) + pagemtimes(G,beta);
            # This creates inputs with shape (input_dim, numSamples, batchSize)
            inputs = np.tile(c[:, :, np.newaxis], (1, 1, numSamples)) + np.einsum('ijk,lkm->im', G, beta).reshape(c.shape[0], numSamples, c.shape[1])
            
            # Compute weights and bias update
            # MATLAB: weightsUpdate = squeeze(mean(pagemtimes(grads,'none', inputs,'transpose'),3));
            # This is equivalent to computing grads @ inputs.T for each batch and taking the mean
            # For each batch element, compute grads[:,:,k] @ inputs[:,:,k].T
            weightsUpdate = np.zeros((grads.shape[0], inputs.shape[0]))
            for k in range(batchSize):
                weightsUpdate += grads[:, :, k] @ inputs[:, :, k].T
            weightsUpdate /= batchSize
            biasUpdate = np.sum(np.mean(grads, axis=2), axis=1, keepdims=True)
            
        elif zonotope_weight_update == 'outer_product':
            # compute outer product of gradient and input zonotope
            # (1) outer product between centers
            centerTerm = gc @ c.T
            
            # (2) outer product between generator matrices
            # MATLAB: gensTerm = 1/3*sum(pagemtimes(gG(:,genIds,:),'none', G(:,genIds,:),'transpose'),3);
            # This computes gG @ G.T for each batch element, then sums over the batch
            # For each batch element, compute gG[:,:,k] @ G[:,:,k].T
            gensTerm = np.zeros((gG.shape[0], G.shape[0]))
            for k in range(batchSize):
                gensTerm += gG[:, genIds, k] @ G[:, genIds, k].T
            gensTerm = 1/3 * gensTerm
            
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
                # MATLAB: gensTerm = sum(pagemtimes(gG(:,genIds,:),'none', G(:,genIds,:),'transpose'),3);
                # This computes gG @ G.T for each batch element, then sums over the batch
                # For each batch element, compute gG[:,:,k] @ G[:,:,k].T
                gensTerm = np.zeros((gG.shape[0], G.shape[0]))
                for k in range(batchSize):
                    gensTerm += gG[:, genIds, k] @ G[:, genIds, k].T
                
                # Compute weights and bias update
                weightsUpdate = gensTerm
                biasUpdate = 0
            else:
                # (1) outer product between centers
                centerTerm = gc @ c.T
                
                # (2) outer product between generator matrices
                # MATLAB: gensTerm = sum(pagemtimes(gG(:,genIds,:),'none', G(:,genIds,:),'transpose'),3);
                # This computes gG @ G.T for each batch element, then sums over the batch
                # For each batch element, compute gG[:,:,k] @ G[:,:,k].T
                gensTerm = np.zeros((gG.shape[0], G.shape[0]))
                for k in range(batchSize):
                    gensTerm += gG[:, genIds, k] @ G[:, genIds, k].T
                
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
        
        # MATLAB: gG = pagemtimes(obj.W',gG);
        # This is batch matrix multiplication: W.T @ gG for each batch element
        # W.T is (input_dim, output_dim) = (2, 3)
        # gG is (output_dim, numGen, batchSize) = (3, 2, 4)
        # Result should be (input_dim, numGen, batchSize) = (2, 2, 4)
        
        # Implement pagemtimes equivalent: W.T @ gG for each batch
        # Use einsum for efficient batch matrix multiplication
        # 'ij,jkl->ikl' means: for each batch k, compute W.T @ gG[:,:,k]
        gG = np.einsum('ij,jkl->ikl', self.W.T, gG)
        
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
    
    def computeSizes(self, inputSize: List[int]) -> List[int]:
        """
        Compute and store input size, return output size (matches MATLAB exactly)
        
        Args:
            inputSize: Input dimensions
            
        Returns:
            outputSize: Output dimensions
        """
        self.inputSize = inputSize
        return self.getOutputSize(inputSize)
    
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
            # For non-empty arrays, return the actual values repeated for each point
            return np.tile(obj, (1, num_points))
        
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
