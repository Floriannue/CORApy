"""
nnGeneratorReductionLayer - reduces the number of generators of a zonotope

Syntax:
    obj = nnGeneratorReductionLayer(maxGens)
    obj = nnGeneratorReductionLayer(maxGens, name)

Inputs:
    maxGens - maximum order
    name - name of the layer, defaults to type

Outputs:
    obj - generated object

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: neuralNetwork

Authors:       Lukas Koller
Written:       30-April-2024
Last update:   ---
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from .nnIdentityLayer import nnIdentityLayer
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck


def sub2ind(shape: Tuple[int, ...], *subscripts: torch.Tensor) -> torch.Tensor:
    """
    Convert subscripts to linear indices (MATLAB sub2ind equivalent)
    Internal to nn - only uses torch tensors.
    
    Args:
        shape: Shape of the array
        *subscripts: Subscript arrays (one per dimension) - must be torch tensors
        
    Returns:
        indices: Linear indices as torch tensor (0-based for Python indexing)
    """
    if len(subscripts) != len(shape):
        raise ValueError(f"Number of subscript arrays ({len(subscripts)}) must match number of dimensions ({len(shape)})")
    
    # Convert all to torch and ensure they're 1D
    device = subscripts[0].device
    dtype = subscripts[0].dtype
    subscripts = [s.flatten() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=dtype, device=device).flatten() 
                 for s in subscripts]
    
    # Compute linear indices using torch
    # MATLAB uses 1-based indexing, Python uses 0-based
    # But MATLAB sub2ind also uses 1-based, so we need to convert
    indices = subscripts[0] - 1  # Convert to 0-based
    multiplier = 1
    
    for i in range(1, len(shape)):
        multiplier *= shape[i-1]
        indices = indices + (subscripts[i] - 1) * multiplier
    
    return indices.long()


def repelem(arr: torch.Tensor, *repeats: int) -> torch.Tensor:
    """
    Repeat elements of array (MATLAB repelem equivalent)
    Internal to nn - only uses torch tensors.
    
    Args:
        arr: Input array (must be torch tensor)
        *repeats: Number of repeats for each dimension
        
    Returns:
        result: Array with repeated elements as torch tensor
    """
    # Ensure array has at least as many dimensions as repeats
    # MATLAB treats scalars and 1D arrays as 2D when needed
    if arr.ndim < len(repeats):
        # Add singleton dimensions to match number of repeats
        new_shape = arr.shape + (1,) * (len(repeats) - arr.ndim)
        arr = arr.reshape(new_shape)
    
    if len(repeats) == 1:
        # 1D case: repeat each element
        return arr.repeat(repeats[0])
    elif len(repeats) == 2:
        # 2D case: repeat along rows and columns
        result = arr.repeat(repeats[0], dim=0)
        result = result.repeat(repeats[1], dim=1)
        return result
    else:
        # General case
        result = arr
        for i, rep in enumerate(repeats):
            result = result.repeat(rep, dim=i)
        return result


def pagemtimes(A: torch.Tensor, transA: Optional[str], B: torch.Tensor, transB: Optional[str]) -> torch.Tensor:
    """
    Batch matrix multiplication (MATLAB pagemtimes equivalent)
    Internal to nn - only uses torch tensors.
    
    Args:
        A: First array (can be 2D or 3D) - must be torch tensor
        transA: 'transpose' or 'none' for A
        B: Second array (can be 2D or 3D) - must be torch tensor
        transB: 'transpose' or 'none' for B
        
    Returns:
        result: Batch matrix multiplication result as torch tensor
    """
    # Handle transpose
    if transA == 'transpose':
        A = A.transpose(-2, -1)
    if transB == 'transpose':
        B = B.transpose(-2, -1)
    
    # Use einsum for efficient batch matrix multiplication
    # MATLAB pagemtimes performs C(:,:,k) = A(:,:,k) * B(:,:,k) for each page k
    if A.ndim == 2 and B.ndim == 3:
        # A is 2D, B is 3D: A @ B for each batch
        return torch.einsum('ij,jkl->ikl', A, B)
    elif A.ndim == 3 and B.ndim == 2:
        # A is 3D, B is 2D: A @ B for each batch
        return torch.einsum('ijk,jl->ikl', A, B)
    elif A.ndim == 3 and B.ndim == 3:
        # Both 3D: batch matrix multiplication
        # For each batch k: C(:,:,k) = A(:,:,k) @ B(:,:,k)
        # A is (i, j, k), B is (j, l, k), result is (i, l, k)
        # MATLAB pagemtimes automatically broadcasts when batch sizes don't match
        # We need to handle this case explicitly
        A_batch = A.shape[2]
        B_batch = B.shape[2]
        if A_batch == B_batch:
            # Same batch size, use einsum directly
            return torch.einsum('ijk,jlk->ilk', A, B)
        elif A_batch == 1:
            # A has batch size 1, broadcast to B's batch size
            # Squeeze A's batch dimension and use 2D @ 3D case
            A_2d = A[:, :, 0]
            return torch.einsum('ij,jlk->ilk', A_2d, B)
        elif B_batch == 1:
            # B has batch size 1, broadcast to A's batch size
            # Squeeze B's batch dimension and use 3D @ 2D case
            B_2d = B[:, :, 0]
            return torch.einsum('ijk,jl->ilk', A, B_2d)
        else:
            # Different batch sizes, need to replicate the smaller one
            # MATLAB pagemtimes would automatically broadcast/replicate
            # Check if one divides the other evenly
            if A_batch < B_batch and B_batch % A_batch == 0:
                # Replicate A to match B's batch size
                nReps = B_batch // A_batch
                A_rep = repelem(A, 1, 1, nReps)
                return torch.einsum('ijk,jlk->ilk', A_rep, B)
            elif B_batch < A_batch and A_batch % B_batch == 0:
                # Replicate B to match A's batch size
                nReps = A_batch // B_batch
                B_rep = repelem(B, 1, 1, nReps)
                return torch.einsum('ijk,jlk->ilk', A, B_rep)
            else:
                # Batch sizes don't divide evenly, this is an error
                raise ValueError(f"pagemtimes: batch size mismatch {A_batch} vs {B_batch} - sizes don't divide evenly")
    else:
        # Fallback to regular matrix multiplication
        return A @ B


def pagetranspose(A: torch.Tensor) -> torch.Tensor:
    """
    Batch transpose (MATLAB pagetranspose equivalent)
    Internal to nn - only uses torch tensors.
    
    Args:
        A: Input array (2D or 3D) - must be torch tensor
        
    Returns:
        result: Transposed array as torch tensor
    """
    if A.ndim == 2:
        return A.T
    elif A.ndim == 3:
        return A.transpose(1, 2)
    else:
        raise ValueError(f"pagetranspose only supports 2D or 3D arrays, got {A.ndim}D")


class nnGeneratorReductionLayer(nnIdentityLayer):
    """
    Generator reduction layer for neural networks
    
    This layer reduces the number of generators of a zonotope.
    """
    
    def __init__(self, maxGens: int, name: Optional[str] = None):
        """
        Constructor for nnGeneratorReductionLayer
        
        Args:
            maxGens: Maximum number of generators
            name: Name of the layer, defaults to type
        """
        # parse input
        inputArgsCheck([
            [maxGens, 'att', ['numeric']],
        ])
        
        # call super class constructor
        super().__init__(name)
        
        self.maxGens = maxGens
        
        # Initialize backprop storage to match other layers
        # (nnIdentityLayer doesn't initialize it, but we need it for backprop)
        if not hasattr(self, 'backprop') or self.backprop is None:
            self.backprop = {'store': {}}
    
    def evaluateZonotopeBatch(self, c, G, options: Dict[str, Any]):
        """
        Evaluate zonotope batch with generator reduction
        
        Args:
            c: Center (n, batchSize) (numpy array or torch tensor) - converted to torch internally
            G: Generators (n, q, batchSize) (numpy array or torch tensor) - converted to torch internally
            options: Evaluation options
            
        Returns:
            c: Center (unchanged, torch tensor)
            G: Reduced generators with approximation errors (torch tensor)
        """
        # Convert numpy inputs to torch if needed
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, dtype=torch.float32)
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        # Reduce the generator matrix.
        G_, I, keepGensIdx, reduceGensIdx = self.aux_reduceGirad(G, self.maxGens)
        n, q, batchSize = G_.shape
        
        # Compute indices for approximation errors in the generator matrix.
        # MATLAB: GdIdx = reshape(sub2ind([n q+n batchSize], ...
        #     repmat(1:n,1,batchSize), ...
        #     repmat(1:n,1,batchSize), ...
        #     repelem(1:batchSize,1,n)),[n batchSize]);
        # Use torch operations internally
        device = G_.device
        i_idx = torch.tile(torch.arange(1, n+1, device=device), (batchSize,))  # repmat(1:n,1,batchSize)
        j_idx = torch.tile(torch.arange(1, n+1, device=device), (batchSize,))  # repmat(1:n,1,batchSize)
        k_idx = torch.repeat_interleave(torch.arange(1, batchSize+1, device=device), n)  # repelem(1:batchSize,1,n)
        # sub2ind expects numpy, so convert temporarily
        i_idx_np = i_idx.cpu().numpy() if isinstance(i_idx, torch.Tensor) else i_idx
        j_idx_np = j_idx.cpu().numpy() if isinstance(j_idx, torch.Tensor) else j_idx
        k_idx_np = k_idx.cpu().numpy() if isinstance(k_idx, torch.Tensor) else k_idx
        GdIdx = sub2ind((n, q+n, batchSize), i_idx_np, j_idx_np, k_idx_np)
        # Convert back to torch
        if isinstance(GdIdx, np.ndarray):
            GdIdx = torch.tensor(GdIdx, dtype=torch.long, device=device)
        GdIdx = GdIdx.reshape(n, batchSize)
        
        # Append generators for the approximation errors.
        device = G_.device
        dtype = G_.dtype
        G = torch.cat([G_, torch.zeros((n, n, batchSize), dtype=dtype, device=device)], dim=1)
        # Add approximation errors to the generators.
        # Direct assignment using indices
        for i in range(n):
            for j in range(batchSize):
                G[i, q + i, j] = I[i, j]
        
        # Store the indices of the reduced generators.
        if options.get('nn', {}).get('train', {}).get('backprop', False):
            # backprop is already initialized in constructor
            self.backprop['store']['keepGensIdx'] = keepGensIdx
            self.backprop['store']['reduceGensIdx'] = reduceGensIdx
            self.backprop['store']['GdIdx'] = GdIdx
        
        return c, G
    
    def backpropZonotopeBatch(self, c: torch.Tensor, G: torch.Tensor, gc: torch.Tensor, 
                              gG: torch.Tensor, options: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backpropagate zonotope batch gradients
        Internal to nn - all inputs are always torch tensors
        
        Args:
            c: Center (torch tensor)
            G: Generators (torch tensor)
            gc: Center gradients (torch tensor)
            gG: Generator gradients (torch tensor)
            options: Backpropagation options
            
        Returns:
            gc: Center gradients (unchanged, torch tensor)
            gG: Backpropagated generator gradients (torch tensor)
        """
        # Internal to nn - all inputs are always torch tensors
        
        # Extract indices of reduced generators.
        keepGensIdx = self.backprop['store']['keepGensIdx']
        reduceGensIdx = self.backprop['store']['reduceGensIdx']
        GdIdx = self.backprop['store']['GdIdx']
        
        n, q_total, batchSize = G.shape
        # Convert reduceGensIdx to torch for indexing
        reduceGensIdx_torch = torch.tensor(reduceGensIdx.flatten(), dtype=torch.long, device=G.device)
        Gred = G[:, reduceGensIdx_torch].reshape(n, -1, batchSize)
        
        device = G.device
        dtype = G.dtype
        gG_ = torch.zeros_like(G)
        # Convert keepGensIdx to torch for indexing
        keepGensIdx_torch = torch.tensor(keepGensIdx.flatten(), dtype=torch.long, device=device)
        keepGens_count = keepGensIdx.shape[0] if keepGensIdx.ndim > 1 else len(keepGensIdx)
        gG_[:, keepGensIdx_torch] = gG[:, :keepGens_count, :].reshape(n, -1)
        # MATLAB: gG_(:,reduceGensIdx) = reshape(sign(Gred).*permute(gG(GdIdx),[1 3 2]),n,[]);
        # Extract gG values at diagonal positions (approximation error positions)
        # The diagonal positions are at column indices q, q+1, ..., q+n-1 (0-based: q-1, q, ..., q+n-2)
        q = q_total - n  # Number of original generators before adding diagonal
        gG_at_GdIdx = torch.zeros((n, batchSize), dtype=dtype, device=device)
        for i in range(n):
            for j in range(batchSize):
                gG_at_GdIdx[i, j] = gG[i, q + i, j]  # q + i is the diagonal position
        gG_[:, reduceGensIdx_torch] = (torch.sign(Gred) * gG_at_GdIdx.unsqueeze(1)).reshape(n, -1)
        gG = gG_
        
        return gc, gG
    
    def aux_reduceGirad(self, G, maxGens: int):
        """
        Reduce generators using Girad method
        
        Args:
            G: Generators (n, q, batchSize) (torch tensor)
            maxGens: Maximum number of generators
            
        Returns:
            G_: Reduced generators (torch tensor)
            I: Approximation errors (torch tensor)
            keepGensIdx: Indices of kept generators (numpy array for indexing)
            reduceGensIdx: Indices of reduced generators (numpy array for indexing)
        """
        # Convert numpy to torch if needed
        if isinstance(G, np.ndarray):
            G = torch.tensor(G, dtype=torch.float32)
        
        # Obtain number of dimensions and generators.
        n, q, batchSize = G.shape
        
        # Compute the length of each generator.
        genLens = torch.sum(torch.abs(G), dim=0).reshape(q, batchSize)
        
        # Sort the generators by their length.
        idx = torch.argsort(genLens, dim=0)  # Sort along first axis (generators)
        # MATLAB: idx = reshape(sub2ind([q batchSize], idx,repmat(1:batchSize,q,1)),size(idx));
        # Use torch operations internally
        device = G.device
        i_idx = idx.flatten() + 1  # Convert to 1-based for sub2ind (torch tensor)
        j_idx = torch.tile(torch.arange(1, batchSize+1, device=device), (q,))  # repmat(1:batchSize,q,1)
        # sub2ind expects torch tensors
        idx_flat = sub2ind((q, batchSize), i_idx, j_idx)
        idx = idx_flat.reshape(idx.shape)
        
        # Identify the generators to keep.
        keepGensIdx = idx[:maxGens - n, :]  # Already torch tensor
        # Flatten for indexing
        keepGensIdx_flat = keepGensIdx.flatten().long()
        G_ = G[:, keepGensIdx_flat].reshape(n, -1, batchSize)
        
        # Identify generators to reduce.
        reduceGensIdx = idx[maxGens - n:, :]  # Already torch tensor
        reduceGensIdx_flat = reduceGensIdx.flatten().long()
        Gred = G[:, reduceGensIdx_flat].reshape(n, -1, batchSize)
        I = torch.sum(torch.abs(Gred), dim=1).reshape(n, batchSize)
        
        return G_, I, keepGensIdx, reduceGensIdx
    
    def aux_reducePCA(self, G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reduce generators using PCA method
        Internal to nn - G is always torch tensor
        
        Args:
            G: Generators (n, q, batchSize) (torch tensor)
            
        Returns:
            G_: Reduced generators (torch tensor)
            U: Transformation matrix (torch tensor)
        """
        # Internal to nn - G is always torch tensor
        
        # Obtain number of dimensions and generators.
        n, _, batchSize = G.shape
        
        # X = pagetranspose(concatenate([G, -G], axis=1))
        X = torch.cat([G, -G], dim=1).permute(0, 2, 1)  # pagetranspose equivalent
        # Co = pagemtimes(X, 'transpose', X, 'none')
        Co = torch.einsum('ijk,ilk->ijl', X, X)  # X^T @ X for each batch
        U = torch.zeros((n, n, batchSize), dtype=G.dtype, device=G.device)
        for i in range(batchSize):
            # Use torch SVD
            U_, _, _ = torch.linalg.svd(Co[:, :, i])
            U[:, :, i] = U_
        
        # r = sum(abs(pagemtimes(U, 'transpose', G, 'none')), axis=1)
        # U^T @ G for each batch
        r = torch.sum(torch.abs(torch.einsum('ijk,jlk->ilk', U, G)), dim=1)
        idmat = torch.eye(n, dtype=G.dtype, device=G.device)
        # G_ = pagemtimes(U, 'none', (r[:, :, np.newaxis] * idmat[np.newaxis, :, :]), 'none')
        # U @ (r * idmat) for each batch
        r_idmat = r.unsqueeze(2) * idmat.unsqueeze(0)  # (n, batch, n) * (1, n, n) -> (n, batch, n)
        G_ = torch.einsum('ijk,ilk->ilk', U, r_idmat.permute(0, 2, 1))  # U @ (r * idmat) for each batch
        
        return G_, U

