"""
Helper functions for neural network verification refinement logic.
These functions are used by verify.py for the zonotack refinement method.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork

# Try to import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _aux_matchBatchSize(c: np.ndarray, G: np.ndarray, bSz: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicate a zonotope batch for splitting.
    
    MATLAB: function [c,G] = aux_matchBatchSize(c,G,bSz,options)
    """
    from ..layers.linear.nnGeneratorReductionLayer import repelem
    
    if bSz != G.shape[2]:  # iff newSplits > 1
        newSplits = bSz // G.shape[2]
        if options.get('nn', {}).get('interval_center', False):
            # c has shape (n, 2, batch) -> (n, 2, batch*newSplits)
            c = repelem(c, 1, 1, newSplits)
        else:
            # c has shape (n, batch) -> (n, batch*newSplits)
            c = repelem(c, 1, newSplits)
        # G has shape (n, q, batch) -> (n, q, batch*newSplits)
        G = repelem(G, 1, 1, newSplits)
    return c, G


def _aux_scaleAndOffsetZonotope(c: np.ndarray, G: np.ndarray, bc: np.ndarray, br: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale and offset the zonotope to a new hypercube with center bc and radius br.
    
    MATLAB: function [c,G] = aux_scaleAndOffsetZonotope(c,G,bc,br)
    """
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Obtain indices of generator.
    qiIds = min(G.shape[1], bc.shape[0])
    # Extract the relevant entries.
    G_ = G[:, :qiIds, :]  # (n, qiIds, batch)
    bc_ = np.transpose(bc[:qiIds, :], (0, 2, 1))  # (qiIds, 1, batch)
    br_ = np.transpose(br[:qiIds, :], (2, 0, 1))  # (1, qiIds, batch)
    
    # Scale and offset the zonotope to a new hypercube with center bc and radius br.
    offset = pagemtimes(G_, 'none', bc_, 'none')  # (n, 1, batch)
    
    # Offset the center.
    if c.ndim > 2:  # iff options.nn.interval_center
        c = c + offset
    else:
        c = c + offset.squeeze(1)  # (n, batch)
    
    # Scale the generators.
    G[:, :qiIds, :] = G[:, :qiIds, :] * br_  # Broadcast: (n, qiIds, batch) * (1, qiIds, batch)
    
    return c, G


def _aux_computeHeuristic(heuristic: str, layerIdx: int, l: np.ndarray, u: np.ndarray, 
                          dr: np.ndarray, sens: np.ndarray, grad: Any,
                          sim: Optional[np.ndarray] = None, prevNrXs: Optional[np.ndarray] = None,
                          neuronIds: Optional[np.ndarray] = None, onlyUnstable: bool = True,
                          layerDiscount: float = 1.0, imgSz: Optional[Any] = None,
                          patchSz: Optional[Any] = None, patchScore: Optional[Any] = None) -> np.ndarray:
    """
    Compute heuristic for splitting.
    
    MATLAB: function h = aux_computeHeuristic(heuristic,layerIdx,l,u,dr,sens,grad,varargin)
    """
    # Compute the heuristic.
    if heuristic == 'least-unstable':
        # Least unstable neuron (normalize the un-stability).
        minBnd = 1.0 / np.maximum(np.minimum(-l, u), 1e-10)  # Avoid division by zero
        # Compute the heuristic.
        h = minBnd * sens
    elif heuristic == 'least-unstable-gradient':
        # Take the absolute value and add small epsilon to avoid numerical problems.
        if isinstance(grad, (int, float)):
            grad = np.abs(grad) + 1e-3
        else:
            grad = np.abs(grad) + 1e-3
        # Least unstable neuron (normalize the un-stability).
        minBnd = 1.0 / np.maximum(np.minimum(-l, u), 1e-10)
        # Compute the heuristic.
        h = minBnd * grad
    elif heuristic == 'most-sensitive-approx-error':
        # Compute the heuristic.
        h = dr * sens
    elif heuristic == 'most-sensitive-input-radius':
        # Compute the radius.
        r = 0.5 * (u - l)
        # Compute the heuristic.
        h = r * sens
    elif heuristic == 'zono-norm-gradient':
        # Take the absolute value and add small epsilon to avoid numerical problems.
        if isinstance(grad, (int, float)):
            grad = np.abs(grad) + 1e-3
        else:
            grad = np.abs(grad) + 1e-3
        # Compute the radius.
        r = dr  # 1/2*(u - l);
        # Compute the heuristic.
        h = grad * r
    else:
        raise ValueError(f"Invalid heuristic: {heuristic}. Must be one of "
                         f"['least-unstable', 'least-unstable-gradient', "
                         f"'most-sensitive-approx-error', 'most-sensitive-input-radius', "
                         f"'zono-norm-gradient']")
    
    if onlyUnstable:
        # Flag unstable neurons.
        unstable = (l < 0) & (0 < u)
        # Only consider unstable neurons.
        h = np.where(unstable, h, -np.inf)
    
    if layerDiscount != 1.0:
        # Prefer earlier layers.
        h = h * (layerDiscount ** layerIdx)
    
    if prevNrXs is not None and prevNrXs.size > 0:
        # Obtain the batch size.
        bSz = prevNrXs.shape[1] if prevNrXs.ndim > 1 else 1
        
        # We floor all entries. We mark unnecessary splits with decimal numbers.
        prevNrXs_floor = np.floor(prevNrXs)
        # Reduce redundancy by not add constraints for split neurons.
        if h.shape[1] > prevNrXs.shape[1]:
            newSplits = h.shape[1] // prevNrXs.shape[1]
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            prevNrXs_ = repelem(prevNrXs_floor, 1, newSplits)
        else:
            prevNrXs_ = prevNrXs_floor
        
        # Identify already split neurons.
        if neuronIds is not None:
            wasSplit = np.any(np.abs(prevNrXs_) == neuronIds.reshape(-1, 1, 1), axis=0)
            # There is no similarity; just prevent splitting the same neuron twice.
            h = np.where(wasSplit, -np.inf, h)
            if sim is not None and sim.size > 0:
                # Specify a tolerance for similarity.
                tol = 1e-3
                # Reduce the heuristic based on the similarity to already split neurons.
                simSplit = np.any((sim > 1 - tol) & wasSplit.reshape(1, -1, 1), axis=1)
                h = np.where(simSplit, -np.inf, h)
        else:
            # No neuron IDs provided, skip this check
            pass
    
    # TODO: Handle imgSz, patchSz, patchScore if needed (for image inputs)
    
    return h


def _aux_dimSplitConstraints(hi: np.ndarray, nSplits: int, nDims: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct dimension split constraints that splits #nDims dimensions into #nSplits pieces.
    
    MATLAB: function [Ai,bi,dimIds,hi] = aux_dimSplitConstraints(hi,nSplits,nDims)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    # Obtain the number of dimensions and batch size.
    n, bSz = hi.shape
    nDims = min(nDims, n)
    
    # Split each input in the batch into nSplits parts.
    # 1. Find the input dimension with the largest heuristic.
    sortDims = np.argsort(hi, axis=0)[::-1]  # Sort descending
    hi_sorted = np.sort(hi, axis=0)[::-1]  # Sort descending
    dimIds = sortDims[:nDims, :]  # (nDims, batch)
    hi = hi_sorted[:nDims, :]  # (nDims, batch)
    
    # Compute dimension indices.
    dimIdx = sub2ind((nDims, n, bSz),
                     repelem(np.arange(1, nDims + 1), 1, bSz),  # 1-based
                     dimIds.flatten('F'),  # Column-major flatten
                     repelem(np.arange(1, bSz + 1), nDims, 1).flatten('F'))  # 1-based
    
    # 2. Construct the constraints.
    Ai = np.zeros((nDims, n, bSz), dtype=hi.dtype)
    # Set non-zero entries
    dimIdx_0based = dimIdx - 1  # Convert to 0-based
    Ai_flat = Ai.flatten()
    Ai_flat[dimIdx_0based] = 1
    Ai = Ai_flat.reshape(nDims, n, bSz)
    
    # Specify offsets: repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz)
    offsets = -1 + np.arange(1, nSplits) * (2.0 / nSplits)  # (nSplits-1,)
    bi = np.tile(offsets.reshape(-1, 1, 1), (1, nDims, bSz))  # (nSplits-1, nDims, bSz)
    bi = np.transpose(bi, (1, 0, 2))  # (nDims, nSplits-1, bSz)
    
    return Ai, bi, dimIds, hi


def _aux_constructUnsafeOutputSet(options: Dict[str, Any], y: np.ndarray, Gy: np.ndarray,
                                   A: np.ndarray, b: np.ndarray, safeSet: bool,
                                   numUnionConst: int) -> Dict[str, Any]:
    """
    Construct unsafe output set.
    
    MATLAB: function uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet,numUnionConstraint)
    """
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Obtain the number of output dimensions and batch size.
    nK, _, bSz = Gy.shape
    
    if options.get('nn', {}).get('interval_center', False):
        # Compute center and center radius.
        yc = 0.5 * (y[:, 1, :] + y[:, 0, :]).reshape(nK, bSz)
        yr = 0.5 * (y[:, 1, :] - y[:, 0, :])
    else:
        # The radius is zero.
        yc = y
        yr = np.zeros((nK, 1, bSz), dtype=y.dtype)
    
    # Compute the output constraints (logit difference).
    # This matches aux_computeLogitDifference logic
    if options.get('nn', {}).get('interval_center', False):
        yic = yc
        yid = yr
    else:
        yic = yc
        yid = np.zeros((nK, 1, bSz), dtype=y.dtype)
    
    # Compute logit difference
    ld_yi = A @ yic  # (spec_dim, batch)
    ld_Gyi = pagemtimes(A, 'none', Gy, 'none')  # (spec_dim, n_gens, batch)
    
    # Compute logit difference of approximation errors
    # MATLAB: ld_Gyi_err = sum(abs(A.*permute(yid,[2 1 3])),2);
    yid_perm = np.transpose(yid, (1, 0, 2))  # (1, nK, bSz)
    A_yid = A[:, :, np.newaxis] * yid_perm  # (spec_dim, nK, bSz)
    ld_Gyi_err = np.sum(np.abs(A_yid), axis=1)  # (spec_dim, bSz)
    
    # Compute output constraints.
    if safeSet:
        # safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
        A_ = -ld_Gyi  # (spec_dim, n_gens, batch)
        b_ = ld_yi - b  # (spec_dim, batch)
        # Invert the sign for the union constraints.
        if numUnionConst < A_.shape[0]:
            A_[(numUnionConst):, :, :] = -A_[(numUnionConst):, :, :]
            b_[(numUnionConst):, :] = -b_[(numUnionConst):, :]
    else:
        # unsafe iff all(A*y <= b)
        A_ = ld_Gyi  # (spec_dim, n_gens, batch)
        b_ = b - ld_yi  # (spec_dim, batch)
    
    # Construct a struct for the output set.
    uYi = {
        'c': yc,
        'r': yr,
        'G': Gy,
        'A': A_,
        'b': b_ + ld_Gyi_err[:, :, np.newaxis] if ld_Gyi_err.ndim == 2 else b_ + ld_Gyi_err.reshape(-1, 1)
    }
    
    return uYi

