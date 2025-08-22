"""
verify - automated verification for specification on neural networks

Description:
    Automated verification for specification on neural networks

Syntax:
    [res, x_, y_] = verify(nn, x, r, A, b, safeSet, options, timeout, verbose)

Inputs:
    nn - Neural network
    x - Center of the initial set
    r - Radius of the initial set
    A - Specification matrix
    b - Specification vector
    safeSet - bool, safe-set or unsafe-set
    options - Evaluation options
    timeout - Timeout value
    verbose - Verbose output

Outputs:
    res - Result string ('VERIFIED', 'COUNTEREXAMPLE', 'UNKNOWN')
    x_ - Counterexample input (if found)
    y_ - Counterexample output (if found)

Example:
    res, x_, y_ = nn.verify(nn, x, 0.1, A, b, True, options, 300, True)

Authors:       Niklas Kochdumper, Tobias Ladner
Written:       23-November-2022 (polish)
Last update:   23-November-2022 (polish)
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import time
from typing import Any, Dict, Optional, Tuple

# Import nnHelper methods for proper integration
from cora_python.nn.nnHelper import validateNNoptions


def verify(self, nn: 'NeuralNetwork', x: np.ndarray, r: float, A: np.ndarray, b: np.ndarray, 
           safeSet: Any, options: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, 
           verbose: bool = False) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Automated verification for specification on neural networks.
    
    Args:
        nn: Neural network
        x: Center of the initial set
        r: Radius of the initial set
        A: Specification matrix
        b: Specification vector
        safeSet: bool, safe-set or unsafe-set
        options: Evaluation options
        timeout: Timeout value
        verbose: Verbose output
        
    Returns:
        Tuple of (res, x_, y_) results
    """
    if options is None:
        options = {}
    if timeout is None:
        timeout = 300.0  # 5 minutes default
    
    # Validate options using nnHelper
    options = validateNNoptions(options, True)
    
    nSplits = 5
    nDims = 1
    
    totalNumSplits = 0
    verifiedPatches = 0
    
    # Extract parameters.
    bs = options.get('nn', {}).get('train', {}).get('mini_batch_size', 32)
    
    # To speed up computations and reduce gpu memory, we only use single precision.
    inputDataClass = np.float32
    
    # Check if a gpu is used during training.
    useGpu = options.get('nn', {}).get('train', {}).get('use_gpu', False)
    if useGpu:
        # Training data is also moved to gpu.
        # For now, we'll use CPU as GPU support requires additional libraries
        inputDataClass = np.float32
    
    # (potentially) move weights of the network to gpu
    nn.castWeights(inputDataClass)
    
    # Specify indices of layers for propagation.
    idxLayer = list(range(len(nn.layers)))
    
    # In each layer, store ids of active generators and identity matrices 
    # for fast adding of approximation errors.
    numGen = nn.prepareForZonoBatchEval(x, options, idxLayer)
    # Allocate generators for initial perturbance set.
    idMat = np.concatenate([np.eye(x.shape[0], dtype=inputDataClass), 
                           np.zeros((x.shape[0], numGen - x.shape[0]), dtype=inputDataClass)], axis=1)
    batchG = np.tile(idMat.reshape(idMat.shape[0], idMat.shape[1], 1), (1, 1, bs))
    
    # Initialize queue.
    xs = x
    rs = r
    # Obtain number of input dimensions.
    n0 = x.shape[0]
    
    res = None
    
    timerVal = time.time()
    
    # Main splitting loop.
    while xs.shape[1] > 0:
        current_time = time.time() - timerVal
        if current_time > timeout:
            res = 'UNKNOWN'
            x_ = None
            y_ = None
            break
        
        if verbose:
            print(f'Queue / Verified / Total: {xs.shape[1]:07d} / {verifiedPatches:07d} / {totalNumSplits:07d} [Avg. radius: {np.mean(rs):.5f}]')
        
        # Pop next batch from the queue.
        xi, ri, xs, rs = self._aux_pop(xs, rs, bs)
        # Move the batch to the GPU.
        xi = xi.astype(inputDataClass)
        ri = ri.astype(inputDataClass)
        
        # Falsification -------------------------------------------------------
        # Try to falsification with a FGSM attack.
        # 1. Compute the sensitivity.
        S, _ = nn.calcSensitivity(xi, options=options, store_sensitivity=False)
        S = np.maximum(S, 1e-3)
        sens = np.sum(np.abs(S), axis=0)
        sens = sens.reshape(-1, 1)
        # 2. Compute adversarial attacks. We want to maximze A*yi + b; 
        # therefore, ...
        zi = xi + ri * np.sign(sens)
        # 3. Check adversarial examples.
        yi = nn.evaluate_(zi, options, idxLayer)
        if safeSet:
            checkSpecs = np.any(A @ yi + b >= 0, axis=0)
        else:
            checkSpecs = np.all(A @ yi + b <= 0, axis=0)
        if np.any(checkSpecs):
            # Found a counterexample.
            res = 'COUNTEREXAMPLE'
            idNzEntry = np.where(checkSpecs)[0]
            id_ = idNzEntry[0]
            x_ = zi[:, id_]
            # Gathering weights from gpu. There is are precision error when 
            # using single gpuArray.
            nn.castWeights(np.float32)
            y_ = nn.evaluate_(x_, options, idxLayer)
            break
        
        # Verification --------------------------------------------------------
        # 1. Use batch-evaluation.
        if not options.get('nn', {}).get('interval_center', False):
            cxi = xi
        else:
            cxi = np.tile(xi.reshape(xi.shape[0], 1, xi.shape[1]), (1, 2, 1))
        Gxi = np.tile(ri.reshape(ri.shape[0], 1, ri.shape[1]), (1, 1, 1)) * batchG[:, :, :ri.shape[1]]
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        # 2. Compute logit-difference.
        if not options.get('nn', {}).get('interval_center', False):
            dyi = A @ yi + b
            dri = np.sum(np.abs(A @ Gyi), axis=0)
        else:
            # Compute the center and the radius of the center-interval.
            yic = 1/2 * (yi[:, 1, :] + yi[:, 0, :])
            yid = 1/2 * (yi[:, 1, :] - yi[:, 0, :])
            # Compute the logit difference.
            dyi = A @ yic + b
            dri = np.sum(np.abs(A @ Gyi), axis=0) + np.sum(np.abs(A * yid.T), axis=1)
        # 3. Check specification.
        if safeSet:
            checkSpecs = np.any(dyi + dri > 0, axis=0)
        else:
            checkSpecs = np.all(dyi - dri < 0, axis=0)
        unknown = checkSpecs
        xi = xi.astype(np.float64)
        ri = ri.astype(np.float64)
        sens = sens.astype(np.float64)
        # 3. Create new splits.
        xis, ris = self._aux_split(xi[:, unknown], ri[:, unknown], sens[:, unknown], nSplits, nDims)
        # Add new splits to the queue.
        xs = np.hstack([xis, xs])
        rs = np.hstack([ris, rs])
        
        totalNumSplits += xis.shape[1]
        verifiedPatches += xi.shape[1] - np.sum(unknown)
    
    # Verified.
    if res is None:
        res = 'VERIFIED'
        x_ = None
        y_ = None
    
    return res, x_, y_

def _aux_pop(self, xs: np.ndarray, rs: np.ndarray, bs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pop elements from the queue"""
    bs = min(bs, xs.shape[1])
    
    # Pop first bs elements from xs.
    idx = list(range(bs))
    
    xi = xs[:, idx]
    xs = xs[:, bs:]
    
    ri = rs[:, idx]
    rs = rs[:, bs:]
    
    return xi, ri, xs, rs

def _aux_split(self, xi: np.ndarray, ri: np.ndarray, sens: np.ndarray, nSplits: int, nDims: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split the input for verification"""
    n, bs = xi.shape
    # Cannot split more than every dimension.
    nDims = min(n, nDims)
    # Split each input in the batch into nSplits parts; use radius*sens 
    # as the splitting heuristic.
    # 1. Find the input dimension with the largest heuristic.
    sortDims = np.argsort(np.abs(sens * ri), axis=0)[::-1]
    dimIds = sortDims[:nDims, :]
    
    splitsIdx = np.tile(np.arange(nSplits), bs)
    bsIdx = np.repeat(np.arange(bs), nSplits)
    
    dim = dimIds[0, :]
    # 2. Split the selected dimension.
    xi_ = xi.copy()
    ri_ = ri.copy()
    # Shift to the lower bound.
    for i in range(bs):
        xi_[dim[i], i] = xi_[dim[i], i] - ri[dim[i], i]
        ri_[dim[i], i] = ri_[dim[i], i] / nSplits
    
    xis = np.tile(xi_.reshape(xi_.shape[0], xi_.shape[1], 1), (1, 1, nSplits))
    ris = np.tile(ri_.reshape(ri_.shape[0], ri_.shape[1], 1), (1, 1, nSplits))
    # Offset the center.
    for i in range(bs):
        for j in range(nSplits):
            xis[dim[i], i, j] = xis[dim[i], i, j] + (2 * j - 1) * ris[dim[i], i, j]
    
    # Flatten.
    xis = xis.reshape(xis.shape[0], -1)
    ris = ris.reshape(ris.shape[0], -1)
    
    return xis, ris
