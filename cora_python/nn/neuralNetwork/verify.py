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

import time
from typing import Optional, Tuple, Any, Dict, TYPE_CHECKING
import numpy as np

# Import CORA Python modules
from ..nnHelper.validateNNoptions import validateNNoptions

# Try to import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU support will be disabled.")


if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork


def verify(nn: 'NeuralNetwork', x: np.ndarray, r: float, A: np.ndarray, b: np.ndarray, 
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
    # When training fields are enabled, poly_method must be one of ['bounds', 'singh', 'center']
    # because 'regression' is not supported for training
    if 'nn' not in options:
        options['nn'] = {}
    if 'poly_method' not in options['nn']:
        options['nn']['poly_method'] = 'bounds'  # Default to 'bounds' for training
    
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
    if useGpu and TORCH_AVAILABLE:
        # Training data is also moved to gpu.
        # In MATLAB: inputDataClass = gpuArray(inputDataClass)
        # For Python, we'll use PyTorch GPU tensors
        inputDataClass = 'gpu_float32'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, falling back to CPU")
            useGpu = False
            inputDataClass = np.float32
    else:
        # No GPU support available
        useGpu = False
        inputDataClass = np.float32
    
    # (potentially) move weights of the network to gpu
    nn.castWeights(inputDataClass)
    
    # Specify indices of layers for propagation.
    idxLayer = list(range(len(nn.layers)))  # 0-based indexing like Python
    
    # In each layer, store ids of active generators and identity matrices 
    # for fast adding of approximation errors.
    numGen = nn.prepareForZonoBatchEval(x, options, idxLayer)
    # Allocate generators for initial perturbance set.
    idMat = np.concatenate([np.eye(x.shape[0], dtype=inputDataClass), 
                           np.zeros((x.shape[0], numGen - x.shape[0]), dtype=inputDataClass)], axis=1)
    batchG = np.tile(idMat.reshape(idMat.shape[0], idMat.shape[1], 1), (1, 1, bs))
    
    # Initialize queue - preserve original shapes like MATLAB
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
        xi, ri, xs, rs = _aux_pop(xs, rs, bs)
        # Move the batch to the GPU.
        # In MATLAB: xi = cast(xi,'like',inputDataClass); ri = cast(ri,'like',inputDataClass);
        if useGpu and TORCH_AVAILABLE:
            # Convert to PyTorch tensors and move to GPU
            xi = torch.tensor(xi, dtype=torch.float32, device=device)
            ri = torch.tensor(ri, dtype=torch.float32, device=device)
        else:
            # Use CPU arrays
            xi = xi.astype(inputDataClass)
            ri = ri.astype(inputDataClass)
        
        # Falsification -------------------------------------------------------
        # Try to falsification with a FGSM attack.
        # 1. Compute the sensitivity.
        S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
        
        # Handle GPU vs CPU arrays - match MATLAB exactly
        if useGpu and TORCH_AVAILABLE:
            # Convert sensitivity to GPU tensor if needed
            if isinstance(S, np.ndarray):
                S = torch.tensor(S, dtype=torch.float32, device=device)
            S = torch.maximum(S, torch.tensor(1e-3, dtype=torch.float32, device=device))
            # MATLAB: sens = permute(sum(abs(S)),[2 1 3]); sens = sens(:,:);
            # sum(abs(S)) sums over output dimension (dim=0), giving (input_dim, batch_size)
            sens = torch.sum(torch.abs(S), dim=0)  # shape: (input_dim, batch_size)
            # permute([2 1 3]) swaps dims 1 and 2, but since we only have 2 dims, this is identity
            # sens(:,:) reshapes to 2D, which is already 2D
        else:
            # Use NumPy operations - match MATLAB exactly
            S = np.maximum(S, 1e-3)
            # MATLAB: sens = permute(sum(abs(S)),[2 1 3]); sens = sens(:,:);
            # sum(abs(S)) sums over output dimension (axis=0), giving (input_dim, batch_size)
            sens = np.sum(np.abs(S), axis=0)  # shape: (input_dim, batch_size)
            # permute([2 1 3]) swaps dims 1 and 2, but since we only have 2 dims, this is identity
            # sens(:,:) reshapes to 2D, which is already 2D
        
        # 2. Compute adversarial attacks. We want to maximze A*yi + b; 
        # therefore, ...
        if useGpu and TORCH_AVAILABLE:
            # Use PyTorch operations for GPU
            zi = xi + ri * torch.sign(sens)
        else:
            # Use NumPy operations for CPU
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
            # In MATLAB: nn.castWeights(single(1)); y_ = nn.evaluate_(gather(x_),options,idxLayer);
            nn.castWeights(np.float32)
            # In MATLAB: gather(x_) moves data from GPU to CPU
            # For Python: x_ is already on CPU, so no gather needed
            y_ = nn.evaluate_(x_, options, idxLayer)
            break
        
        # Verification --------------------------------------------------------
        # 1. Use batch-evaluation.
        if not options.get('nn', {}).get('interval_center', False):
            cxi = xi
        else:
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                cxi = torch.tile(xi.reshape(xi.shape[0], 1, xi.shape[1]), (1, 2, 1))
            else:
                # Use NumPy operations for CPU
                cxi = np.tile(xi.reshape(xi.shape[0], 1, xi.shape[1]), (1, 2, 1))
        
        # Handle batchG creation for GPU vs CPU
        if useGpu and TORCH_AVAILABLE:
            # Convert batchG to GPU tensor
            batchG_gpu = torch.tensor(batchG, dtype=torch.float32, device=device)
            Gxi = torch.tile(ri.reshape(ri.shape[0], 1, ri.shape[1]), (1, 1, 1)) * batchG_gpu[:, :, :ri.shape[1]]
        else:
            # Use NumPy operations for CPU
            Gxi = np.tile(ri.reshape(ri.shape[0], 1, ri.shape[1]), (1, 1, 1)) * batchG[:, :, :ri.shape[1]]
        
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        
        # 2. Compute logit-difference.
        if not options.get('nn', {}).get('interval_center', False):
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                dyi = torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), yi) + torch.tensor(b, dtype=torch.float32, device=device)
                dri = torch.sum(torch.abs(torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), Gyi)), dim=0)
            else:
                # Use NumPy operations for CPU
                dyi = A @ yi + b
                dri = np.sum(np.abs(A @ Gyi), axis=0)
        else:
            # Compute the center and the radius of the center-interval.
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                yic = 1/2 * (yi[:, 1, :] + yi[:, 0, :])
                yid = 1/2 * (yi[:, 1, :] - yi[:, 0, :])
                # Compute the logit difference.
                dyi = torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), yic) + torch.tensor(b, dtype=torch.float32, device=device)
                dri = torch.sum(torch.abs(torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), Gyi)), dim=0) + torch.sum(torch.abs(torch.tensor(A, dtype=torch.float32, device=device) * yid.T), dim=1)
            else:
                # Use NumPy operations for CPU
                yic = 1/2 * (yi[:, 1, :] + yi[:, 0, :])
                yid = 1/2 * (yi[:, 1, :] - yi[:, 0, :])
                # Compute the logit difference.
                dyi = A @ yic + b
                dri = np.sum(np.abs(A @ Gyi), axis=0) + np.sum(np.abs(A * yid.T), axis=1)
        # 3. Check specification.
        if safeSet:
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                checkSpecs = torch.any(dyi + dri > 0, dim=0)
            else:
                # Use NumPy operations for CPU
                checkSpecs = np.any(dyi + dri > 0, axis=0)
        else:
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                checkSpecs = torch.all(dyi - dri < 0, dim=0)
            else:
                # Use NumPy operations for CPU
                checkSpecs = np.all(dyi - dri < 0, axis=0)
        
        unknown = checkSpecs
        
        # In MATLAB: xi = gather(xi); ri = gather(ri); sens = gather(sens);
        # For Python: gather() moves data from GPU to CPU
        if useGpu and TORCH_AVAILABLE:
            # Move data from GPU to CPU using PyTorch's .cpu() method
            xi = xi.cpu().numpy().astype(np.float64)
            ri = ri.cpu().numpy().astype(np.float64)
            sens = sens.cpu().numpy().astype(np.float64)
        else:
            # Data is already on CPU
            xi = xi.astype(np.float64)
            ri = ri.astype(np.float64)
            sens = sens.astype(np.float64)
        
        # 3. Create new splits.
        xis, ris = _aux_split(xi[:, unknown], ri[:, unknown], sens[:, unknown], nSplits, nDims)
        # Add new splits to the queue.
        xs = np.hstack([xis, xs])
        rs = np.hstack([ris, rs])
        
        totalNumSplits += xis.shape[1]
        verifiedPatches += xi.shape[1] - np.sum(unknown)
        
        # To save memory, we clear all variables that are no longer used.
        # This is equivalent to MATLAB's clear(batchVars{:})
        batchVars = ['xi', 'ri', 'xGi', 'yi', 'Gyi', 'dyi', 'dri']
        # In Python, we rely on garbage collection, but we can explicitly delete
        del xi, ri, yi, Gyi, dyi, dri
    
    # Verified.
    if res is None:
        res = 'VERIFIED'
        x_ = None
        y_ = None
    
    return res, x_, y_


def _aux_pop(xs: np.ndarray, rs: np.ndarray, bs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pop elements from the queue"""
    bs = min(bs, xs.shape[1])
    
    # Pop first bs elements from xs.
    idx = list(range(bs))
    
    xi = xs[:, idx]
    xs = xs[:, bs:]
    
    ri = rs[:, idx]
    rs = rs[:, bs:]
    
    return xi, ri, xs, rs


def _aux_split(xi: np.ndarray, ri: np.ndarray, sens: np.ndarray, nSplits: int, nDims: int) -> Tuple[np.ndarray, np.ndarray]:
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
