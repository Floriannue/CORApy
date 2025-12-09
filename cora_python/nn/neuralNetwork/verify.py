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

Authors:       Niklas Kochdumper, Tobias Ladner, Lukas Koller
Written:       23-November-2021
Last update:   14-June-2024 (LK, rewritten with efficient splitting)
               Automatic python translation: Florian Nüssel BA 2025
"""

import time
from typing import Optional, Tuple, Any, Dict, TYPE_CHECKING
import numpy as np
import torch

# Import CORA Python modules
from ..nnHelper.validateNNoptions import validateNNoptions
from .verify_helpers import _aux_split

if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork


def verify(nn: 'NeuralNetwork', x: np.ndarray, r: np.ndarray, A: np.ndarray, b: np.ndarray, 
           safeSet: Any, options: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, 
           verbose: bool = False, plotDims: Optional[Any] = None, 
           plotSplittingTree: bool = False) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Automated verification for specification on neural networks.
    
    Exact translation of MATLAB verify.m (lines 1-247).
    Uses PyTorch for all computations (assumes torch is always available).
    """
    if options is None:
        options = {}
    if timeout is None:
        timeout = 300.0  # Default timeout
    
    # Validate options
    options = validateNNoptions(options, True)
    
    # Ensure x and r are 2D column vectors like MATLAB
    x = np.asarray(x, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0):
        r = np.full((x.shape[0], 1), float(r), dtype=np.float32)
    elif r.ndim == 1:
        r = r.reshape(-1, 1)
    if x.shape[0] != r.shape[0]:
        raise ValueError(f"x and r must have the same number of rows: x.shape={x.shape}, r.shape={r.shape}")
    
    # Ensure b has correct shape for broadcasting
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 0:
        b = b.reshape(1, 1)
    elif b.ndim == 1:
        b = b.reshape(-1, 1)
    elif b.ndim == 2:
        if b.shape[0] == 1 and b.shape[1] > 1:
            b = b.T
    
    # MATLAB lines 38-39
    nSplits = 5
    nDims = 1
    
    # MATLAB lines 41-42
    totalNumSplits = 0
    verifiedPatches = 0
    
    # MATLAB line 45: Extract parameters
    bs = options.get('nn', {}).get('train', {}).get('mini_batch_size', 32)
    
    # MATLAB lines 47-55: To speed up computations and reduce gpu memory, we only use single precision
    inputDataClass = np.float32  # MATLAB: single(1)
    useGpu = options.get('nn', {}).get('train', {}).get('use_gpu', False)
    device = torch.device('cuda' if (useGpu and torch.cuda.is_available()) else 'cpu')
    
    # MATLAB line 57: (potentially) move weights of the network to gpu
    nn.castWeights('gpu_float32' if device.type == 'cuda' else np.float32)
    
    # MATLAB line 60: Specify indices of layers for propagation
    idxLayer = list(range(len(nn.layers)))  # 0-based for Python
    
    # MATLAB lines 64-68: In each layer, store ids of active generators and identity matrices
    numGen = nn.prepareForZonoBatchEval(x, options, idxLayer)
    # Allocate generators for initial perturbance set
    # Convert to torch for GPU support
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    idMat = torch.cat([
        torch.eye(x.shape[0], dtype=torch.float32, device=device),
        torch.zeros((x.shape[0], numGen - x.shape[0]), dtype=torch.float32, device=device)
    ], dim=1)
    batchG = idMat.unsqueeze(2).repeat(1, 1, bs)
    
    # MATLAB lines 70-72: Initialize queue - use torch internally
    xs = torch.tensor(x, dtype=torch.float32, device=device)
    rs = torch.tensor(r, dtype=torch.float32, device=device)
    # MATLAB line 74: Obtain number of input dimensions
    n0 = x.shape[0]
    
    # Convert A and b to torch for internal computations
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=device)
    
    # MATLAB line 76
    res = None

    # MATLAB line 78
    timerVal = time.time()

    # MATLAB lines 81-178: Main splitting loop
    while xs.shape[1] > 0:
        # MATLAB line 83
        current_time = time.time() - timerVal
        if current_time > timeout:
            res = 'UNKNOWN'
            x_ = None
            y_ = None
            break
        
        # MATLAB lines 91-94
        if verbose:
            rs_mean = torch.mean(rs).cpu().item()
            print(f'Queue / Verified / Total: {xs.shape[1]:07d} / {verifiedPatches:07d} / {totalNumSplits:07d} [Avg. radius: {rs_mean:.5f}]')
        
        # MATLAB lines 96-100: Pop next batch from the queue
        # MATLAB: [xi,ri,xs,rs] = aux_pop(xs,rs,bs);
        bs_actual = min(bs, xs.shape[1])
        idx = torch.arange(bs_actual, device=device)
        xi = xs[:, idx].clone()
        # Use torch indexing for queue management
        remaining_idx = torch.arange(bs_actual, xs.shape[1], device=device)
        xs = xs[:, remaining_idx]
        ri = rs[:, idx].clone()
        rs = rs[:, remaining_idx]
        
        # MATLAB lines 102-131: Falsification
        # MATLAB line 105: Compute the sensitivity
        S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
        
        # MATLAB line 106
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32, device=device)
        S = torch.maximum(S, torch.tensor(1e-3, dtype=torch.float32, device=device))
        
        # MATLAB lines 108-109
        # MATLAB: sens = permute(sum(abs(S)),[2 1 3]);
        # MATLAB: sens = sens(:,:);
        S_abs = torch.abs(S)
        sens_sum = torch.sum(S_abs, dim=0)  # (n0, cbSz)
        sens = sens_sum.permute(1, 0)  # (cbSz, n0)
        
        # MATLAB line 112: Compute adversarial attacks
        # MATLAB: zi = xi + ri.*sign(sens);
        # sens has shape (cbSz, n0), need to transpose for element-wise multiplication
        # xi and ri have shape (n0, cbSz)
        sens_sign = torch.sign(sens)  # (cbSz, n0)
        sens_sign_T = sens_sign.T  # (n0, cbSz)
        zi = xi + ri * sens_sign_T
        
        # MATLAB line 114: Check adversarial examples
        yi = nn.evaluate_(zi, options, idxLayer)
        
        # MATLAB lines 115-119: Use torch for all computations
        if isinstance(yi, torch.Tensor):
            yi_torch = yi
        else:
            yi_torch = torch.tensor(yi, dtype=torch.float32, device=device)
        
        # Ensure yi is 2D: (num_outputs, batch_size)
        if yi_torch.ndim == 1:
            yi_torch = yi_torch.unsqueeze(1)
        elif yi_torch.ndim == 3:
            yi_torch = yi_torch.squeeze(1) if yi_torch.shape[1] == 1 else yi_torch.reshape(yi_torch.shape[0], -1)
        
        # Compute logit difference using torch
        ld_yi = A_torch @ yi_torch + b_torch  # (num_constraints, batch_size)
        
        if safeSet:
            # MATLAB: checkSpecs = any(A*yi + b >= 0,1);
            checkSpecs = torch.any(ld_yi >= 0, dim=0).cpu().numpy()
        else:
            # MATLAB: checkSpecs = all(A*yi + b <= 0,1);
            checkSpecs = torch.all(ld_yi <= 0, dim=0).cpu().numpy()
        
        # MATLAB lines 120-130
        if torch.any(checkSpecs):
            res = 'COUNTEREXAMPLE'
            idNzEntry = torch.where(checkSpecs)[0]
            id_ = idNzEntry[0].item() if isinstance(idNzEntry, torch.Tensor) else idNzEntry[0]
            x_ = zi[:, id_].cpu().numpy().reshape(-1, 1)
            nn.castWeights(np.float32)
            y_ = nn.evaluate_(x_, options, idxLayer)
            break
        
        # MATLAB lines 133-160: Verification
        # MATLAB lines 135-139: Use batch-evaluation
        if not options.get('nn', {}).get('interval_center', False):
            cxi = xi
        else:
            cxi = torch.tile(xi.reshape(xi.shape[0], xi.shape[1], 1), (1, 1, 2))
            cxi = cxi.permute(0, 2, 1)
        
        # MATLAB line 140
        ri_perm = ri.reshape(ri.shape[0], 1, ri.shape[1])
        batchG_subset = batchG[:, :, :ri.shape[1]]
        Gxi = ri_perm * batchG_subset
        
        if cxi.ndim == 2:
            cxi = cxi.reshape(cxi.shape[0], 1, cxi.shape[1])
        
        # MATLAB line 141
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        
        # MATLAB lines 143-154: Compute logit-difference using torch
        if not options.get('nn', {}).get('interval_center', False):
            # MATLAB lines 144-145
            if isinstance(yi, torch.Tensor):
                yi_torch = yi
            else:
                yi_torch = torch.tensor(yi, dtype=torch.float32, device=device)
            if yi_torch.ndim == 3 and yi_torch.shape[1] == 1:
                yi_2d = yi_torch.squeeze(dim=1)
            else:
                yi_2d = yi_torch.reshape(yi_torch.shape[0], -1)
            
            # Use torch for matrix operations
            dyi = A_torch @ yi_2d + b_torch
            # Use torch einsum for pagemtimes equivalent
            # A is (num_constraints, num_outputs), Gyi is (num_outputs, q, batch)
            # Result should be (num_constraints, q, batch)
            ld_Gyi = torch.einsum('ij,jkb->ikb', A_torch, Gyi)
            dri = torch.sum(torch.abs(ld_Gyi), dim=1)  # (num_constraints, batch)
        else:
            # MATLAB lines 148-153
            if isinstance(yi, torch.Tensor):
                yi_torch = yi
            else:
                yi_torch = torch.tensor(yi, dtype=torch.float32, device=device)
            yic = 0.5 * (yi_torch[:, 1, :] + yi_torch[:, 0, :])
            yid = 0.5 * (yi_torch[:, 1, :] - yi_torch[:, 0, :])
            dyi = A_torch @ yic + b_torch
            # Use torch einsum for pagemtimes
            ld_Gyi = torch.einsum('ij,jkb->ikb', A_torch, Gyi)
            # Use torch permute for pagetranspose equivalent
            yid_trans = yid.permute(1, 0)  # (batch, num_outputs) -> (num_outputs, batch)
            # A_yid = A[:, :, np.newaxis] * yid_trans
            A_yid = A_torch.unsqueeze(2) * yid_trans.unsqueeze(0)  # (num_constraints, num_outputs, batch)
            dri = torch.sum(torch.abs(ld_Gyi), dim=1) + torch.sum(torch.abs(A_yid), dim=1)  # (num_constraints, batch)
        
        # MATLAB lines 156-160: Check specification
        if safeSet:
            # MATLAB: checkSpecs = any(dyi(:,:) + dri(:,:) > 0,1);
            checkSpecs = torch.any(dyi + dri > 0, dim=0).cpu().numpy()
        else:
            # MATLAB: checkSpecs = all(dyi(:,:) - dri(:,:) < 0,1);
            checkSpecs = torch.all(dyi - dri < 0, dim=0).cpu().numpy()
        unknown = checkSpecs
        
        # MATLAB lines 162-164: _aux_split now works with torch internally
        # sens has shape (cbSz, n0), need to transpose to (n0, cbSz) for aux_split
        sens_T = sens.T  # (n0, cbSz) - use torch transpose
        
        # MATLAB lines 166-167: Create new splits
        # MATLAB: [xis,ris] = aux_split(xi(:,unknown),ri(:,unknown),sens(:,unknown), nSplits,nDims);
        # Convert unknown to torch tensor for indexing
        if isinstance(unknown, np.ndarray):
            unknown_torch = torch.tensor(unknown, dtype=torch.bool, device=device)
        else:
            unknown_torch = unknown
        xis, ris = _aux_split(xi[:, unknown_torch], ri[:, unknown_torch], sens_T[:, unknown_torch], nSplits, nDims)
        
        # MATLAB lines 169-170: Add new splits to the queue - xis, ris are already torch tensors
        xs = torch.cat([xis, xs], dim=1)
        rs = torch.cat([ris, rs], dim=1)
        
        # MATLAB lines 172-173
        totalNumSplits = totalNumSplits + xis.shape[1]
        verifiedPatches = verifiedPatches + xi.shape[1] - torch.sum(unknown).item()
        
        # MATLAB lines 175-177: To save memory, we clear all variables that are no longer used
        # Note: MATLAB clears 'xGi' but it's never defined, so we skip it
        del xi, ri, yi, Gyi, dyi, dri
    
    # MATLAB lines 181-185: Verified
    if res is None:
        res = 'VERIFIED'
        x_ = None
        y_ = None
    
    # Ensure x_ and y_ are numpy arrays at final boundary (for external interface)
    if x_ is not None and isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    if y_ is not None and isinstance(y_, torch.Tensor):
        y_ = y_.cpu().numpy()
    
    return res, x_, y_
