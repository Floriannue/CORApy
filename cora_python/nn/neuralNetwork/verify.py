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
from .verify_helpers import _aux_split, _aux_split_with_dim, _aux_constructInputZonotope, _aux_pop_simple

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
    
    # Convert inputs to torch tensors for GPU support
    # Use numpy for initial validation, then convert to torch immediately
    x_np = np.asarray(x, dtype=np.float32)
    r_np = np.asarray(r, dtype=np.float32)
    A_np = np.asarray(A, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    
    # Validate and reshape inputs
    if x_np.ndim == 1:
        x_np = x_np.reshape(-1, 1)
    if np.isscalar(r_np) or (isinstance(r_np, np.ndarray) and r_np.ndim == 0):
        r_np = np.full((x_np.shape[0], 1), float(r_np), dtype=np.float32)
    elif r_np.ndim == 1:
        r_np = r_np.reshape(-1, 1)
    if x_np.shape[0] != r_np.shape[0]:
        raise ValueError(f"x and r must have the same number of rows: x.shape={x_np.shape}, r.shape={r_np.shape}")
    
    # Ensure b has correct shape for broadcasting
    if b_np.ndim == 0:
        b_np = b_np.reshape(1, 1)
    elif b_np.ndim == 1:
        b_np = b_np.reshape(-1, 1)
    elif b_np.ndim == 2:
        if b_np.shape[0] == 1 and b_np.shape[1] > 1:
            b_np = b_np.T
    
    # Store numpy versions for external interface (return values)
    x = x_np
    r = r_np
    A = A_np
    b = b_np
    
    # MATLAB lines 38-39
    nSplits = 5
    nDims = 1
    
    # MATLAB lines 41-42
    totalNumSplits = 0
    verifiedPatches = 0
    
    # MATLAB line 45: Extract parameters
    # MATLAB: bs = options.nn.train.mini_batch_size; (tests set this to 512)
    bs = options.get('nn', {}).get('train', {}).get('mini_batch_size', 512)
    
    # MATLAB lines 47-55: To speed up computations and reduce gpu memory, we only use single precision
    # Use torch for all internal computations
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
    # Scale initial generators by radii (per batch element) to match MATLAB zonotope construction
    Gs = rs.reshape(rs.shape[0], 1, rs.shape[1]) * batchG[:, :, :rs.shape[1]]
    # MATLAB line 74: Obtain number of input dimensions
    n0 = x.shape[0]
    
    # Convert A and b to torch for internal computations
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=device)
    # Ensure b_torch is a column vector (num_constraints, 1) for proper broadcasting
    if b_torch.ndim == 1:
        b_torch = b_torch.unsqueeze(1)  # (num_constraints,) -> (num_constraints, 1)
    elif b_torch.ndim == 2 and b_torch.shape[1] != 1:
        # If b is (num_constraints, batch_size), take first column
        b_torch = b_torch[:, 0:1]  # (num_constraints, 1)
    
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
        if options.get('nn', {}).get('_debug_verify', False):
            print(f'ITER DEBUG: queue={xs.shape[1]} verified={verifiedPatches} totalSplits={totalNumSplits}')
        
        # MATLAB lines 96-100: Pop next batch from the queue (front)
        bs_actual = min(bs, xs.shape[1])
        idx = torch.arange(bs_actual, device=xs.device)
        xi = xs[:, idx].clone()
        ri = rs[:, idx].clone()
        Gi = Gs[:, :, idx].clone()
        remaining_idx = torch.arange(bs_actual, xs.shape[1], device=xs.device)
        xs = xs[:, remaining_idx]
        rs = rs[:, remaining_idx]
        Gs = Gs[:, :, remaining_idx]
        # MATLAB lines 99-100: Move the batch to the GPU (cast to match inputDataClass)
        xi = xi.to(dtype=torch.float32, device=device)
        ri = ri.to(dtype=torch.float32, device=device)
        Gi = Gi.to(dtype=torch.float32, device=device)
        
        # MATLAB lines 102-131: Falsification
        # MATLAB line 105: Compute the sensitivity
        options.setdefault('nn', {})['_debug_iteration'] = totalNumSplits
        S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
        
        # MATLAB line 106
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32, device=device)
        S = torch.maximum(S, torch.tensor(1e-3, dtype=torch.float32, device=device))
        
        # MATLAB lines 108-109: sens = permute(sum(abs(S)),[2 1 3]); sens = sens(:,:);
        # S shape: (num_outputs, num_inputs, batch_size) -> sum over outputs -> (num_inputs, batch_size)
        sens = torch.sum(torch.abs(S), dim=0)  # (n0, bSz)
        sens = sens.reshape(sens.shape[0], -1)  # sens(:,:) flatten to 2D (n0, bSz)
        
        # MATLAB line 112: zi = xi + ri.*sign(sens);  (all are (n0, bSz))
        sens_sign = torch.sign(sens)  # (n0, bSz)
        zi = xi + ri * sens_sign
        
        # MATLAB line 114: Check adversarial examples (single FGSM direction)
        yi = nn.evaluate_(zi, options, idxLayer)
        
        if isinstance(yi, torch.Tensor):
            yi_torch = yi.to(dtype=torch.float32, device=device)
        else:
            yi_torch = torch.tensor(yi, dtype=torch.float32, device=device)
        
        num_outputs = A_torch.shape[1]
        yi_torch = yi_torch.reshape(num_outputs, -1)
        
        # MATLAB: checkSpecs = any(A*yi + b >= 0,1) for safeSet, all(A*yi + b <= 0,1) otherwise
        ld_yi = A_torch @ yi_torch + b_torch  # (num_constraints, batch_size)
        if safeSet:
            checkSpecs = torch.any(ld_yi >= 0, dim=0)
        else:
            checkSpecs = torch.all(ld_yi <= 0, dim=0)
        
        if torch.any(checkSpecs):
            if verbose:
                print("FALSIFICATION: Found potential counterexample in falsification phase")
                print(f"  ld_yi (A*yi + b) = {ld_yi.cpu().numpy().flatten()}")
                print(f"  b = {b_torch.cpu().numpy().flatten()}")
                print(f"  checkSpecs = {checkSpecs.cpu().numpy()}")
                print(f"  safeSet = {safeSet}")
            idNzEntry = torch.where(checkSpecs)[0]
            id_ = idNzEntry[0].item() if len(idNzEntry) > 0 else 0
            x_ = zi[:, id_].cpu().numpy().reshape(-1, 1)
            nn.castWeights(np.float32)
            y_ = nn.evaluate_(x_, options, idxLayer)
            if isinstance(y_, torch.Tensor):
                y_ = y_.cpu().numpy()
            if y_.ndim == 1:
                y_ = y_.reshape(-1, 1)
            res = 'COUNTEREXAMPLE'
            break

        # MATLAB lines 120-130
        # Always print debug output for falsification (critical for debugging)
        if verbose:
            print(f"\nFALSIFICATION DEBUG (safeSet={safeSet}):")
            print(f"  A_torch shape: {A_torch.shape}, yi_torch shape: {yi_torch.shape}, b_torch shape: {b_torch.shape}")
            print(f"  ld_yi shape: {ld_yi.shape}")
            ld_yi_np = ld_yi.cpu().numpy()
            print(f"  ld_yi values:\n{ld_yi_np}")
            if ld_yi.shape[1] > 5:
                print(f"  ld_yi (first 5 cols only):\n{ld_yi_np[:, :5]}")
            print(f"  b_torch: {b_torch.cpu().numpy().flatten()}")
            if safeSet:
                print(f"  For safeSet: checking any(ld_yi >= 0) along dim=0")
                ld_yi_ge_0 = (ld_yi >= 0).cpu().numpy()
                print(f"  ld_yi >= 0:\n{ld_yi_ge_0}")
                print(f"  any(ld_yi >= 0) per sample: {np.any(ld_yi_ge_0, axis=0)}")
            else:
                print(f"  For unsafeSet: checking all(ld_yi <= 0) along dim=0")
                ld_yi_le_0 = (ld_yi <= 0).cpu().numpy()
                print(f"  ld_yi <= 0:\n{ld_yi_le_0}")
                print(f"  all(ld_yi <= 0) per sample: {np.all(ld_yi_le_0, axis=0)}")
            checkSpecs_np = checkSpecs.cpu().numpy()
            print(f"  checkSpecs shape: {checkSpecs.shape}, checkSpecs: {checkSpecs_np}")
            print(f"  torch.any(checkSpecs): {torch.any(checkSpecs).item()}\n")
        
        if torch.any(checkSpecs):
            if verbose:
                print(f"FALSIFICATION: Found potential counterexample in falsification phase")
                print(f"  ld_yi (A*yi + b) = {ld_yi.cpu().numpy().flatten()}")
                print(f"  b = {b_torch.cpu().numpy().flatten()}")
                print(f"  checkSpecs = {checkSpecs.cpu().numpy()}")
                print(f"  safeSet = {safeSet}")
            idNzEntry = torch.where(checkSpecs)[0]
            id_ = idNzEntry[0].item() if len(idNzEntry) > 0 else 0
            # MATLAB: x_ = zi(:,id); - extract column vector (num_inputs, 1)
            x_ = zi[:, id_].cpu().numpy().reshape(-1, 1)
            # MATLAB: nn.castWeights(single(1));
            nn.castWeights(np.float32)
            # MATLAB: y_ = nn.evaluate_(gather(x_),options,idxLayer); % yi(:,id);
            # MATLAB re-evaluates for precision, but comment suggests using yi(:,id) directly
            # To match MATLAB exactly, we re-evaluate the network with the counterexample input
            # This ensures precision and matches MATLAB's behavior
            y_ = nn.evaluate_(x_, options, idxLayer)
            # Ensure y_ is a column vector (num_outputs, 1)
            if isinstance(y_, torch.Tensor):
                y_ = y_.cpu().numpy()
            if y_.ndim == 1:
                y_ = y_.reshape(-1, 1)
            elif y_.ndim == 2 and y_.shape[1] != 1:
                y_ = y_.T if y_.shape[0] == 1 else y_.reshape(-1, 1)
            res = 'COUNTEREXAMPLE'
            break
        
        # MATLAB lines 133-160: Verification
        # MATLAB lines 135-139: Use batch-evaluation
        if not options.get('nn', {}).get('interval_center', False):
            cxi = xi
        else:
            cxi = torch.tile(xi.reshape(xi.shape[0], xi.shape[1], 1), (1, 1, 2))
            cxi = cxi.permute(0, 2, 1)
        
        # MATLAB line 140
        # Use propagated generators Gi (already scaled to current radii)
        Gxi = Gi
        if options.get('nn', {}).get('_debug_verify', False):
            print(f"[DEBUG zonotope pre] Gxi min/max {float(Gxi.min())}/{float(Gxi.max())} "
                  f"ri min/max {float(ri.min())}/{float(ri.max())}")
        
        if cxi.ndim == 2:
            cxi = cxi.reshape(cxi.shape[0], 1, cxi.shape[1])
        
        # MATLAB line 141
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        if options.get('nn', {}).get('_debug_verify', False):
            gyi_min = float(Gyi.min()) if hasattr(Gyi, 'min') else None
            gyi_max = float(Gyi.max()) if hasattr(Gyi, 'max') else None
            print(f"[DEBUG zonotope post] Gyi min/max {gyi_min}/{gyi_max}")
        
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
                yi_torch = yi.to(dtype=torch.float32, device=device)
            else:
                yi_torch = torch.tensor(yi, dtype=torch.float32, device=device)
            yic = 0.5 * (yi_torch[:, 1, :] + yi_torch[:, 0, :])
            yid = 0.5 * (yi_torch[:, 1, :] - yi_torch[:, 0, :])
            dyi = A_torch @ yic + b_torch
            # Use torch einsum for pagemtimes
            ld_Gyi = torch.einsum('ij,jkb->ikb', A_torch, Gyi)
            # Use torch permute for pagetranspose equivalent
            yid_trans = yid.permute(1, 0)  # (batch, num_outputs) -> (num_outputs, batch)
            # Use torch unsqueeze instead of np.newaxis
            A_yid = A_torch.unsqueeze(2) * yid_trans.unsqueeze(0)  # (num_constraints, num_outputs, batch)
            dri = torch.sum(torch.abs(ld_Gyi), dim=1) + torch.sum(torch.abs(A_yid), dim=1)  # (num_constraints, batch)
        
        # MATLAB lines 156-160: Check specification (pure MATLAB logic)
        if safeSet:
            # MATLAB: checkSpecs = any(dyi(:,:) + dri(:,:) > 0,1);
            checkSpecs = torch.any(dyi + dri > 0, dim=0)
        else:
            # MATLAB: checkSpecs = all(dyi(:,:) - dri(:,:) < 0,1);
            checkSpecs = torch.all(dyi - dri < 0, dim=0)
        unknown = checkSpecs
        if options.get('nn', {}).get('_debug_verify', False):
            print(f'ITER DEBUG dyi min/max {float(dyi.min())}/{float(dyi.max())} '
                  f'dri min/max {float(dri.min())}/{float(dri.max())} '
                  f'unknown_count {int(torch.sum(unknown))}')
        
        # MATLAB lines 162-164: Gather from GPU before split (matching MATLAB exactly)
        # MATLAB: xi = gather(xi); ri = gather(ri); sens = gather(sens);
        # In Python, gather means move from GPU to CPU, but _aux_split can work on GPU
        # However, to match MATLAB exactly, we gather (move to CPU) before split
        # Note: _aux_split will move back to GPU if needed, but we match MATLAB's gather here
        xi_gathered = xi.cpu() if device.type == 'cuda' else xi
        ri_gathered = ri.cpu() if device.type == 'cuda' else ri
        sens_gathered = sens.cpu() if device.type == 'cuda' else sens
        unknown_gathered = unknown.cpu() if device.type == 'cuda' else unknown
        
        # MATLAB lines 166-167: Create new splits
        # MATLAB: [xis,ris] = aux_split(xi(:,unknown),ri(:,unknown),sens(:,unknown), nSplits,nDims);
        # sens is already (n0, batch); slice columns directly
        # Convert boolean mask to indices for proper indexing
        # unknown_gathered has shape (batch_size,), convert to indices
        unknown_indices = torch.where(unknown_gathered)[0]  # Get indices where True
        if len(unknown_indices) > 0:
            # Index using integer indices; ensure sens shape aligns with unknown batch
            sens_unknown = sens_gathered[:, unknown_indices]  # (n0, batch_unknown)
            # Heuristic for splitting: scale sensitivity by current radii (matches MATLAB intent)
            hi_unknown = ri_gathered[:, unknown_indices] * sens_unknown
            Gi_unknown = Gi[:, :, unknown_indices] if 'Gi' in locals() else batchG[:, :, :len(unknown_indices)]
            # Debug: log radii before split
            if options.get('nn', {}).get('_debug_verify', False):
                print(f"[DEBUG split pre] xi min/max {float(xi_gathered[:, unknown_indices].min())}/{float(xi_gathered[:, unknown_indices].max())} "
                      f"ri min/max {float(ri_gathered[:, unknown_indices].min())}/{float(ri_gathered[:, unknown_indices].max())}")
            # Split centers/radii and get split dims (MATLAB: [xis,ris,dimId] = aux_split(xis,ris,his,nSplits))
            xis, ris, dimIds = _aux_split_with_dim(
                xi_gathered[:, unknown_indices],
                ri_gathered[:, unknown_indices],
                hi_unknown,
                nSplits
            )
            if options.get('nn', {}).get('_debug_verify', False) and totalNumSplits < 50:
                print(f"[DEBUG split dims] dimIds (1-based) {dimIds.flatten().tolist()}")
            # Rebuild zonotope from scratch for new splits (match MATLAB aux_constructInputZonotope)
            # Use numInitGens = n0 (one generator per input dim)
            n0 = xi_gathered.shape[0]
            numInitGens = numGen  # match MATLAB: use all available generators
            cxi_children = []
            Gis_children = []
            for child_idx in range(ris.shape[1]):
                cxi_child, Gi_child, _ = _aux_constructInputZonotope(
                    options,  # pass full options (MATLAB passes options)
                    'linf',
                    xis[:, child_idx:child_idx+1],
                    ris[:, child_idx:child_idx+1],
                    batchG[:, :, :numInitGens],
                    None,
                    None,
                    numInitGens
                )
                cxi_children.append(cxi_child)
                Gis_children.append(Gi_child)
            xis = torch.cat(cxi_children, dim=1) if cxi_children else torch.empty((n0, 0), device=xi_gathered.device)
            Gis = torch.cat(Gis_children, dim=2) if Gis_children else torch.empty((n0, numGen, 0), device=xi_gathered.device)
            # Debug: log radii after split
            if options.get('nn', {}).get('_debug_verify', False):
                print(f"[DEBUG split post] xis min/max {float(xis.min())}/{float(xis.max())} "
                      f"ris min/max {float(ris.min())}/{float(ris.max())} "
                      f"Gis min/max {float(Gis.min())}/{float(Gis.max())}")
        else:
            # No unknown samples to split
            xis = torch.empty((xi_gathered.shape[0], 0), dtype=xi_gathered.dtype, device=xi_gathered.device)
            ris = torch.empty((ri_gathered.shape[0], 0), dtype=ri_gathered.dtype, device=ri_gathered.device)
            Gis = torch.empty((batchG.shape[0], batchG.shape[1], 0), dtype=batchG.dtype, device=batchG.device)
        
        # Move results back to original device (GPU if originally on GPU)
        if device.type == 'cuda':
            xis = xis.to(device)
            ris = ris.to(device)
            Gis = Gis.to(device)
        
        # MATLAB lines 169-170: Add new splits to the queue - xis, ris are already torch tensors
        xs = torch.cat([xis, xs], dim=1)
        rs = torch.cat([ris, rs], dim=1)
        Gs = torch.cat([Gis, Gs], dim=2)
        
        # MATLAB lines 172-173
        totalNumSplits = totalNumSplits + xis.shape[1]
        # unknown is torch tensor, use torch.sum
        verifiedPatches = verifiedPatches + xi.shape[1] - torch.sum(unknown).item()
        
        # MATLAB lines 175-177: To save memory, we clear all variables that are no longer used
        # Note: MATLAB clears 'xGi' but it's never defined, so we skip it
        del xi, ri, Gi, yi, Gyi, dyi, dri
    
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
