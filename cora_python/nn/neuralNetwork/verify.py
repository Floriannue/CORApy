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
    
    # Ensure x and r are 2D column vectors like MATLAB
    # MATLAB: x and r are column vectors (n, 1) or (n, num_patches)
    x = np.asarray(x)
    r = np.asarray(r)
    if x.ndim == 1:
        x = x.reshape(-1, 1)  # (n,) -> (n, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)  # (n,) -> (n, 1)
    # Ensure x and r have the same number of rows
    if x.shape[0] != r.shape[0]:
        raise ValueError(f"x and r must have the same number of rows: x.shape={x.shape}, r.shape={r.shape}")
    
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
    # MATLAB: xs = x; rs = r;
    xs = x.copy()  # (n, num_patches) - initially (n, 1) for single patch
    rs = r.copy()  # (n, num_patches) - initially (n, 1) for single patch
    
    if verbose:
        print(f"Initial radius: min={np.min(rs):.6f}, max={np.max(rs):.6f}")
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
        # MATLAB: sens = permute(sum(abs(S)),[2 1 3]); sens = sens(:,:);
        # S has shape (output_dim, input_dim, batch)
        # sum(abs(S)) over output_dim (axis=0) gives (input_dim, batch)
        # permute([2 1 3]) swaps first two dims: (input_dim, batch) -> (batch, input_dim)
        # sens(:,:) keeps it as (batch, input_dim)
        if useGpu and TORCH_AVAILABLE:
            # Convert sensitivity to GPU tensor if needed
            if isinstance(S, np.ndarray):
                S = torch.tensor(S, dtype=torch.float32, device=device)
            S = torch.maximum(S, torch.tensor(1e-3, dtype=torch.float32, device=device))
            # sum(abs(S)) over output dimension (dim=0): (output_dim, input_dim, batch) -> (input_dim, batch)
            sens_sum = torch.sum(torch.abs(S), dim=0)  # (input_dim, batch)
            # permute([2 1 3]) swaps first two dims: (input_dim, batch) -> (batch, input_dim)
            sens = sens_sum.T  # Transpose to get (batch, input_dim) to match MATLAB
        else:
            # Use NumPy operations - match MATLAB exactly
            S = np.maximum(S, 1e-3)
            # sum(abs(S)) over output dimension (axis=0): (output_dim, input_dim, batch) -> (input_dim, batch)
            sens_sum = np.sum(np.abs(S), axis=0)  # (input_dim, batch)
            # permute([2,1,3]) swaps first two dims: (input_dim, batch) -> (batch, input_dim)
            sens = sens_sum.T  # Transpose to get (batch, input_dim) to match MATLAB
        
        # 2. Compute adversarial attacks. We want to maximize A*yi + b;
        # therefore, ...
        # MATLAB: zi = xi + ri.*sign(sens);
        # xi, ri have shape (input_dim, batch), sens has shape (batch, input_dim)
        # Need to transpose sens to match xi, ri shapes for element-wise multiplication
        if useGpu and TORCH_AVAILABLE:
            # Use PyTorch operations for GPU
            # sens.T has shape (input_dim, batch) to match xi, ri
            zi = xi + ri * torch.sign(sens.T)
        else:
            # Use NumPy operations for CPU
            # sens.T has shape (input_dim, batch) to match xi, ri
            zi = xi + ri * np.sign(sens.T)
        # 3. Check adversarial examples.
        yi = nn.evaluate_(zi, options, idxLayer)
        if safeSet:
            checkSpecs = np.any(A @ yi + b >= 0, axis=0)
        else:
            checkSpecs = np.all(A @ yi + b <= 0, axis=0)
        
        # Debug adversarial attack results
        if verbose:
            attack_result = A @ yi + b
            print(f"DEBUG Adversarial: safeSet={safeSet}, attack range=[{np.min(attack_result):.6f}, {np.max(attack_result):.6f}], counterexamples found: {np.sum(checkSpecs)}")
            print(f"DEBUG Adversarial: checkSpecs={checkSpecs}")
            print(f"DEBUG Adversarial: xi.shape={xi.shape}, ri.shape={ri.shape}, sens.shape={sens.shape}")
            print(f"DEBUG Adversarial: zi.shape={zi.shape}, yi.shape={yi.shape}")
            print(f"DEBUG Adversarial: zi={zi.flatten()}")
            print(f"DEBUG Adversarial: yi={yi.flatten()}")
            
            if np.sum(checkSpecs) == 0:
                print(f"DEBUG Adversarial: ri range=[{np.min(ri):.6f}, {np.max(ri):.6f}], sens range=[{np.min(sens):.6f}, {np.max(sens):.6f}]")
        if np.any(checkSpecs):
            # Found a counterexample.
            res = 'COUNTEREXAMPLE'
            idNzEntry = np.where(checkSpecs)[0]
            id_ = idNzEntry[0]
            # MATLAB: x_ = zi(:,id);
            if useGpu and TORCH_AVAILABLE:
                x_ = zi[:, id_].cpu().numpy().reshape(-1, 1)
            else:
                x_ = zi[:, id_].reshape(-1, 1)
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
        # MATLAB: if ~options.nn.interval_center; cxi = xi; else; cxi = permute(repmat(xi,1,1,2),[1 3 2]); end
        if not options.get('nn', {}).get('interval_center', False):
            # MATLAB: cxi = xi; (2D: input_dim, batch)
            # But Python layers expect 3D, so reshape to (input_dim, 1, batch)
            if useGpu and TORCH_AVAILABLE:
                cxi = xi.reshape(xi.shape[0], 1, xi.shape[1])
            else:
                cxi = xi.reshape(xi.shape[0], 1, xi.shape[1])
        else:
            # MATLAB: cxi = permute(repmat(xi,1,1,2),[1 3 2]);
            # repmat(xi,1,1,2) creates (input_dim, batch, 2)
            # permute([1 3 2]) gives (input_dim, 2, batch)
            if useGpu and TORCH_AVAILABLE:
                cxi = torch.tile(xi.reshape(xi.shape[0], xi.shape[1], 1), (1, 1, 2))
                cxi = torch.permute(cxi, (0, 2, 1))  # (input_dim, 2, batch)
            else:
                cxi = np.tile(xi.reshape(xi.shape[0], xi.shape[1], 1), (1, 1, 2))
                cxi = np.transpose(cxi, (0, 2, 1))  # (input_dim, 2, batch)
        
        # MATLAB: Gxi = permute(ri,[1 3 2]).*batchG(:,:,1:size(ri,2));
        # permute(ri,[1 3 2]) reshapes ri from (input_dim, batch) to (input_dim, 1, batch)
        # batchG(:,:,1:size(ri,2)) is (input_dim, numGen, actual_batch_size)
        # Result: (input_dim, numGen, batch)
        actual_batch_size = ri.shape[1]  # This is the actual batch size after _aux_pop
        
        if useGpu and TORCH_AVAILABLE:
            # Convert batchG to GPU tensor
            batchG_gpu = torch.tensor(batchG, dtype=torch.float32, device=device)
            # permute(ri,[1 3 2]): reshape ri from (input_dim, batch) to (input_dim, 1, batch)
            ri_3d = ri.reshape(ri.shape[0], 1, ri.shape[1])  # (input_dim, 1, batch)
            # Multiply: (input_dim, 1, batch) * (input_dim, numGen, actual_batch_size)
            Gxi = ri_3d * batchG_gpu[:, :, :actual_batch_size]  # (input_dim, numGen, batch)
        else:
            # Use NumPy operations for CPU
            # permute(ri,[1 3 2]): reshape ri from (input_dim, batch) to (input_dim, 1, batch)
            ri_3d = ri.reshape(ri.shape[0], 1, ri.shape[1])  # (input_dim, 1, batch)
            # Multiply: (input_dim, 1, batch) * (input_dim, numGen, actual_batch_size)
            Gxi = ri_3d * batchG[:, :, :actual_batch_size]  # (input_dim, numGen, batch)
        
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        
        # 2. Compute logit-difference.
        if not options.get('nn', {}).get('interval_center', False):
            # MATLAB: dyi = A*yi + b;
            # yi has shape (n_out, 1, batch) or (n_out, batch)
            if yi.ndim == 3 and yi.shape[1] == 1:
                yi_squeezed = yi.squeeze(axis=1)  # (n_out, batch)
            else:
                yi_squeezed = yi  # (n_out, batch)
            
            # MATLAB: dyi = A*yi + b;
            dyi = A @ yi_squeezed + b  # Shape: (spec_dim, batch)
            
            # MATLAB: dri = sum(abs(pagemtimes(A,Gyi)),2);
            # pagemtimes(A,Gyi) where A is (spec_dim, n_out) and Gyi is (n_out, n_generators, batch)
            # returns (spec_dim, n_generators, batch), then sum over dimension 2 (generators) to get (spec_dim, batch)
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                # A @ Gyi for each batch: (spec_dim, n_out) @ (n_out, n_generators, batch) -> (spec_dim, n_generators, batch)
                AGyi = torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), 
                                    torch.tensor(Gyi, dtype=torch.float32, device=device))  # (spec_dim, n_generators, batch)
                dri = torch.sum(torch.abs(AGyi), dim=1)  # Sum over generators: (spec_dim, batch)
                dri = dri.cpu().numpy()
            else:
                # Use NumPy operations for CPU
                # Import pagemtimes from nnGeneratorReductionLayer
                from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
                
                # pagemtimes(A, 'none', Gyi, 'none') where A is 2D and Gyi is 3D
                # A: (spec_dim, n_out), Gyi: (n_out, n_generators, batch)
                # Result: (spec_dim, n_generators, batch)
                AGyi = pagemtimes(A, 'none', Gyi, 'none')  # (spec_dim, n_generators, batch)
                # Sum over generators (axis=1) to get (spec_dim, batch)
                dri = np.sum(np.abs(AGyi), axis=1)  # (spec_dim, batch)
        else:
            # Compute the center and the radius of the center-interval.
            # MATLAB: yic = 1/2*(yi(:,2,:) + yi(:,1,:));
            # MATLAB: yid = 1/2*(yi(:,2,:) - yi(:,1,:));
            yic = 1/2 * (yi[:, 1, :] + yi[:, 0, :])  # (n_out, batch)
            yid = 1/2 * (yi[:, 1, :] - yi[:, 0, :])  # (n_out, batch)
            
            # MATLAB: dyi = A*yic(:,:) + b;
            dyi = A @ yic + b  # (spec_dim, batch)
            
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                # MATLAB: dri = sum(abs(pagemtimes(A,Gyi)),2) + sum(abs(A.*pagetranspose(yid)),2);
                # pagemtimes(A,Gyi): (spec_dim, n_out) @ (n_out, n_generators, batch) -> (spec_dim, n_generators, batch)
                AGyi = torch.matmul(torch.tensor(A, dtype=torch.float32, device=device), 
                                    torch.tensor(Gyi, dtype=torch.float32, device=device))  # (spec_dim, n_generators, batch)
                dri1 = torch.sum(torch.abs(AGyi), dim=1)  # Sum over generators: (spec_dim, batch)
                
                # A.*pagetranspose(yid): element-wise multiply A with yid.T
                # A: (spec_dim, n_out), yid: (n_out, batch), pagetranspose(yid): (batch, n_out)
                yid_T = torch.tensor(yid.T, dtype=torch.float32, device=device)  # (batch, n_out)
                A_tensor = torch.tensor(A, dtype=torch.float32, device=device)  # (spec_dim, n_out)
                # Broadcast: (spec_dim, 1, n_out) * (1, batch, n_out) -> (spec_dim, batch, n_out)
                A_yid = A_tensor.unsqueeze(1) * yid_T.unsqueeze(0)  # (spec_dim, batch, n_out)
                dri2 = torch.sum(torch.abs(A_yid), dim=2)  # Sum over n_out: (spec_dim, batch)
                
                dri = (dri1 + dri2).cpu().numpy()  # (spec_dim, batch)
            else:
                # Use NumPy operations for CPU
                from ..layers.linear.nnGeneratorReductionLayer import pagemtimes, pagetranspose
                
                # MATLAB: dri = sum(abs(pagemtimes(A,Gyi)),2) + sum(abs(A.*pagetranspose(yid)),2);
                # pagemtimes(A,Gyi): (spec_dim, n_out) @ (n_out, n_generators, batch) -> (spec_dim, n_generators, batch)
                AGyi = pagemtimes(A, 'none', Gyi, 'none')  # (spec_dim, n_generators, batch)
                dri1 = np.sum(np.abs(AGyi), axis=1)  # Sum over generators: (spec_dim, batch)
                
                # A.*pagetranspose(yid): element-wise multiply
                # A: (spec_dim, n_out), yid: (n_out, batch), pagetranspose(yid): (batch, n_out)
                yid_T = pagetranspose(yid)  # (batch, n_out)
                # A.*yid_T: broadcast (spec_dim, n_out) with (batch, n_out) -> (spec_dim, batch, n_out)
                A_yid = A[:, np.newaxis, :] * yid_T[np.newaxis, :, :]  # (spec_dim, batch, n_out)
                dri2 = np.sum(np.abs(A_yid), axis=2)  # Sum over n_out: (spec_dim, batch)
                
                dri = dri1 + dri2  # (spec_dim, batch)
        # 3. Check specification.
        if safeSet:
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                # MATLAB: checkSpecs = any(dyi(:,:) + dri(:,:) > 0,1);
                checkSpecs = torch.any(dyi + dri > 0, dim=0)  # MATLAB logic: any spec violation per batch
            else:
                # Use NumPy operations for CPU
                # MATLAB: checkSpecs = any(dyi(:,:) + dri(:,:) > 0,1);
                # For each batch (column), check if ANY spec (row) is violated
                # dyi + dri has shape (num_specs, num_batches), we want (num_batches,)
                checkSpecs = np.any(dyi + dri > 0, axis=0)  # MATLAB logic: any spec violation per batch
        else:
            if useGpu and TORCH_AVAILABLE:
                # Use PyTorch operations for GPU
                # MATLAB: checkSpecs = all(dyi(:,:) - dri(:,:) < 0,1);
                checkSpecs = torch.all(dyi - dri < 0, dim=0)  # MATLAB logic: all specs must be satisfied per batch
            else:
                # Use NumPy operations for CPU
                # MATLAB: checkSpecs = all(dyi(:,:) - dri(:,:) < 0,1);
                # For each batch (column), check if ALL specs (rows) satisfy the condition
                # dyi - dri has shape (num_specs, num_batches), we want (num_batches,)
                checkSpecs = np.all(dyi - dri < 0, axis=0)  # MATLAB logic: all specs must be satisfied per batch
        
        # In MATLAB: unknown = checkSpecs
        # For safeSet=True: checkSpecs=true means violation found (needs splitting)
        # For safeSet=False: checkSpecs=true means all specs satisfied (needs splitting to find counterexample)
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
        # MATLAB: [xis,ris] = aux_split(xi(:,unknown),ri(:,unknown),sens(:,unknown),nSplits,nDims);
        # xi, ri have shape (input_dim, batch), sens has shape (batch, input_dim)
        # unknown has shape (batch,), so xi[:, unknown] selects columns, sens[unknown, :] selects rows
        # sens[unknown, :] has shape (num_unknown, input_dim)
        # For MATLAB: sens.*ri where sens is (batch, input_dim) and ri is (input_dim, batch)
        # MATLAB broadcasts: sens' .* ri which is (input_dim, batch)
        # So we need sens.T to match ri shape: (input_dim, batch)
        # Then sens[:, unknown] selects columns: (input_dim, num_unknown)
        xis, ris = _aux_split(xi[:, unknown], ri[:, unknown], sens[unknown, :].T, nSplits, nDims)
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
    """
    Split the input for verification (MATLAB aux_split equivalent)
    
    MATLAB signature:
    function [xis,ris] = aux_split(xi,ri,sens,nSplits,nDims)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    n, bs = xi.shape
    # Cannot split more than every dimension.
    nDims = min(n, nDims)
    # Split each input in the batch into nSplits parts; use radius*sens 
    # as the splitting heuristic.
    # 1. Find the input dimension with the largest heuristic.
    # MATLAB: [~,sortDims] = sort(abs(sens.*ri),1,'descend');
    # sens and ri both have shape (input_dim, batch)
    # sort along dimension 1 (columns), descending
    sortDims = np.argsort(np.abs(sens * ri), axis=0)[::-1]  # Sort descending along axis=0 (rows)
    # MATLAB: dimIds = sortDims(1:nDims,:);
    dimIds = sortDims[:nDims, :]  # Shape: (nDims, batch)
    
    # MATLAB: splitsIdx = repmat(1:nSplits,1,bs);
    splitsIdx = np.tile(np.arange(1, nSplits + 1), bs)  # 1-based like MATLAB: (nSplits*bs,)
    # MATLAB: bsIdx = repelem((1:bs)',nSplits);
    # (1:bs)' is a column vector, repelem repeats each element nSplits times
    bsIdx = repelem(np.arange(1, bs + 1), nSplits)  # 1-based: (bs*nSplits,)
    
    # MATLAB: dim = dimIds(1,:);
    dim = dimIds[0, :]  # Shape: (batch,), 1-based dimension indices
    
    # MATLAB: linIdx = sub2ind([n bs nSplits], repelem(dim,nSplits),bsIdx(:)',splitsIdx(:)');
    # repelem(dim,nSplits): repeat each element of dim nSplits times
    dim_repeated = repelem(dim, nSplits)  # Shape: (batch*nSplits,), 1-based
    # sub2ind([n bs nSplits], dim_repeated, bsIdx, splitsIdx)
    # All inputs are 1-based MATLAB indices
    linIdx = sub2ind((n, bs, nSplits), dim_repeated, bsIdx, splitsIdx)  # 1-based linear indices
    
    # 2. Split the selected dimension.
    # MATLAB: xi_ = xi; ri_ = ri;
    xi_ = xi.copy()
    ri_ = ri.copy()
    # Shift to the lower bound.
    # MATLAB: dimIdx = sub2ind([n bs],dim,1:bs);
    dimIdx = sub2ind((n, bs), dim, np.arange(1, bs + 1))  # 1-based linear indices
    # MATLAB: xi_(dimIdx) = xi_(dimIdx) - ri(dimIdx);
    # Convert 1-based indices to 0-based for Python indexing
    dimIdx_0based = dimIdx - 1
    xi_flat = xi_.flatten()
    ri_flat = ri_.flatten()
    xi_flat[dimIdx_0based] = xi_flat[dimIdx_0based] - ri_flat[dimIdx_0based]
    xi_ = xi_flat.reshape(n, bs)
    # MATLAB: ri_(dimIdx) = ri_(dimIdx)/nSplits;
    ri_flat = ri_.flatten()
    ri_flat[dimIdx_0based] = ri_flat[dimIdx_0based] / nSplits
    ri_ = ri_flat.reshape(n, bs)
    
    # MATLAB: xis = repmat(xi_,1,1,nSplits);
    xis = np.tile(xi_.reshape(n, bs, 1), (1, 1, nSplits))  # Shape: (n, bs, nSplits)
    # MATLAB: ris = repmat(ri_,1,1,nSplits);
    ris = np.tile(ri_.reshape(n, bs, 1), (1, 1, nSplits))  # Shape: (n, bs, nSplits)
    
    # MATLAB: xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    # Offset the center.
    # Convert 1-based linIdx to 0-based for Python indexing
    linIdx_0based = linIdx - 1
    xis_flat = xis.flatten()
    ris_flat = ris.flatten()
    # splitsIdx is 1-based (1, 2, ..., nSplits), convert to 0-based for calculation
    splitsIdx_0based = splitsIdx - 1  # Now 0, 1, ..., nSplits-1
    xis_flat[linIdx_0based] = xis_flat[linIdx_0based] + (2 * splitsIdx_0based - 1) * ris_flat[linIdx_0based]
    xis = xis_flat.reshape(n, bs, nSplits)
    
    # MATLAB: xis = xis(:,:); ris = ris(:,:);
    # Flatten last two dimensions: (n, bs, nSplits) -> (n, bs*nSplits)
    xis = xis.reshape(n, -1)
    ris = ris.reshape(n, -1)
    
    return xis, ris
