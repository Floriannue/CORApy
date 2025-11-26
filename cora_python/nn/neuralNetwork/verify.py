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
from typing import Optional, Tuple, Any, Dict, List, TYPE_CHECKING
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


def verify(nn: 'NeuralNetwork', x: np.ndarray, r: np.ndarray, A: np.ndarray, b: np.ndarray, 
           safeSet: Any, options: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, 
           verbose: bool = False, plotDims: Optional[Any] = None, 
           plotSplittingTree: bool = False) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
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
    # r can be scalar or array
    if np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0):
        # r is scalar, broadcast to match x shape
        r = np.full((x.shape[0], 1), float(r))
    elif r.ndim == 1:
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
        
        # Store ld_Gyi for zonotack falsification method (computed later in verification section)
        ld_Gyi = None
        
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
        # MATLAB: [ld_yi,ld_Gyi,ld_Gyi_err,yic,yid,Gyi] = aux_computeLogitDifference(yi,Gyi,A,options);
        # Match MATLAB aux_computeLogitDifference exactly
        if not options.get('nn', {}).get('interval_center', False):
            # The center is just a vector.
            # MATLAB: yic = yi;
            if yi.ndim == 3 and yi.shape[1] == 1:
                yic = yi.squeeze(axis=1)  # (n_out, batch)
            else:
                yic = yi  # (n_out, batch)
            # There are no approximation errors stored in the center.
            # MATLAB: yid = zeros([nK 1 bSz],'like',yi);
            nK = yic.shape[0]
            bSz = yic.shape[1] if yic.ndim == 2 else 1
            if useGpu and TORCH_AVAILABLE:
                yid = torch.zeros((nK, 1, bSz), dtype=torch.float32, device=device)
            else:
                yid = np.zeros((nK, 1, bSz), dtype=yic.dtype)
        else:
            # Compute the center and the radius of the center-interval.
            # MATLAB: yic = reshape(1/2*(yi(:,2,:) + yi(:,1,:)),[nK bSz]);
            # MATLAB: yid = 1/2*(yi(:,2,:) - yi(:,1,:));
            nK = yi.shape[0]
            bSz = yi.shape[2] if yi.ndim == 3 else 1
            yic = 1/2 * (yi[:, 1, :] + yi[:, 0, :])  # (n_out, batch)
            yic = yic.reshape(nK, bSz)  # Ensure 2D: (n_out, batch)
            yid = 1/2 * (yi[:, 1, :] - yi[:, 0, :])  # (n_out, batch) - but MATLAB keeps it 3D: (nK, 1, bSz)
            if yid.ndim == 2:
                yid = yid.reshape(nK, 1, bSz)  # (n_out, 1, batch)
        
        # Compute the logit difference of the input generators.
        # MATLAB: ld_yi = A*yic;
        # Note: NO +b here! b is added later in the unknown check
        if useGpu and TORCH_AVAILABLE:
            if isinstance(yic, np.ndarray):
                yic_tensor = torch.tensor(yic, dtype=torch.float32, device=device)
            else:
                yic_tensor = yic
            if isinstance(A, np.ndarray):
                A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
            else:
                A_tensor = A
            ld_yi = torch.matmul(A_tensor, yic_tensor)  # (spec_dim, batch)
            ld_yi = ld_yi.cpu().numpy()
        else:
            ld_yi = A @ yic  # (spec_dim, batch)
        
        # MATLAB: ld_Gyi = pagemtimes(A,Gyi);
        if useGpu and TORCH_AVAILABLE:
            if isinstance(Gyi, np.ndarray):
                Gyi_tensor = torch.tensor(Gyi, dtype=torch.float32, device=device)
            else:
                Gyi_tensor = Gyi
            if isinstance(A, np.ndarray):
                A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
            else:
                A_tensor = A
            # pagemtimes: (spec_dim, n_out) @ (n_out, n_generators, batch) -> (spec_dim, n_generators, batch)
            ld_Gyi = torch.einsum('ij,jkl->ikl', A_tensor, Gyi_tensor)  # (spec_dim, n_generators, batch)
            ld_Gyi = ld_Gyi.cpu().numpy()
        else:
            from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
            ld_Gyi = pagemtimes(A, 'none', Gyi, 'none')  # (spec_dim, n_generators, batch)
        
        # Compute logit difference of the approximation errors.
        # MATLAB: ld_Gyi_err = sum(abs(A.*permute(yid,[2 1 3])),2);
        # permute(yid,[2 1 3]): (nK, 1, bSz) -> (1, nK, bSz)
        # A.*permute(yid,[2 1 3]): element-wise multiply A (spec_dim, nK) with (1, nK, bSz)
        # Broadcasting: (spec_dim, nK) .* (1, nK, bSz) -> (spec_dim, nK, bSz)
        # sum(abs(...), 2): sum over nK dimension -> (spec_dim, bSz) = (spec_dim, batch)
        if useGpu and TORCH_AVAILABLE:
            if isinstance(yid, np.ndarray):
                yid_tensor = torch.tensor(yid, dtype=torch.float32, device=device)
            else:
                yid_tensor = yid
            if isinstance(A, np.ndarray):
                A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
            else:
                A_tensor = A
            # permute(yid,[2 1 3]): (nK, 1, bSz) -> (1, nK, bSz)
            yid_perm = yid_tensor.permute(1, 0, 2)  # (1, nK, bSz)
            # A.*yid_perm: broadcast (spec_dim, nK) with (1, nK, bSz) -> (spec_dim, nK, bSz)
            A_yid = A_tensor.unsqueeze(2) * yid_perm  # (spec_dim, nK, bSz)
            ld_Gyi_err = torch.sum(torch.abs(A_yid), dim=1)  # Sum over nK: (spec_dim, bSz)
            ld_Gyi_err = ld_Gyi_err.cpu().numpy()
        else:
            # permute(yid,[2 1 3]): (nK, 1, bSz) -> (1, nK, bSz)
            yid_perm = np.transpose(yid, (1, 0, 2))  # (1, nK, bSz)
            # A.*yid_perm: broadcast (spec_dim, nK) with (1, nK, bSz) -> (spec_dim, nK, bSz)
            A_yid = A[:, :, np.newaxis] * yid_perm  # (spec_dim, nK, bSz)
            ld_Gyi_err = np.sum(np.abs(A_yid), axis=1)  # Sum over nK: (spec_dim, bSz)
        
        # Compute the radius of the logit difference.
        # MATLAB: ld_ri = sum(abs(ld_Gyi),2) + ld_Gyi_err;
        # sum(abs(ld_Gyi),2): sum over generators (axis=1) -> (spec_dim, batch)
        # ld_Gyi_err: (spec_dim, batch)
        ld_ri = np.sum(np.abs(ld_Gyi), axis=1) + ld_Gyi_err  # (spec_dim, batch)
        
        # 2.3. Check specification.
        # MATLAB: if safeSet
        #     unknown = any(ld_yi + ld_ri(:,:) > b,1);
        # else
        #     unknown = all(ld_yi - ld_ri(:,:) <= b,1);
        # end
        if safeSet:
            # safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
            # Thus, unknown if any(A*y > b).
            # MATLAB: unknown = any(ld_yi + ld_ri(:,:) > b,1);
            unknown = np.any(ld_yi + ld_ri > b, axis=0)  # (batch,)
        else:
            # unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
            # Thus, unknown if all(A*y <= b).
            # MATLAB: unknown = all(ld_yi - ld_ri(:,:) <= b,1);
            unknown = np.all(ld_yi - ld_ri <= b, axis=0)  # (batch,)
        
        # Update counter for verified patches.
        # MATLAB: numVerified = numVerified + sum(~unknown,'all');
        verifiedPatches += np.sum(~unknown)
        
        # MATLAB: if all(~unknown)
        if np.all(~unknown):
            # Verified all subsets of the current batch. We can skip to next iteration.
            # MATLAB: iter = iter + 1; continue;
            continue
        elif np.any(~unknown):
            # Only keep un-verified patches.
            # MATLAB removes verified patches from all arrays using ~unknown indexing
            # In Python, we need to keep only the unknown (unverified) patches
            unknown_indices = np.where(unknown)[0]  # Indices of unknown (unverified) patches
            
            # Queue entries: keep only unknown patches
            # MATLAB: xi(:,~unknown) = []; -> keep xi(:,unknown)
            xi = xi[:, unknown_indices]  # Keep only unknown patches
            ri = ri[:, unknown_indices]
            # Note: nrXi, inputDimIdx, S, sens are not used in current Python implementation
            # but if they exist, they should also be filtered
            
            # Input sets: keep only unknown patches
            # Note: In Python, cxi is always 3D (input_dim, 1 or 2, batch) even when interval_center is False
            # because we reshape it to 3D for layer compatibility
            if options.get('nn', {}).get('interval_center', False):
                # MATLAB: cxi(:,:,~unknown) = []; -> keep cxi(:,:,unknown)
                # cxi is (input_dim, 2, batch)
                cxi = cxi[:, :, unknown_indices]
                yi = yi[:, :, unknown_indices]
            else:
                # MATLAB: cxi(:,~unknown) = []; -> keep cxi(:,unknown)
                # But in Python, cxi is (input_dim, 1, batch), so we need to index the last dimension
                cxi = cxi[:, :, unknown_indices]  # (input_dim, 1, batch) -> (input_dim, 1, num_unknown)
                # yi might be 2D or 3D depending on evaluateZonotopeBatch_ output
                if yi.ndim == 3:
                    yi = yi[:, :, unknown_indices]
                else:
                    yi = yi[:, unknown_indices]
            Gxi = Gxi[:, :, unknown_indices]
            
            # Output sets: keep only unknown patches
            # MATLAB: yic(:,~unknown) = []; Gyi(:,:,~unknown) = []; yid(:,:,~unknown) = [];
            yic = yic[:, unknown_indices]
            Gyi = Gyi[:, :, unknown_indices]
            if yid.ndim == 3:
                yid = yid[:, :, unknown_indices]
            else:
                yid = yid[:, unknown_indices]
            
            # Logit difference: keep only unknown patches
            # MATLAB: ld_yi(:,~unknown) = []; ld_Gyi(:,:,~unknown) = [];
            ld_yi = ld_yi[:, unknown_indices]
            ld_Gyi = ld_Gyi[:, :, unknown_indices]
            
            # Update batch size after filtering
            cbSz = xi.shape[1]
        
        # Falsification -------------------------------------------------------
        # 2.1. Compute adversarial examples based on falsification_method
        falsification_method = options.get('nn', {}).get('falsification_method', 'fgsm')
        cbSz = xi.shape[1]  # Current batch size (after filtering verified patches)
        n0 = xi.shape[0]  # Number of input dimensions
        
        # Initialize sens for splitting (needed later)
        sens = None
        
        if falsification_method == 'zonotack':
            # Zonotack method: uses zonotope generators to compute attack
            # MATLAB: beta_ = -permute(sign(ld_Gyi(:,1:numInitGens,:)),[2 4 1 3]);
            # Requires ld_Gyi which should be computed in verification section above
            if ld_Gyi is None:
                # ld_Gyi not computed yet, fall back to center
                zi = xi
            else:
                # Obtain number of constraints
                p = A.shape[0] if A.ndim == 2 else 1
                numInitGens = options.get('nn', {}).get('train', {}).get('num_init_gens', n0)
                numInitGens = min(numInitGens, n0)
                numUnionConst = 1 if not safeSet else (safeSet if isinstance(safeSet, (int, np.integer)) and safeSet > 1 else A.shape[0])
                
                # MATLAB: beta_ = -permute(sign(ld_Gyi(:,1:numInitGens,:)),[2 4 1 3]);
                # ld_Gyi has shape (p, num_gens, cbSz) = (num_constraints, num_generators, batch)
                # We take ld_Gyi[:, 1:numInitGens, :] = (p, numInitGens, cbSz)
                # permute([2 4 1 3]) on 3D array adds singleton dimension: (p, numInitGens, cbSz) -> (numInitGens, 1, p, cbSz)
                # MATLAB's permute with 4 indices on 3D array: dimension 2 (4 in permute) becomes singleton
                ld_Gyi_subset = ld_Gyi[:, :numInitGens, :]  # (p, numInitGens, cbSz)
            if useGpu and TORCH_AVAILABLE:
                beta_ = -torch.sign(torch.tensor(ld_Gyi_subset, dtype=torch.float32, device=device))
                # permute [2 4 1 3]: (p, numInitGens, cbSz) -> (numInitGens, 1, p, cbSz)
                # Add singleton dimension at position 1, then permute
                beta_ = beta_.unsqueeze(1)  # (p, 1, numInitGens, cbSz)
                beta_ = beta_.permute(2, 1, 0, 3)  # (numInitGens, 1, p, cbSz)
                if safeSet:
                    # MATLAB: beta_(:,:,1:numUnionConst,:) = -beta_(:,:,1:numUnionConst,:);
                    beta_[:, :, :numUnionConst, :] = -beta_[:, :, :numUnionConst, :]
                # reshape to (numInitGens, 1, p*cbSz)
                beta = beta_.reshape(numInitGens, 1, p * cbSz)
                # Compute attack: delta = pagemtimes(repelem(Gxi(:,1:numInitGens,:),1,1,p),beta)
                # Gxi has shape (n0, num_gens, cbSz)
                # repelem(Gxi(:,1:numInitGens,:),1,1,p) repeats along 3rd dim: (n0, numInitGens, p*cbSz)
                Gxi_subset = Gxi[:, :numInitGens, :]  # (n0, numInitGens, cbSz)
                Gxi_repeated = Gxi_subset.repeat(1, 1, p)  # (n0, numInitGens, p*cbSz)
                # pagemtimes: (n0, numInitGens, p*cbSz) @ (numInitGens, 1, p*cbSz) = (n0, 1, p*cbSz)
                delta = torch.einsum('ijk,jlk->ilk', Gxi_repeated, beta)  # (n0, 1, p*cbSz)
                delta = delta.squeeze(1)  # (n0, p*cbSz)
                # xi_ = repelem(xi,1,p) + delta(:,:)
                xi_repeated = xi.repeat(1, p)  # (n0, p*cbSz)
                zi = xi_repeated + delta
                zi = zi.cpu().numpy() if isinstance(xi, np.ndarray) else zi
            else:
                beta_ = -np.sign(ld_Gyi_subset)  # (p, numInitGens, cbSz)
                # permute [2 4 1 3]: (p, numInitGens, cbSz) -> (numInitGens, 1, p, cbSz)
                # Add singleton dimension at position 1, then permute
                beta_ = beta_[:, np.newaxis, :, :]  # (p, 1, numInitGens, cbSz)
                beta_ = np.transpose(beta_, (2, 1, 0, 3))  # (numInitGens, 1, p, cbSz)
                if safeSet:
                    # MATLAB: beta_(:,:,1:numUnionConst,:) = -beta_(:,:,1:numUnionConst,:);
                    beta_[:, :, :numUnionConst, :] = -beta_[:, :, :numUnionConst, :]
                # reshape to (numInitGens, 1, p*cbSz)
                beta = beta_.reshape(numInitGens, 1, p * cbSz)
                # Compute attack: delta = pagemtimes(repelem(Gxi(:,1:numInitGens,:),1,1,p),beta)
                Gxi_subset = Gxi[:, :numInitGens, :]  # (n0, numInitGens, cbSz)
                Gxi_repeated = np.repeat(Gxi_subset, p, axis=2)  # (n0, numInitGens, p*cbSz)
                # pagemtimes: (n0, numInitGens, p*cbSz) @ (numInitGens, 1, p*cbSz) = (n0, 1, p*cbSz)
                # Use einsum: 'ijk,jlk->ilk'
                delta = np.einsum('ijk,jlk->ilk', Gxi_repeated, beta)  # (n0, 1, p*cbSz)
                delta = delta.squeeze(1)  # (n0, p*cbSz)
                # xi_ = repelem(xi,1,p) + delta(:,:)
                xi_repeated = np.repeat(xi, p, axis=1)  # (n0, p*cbSz)
                zi = xi_repeated + delta
        elif falsification_method == 'center':
            # Use the center for falsification
            zi = xi
        else:  # 'fgsm' or default
            # FGSM method: uses sensitivity to compute gradient-based attack
            # 1. Compute the sensitivity.
            S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
            
            # Compute sens for splitting (needed later)
            # MATLAB: sens = reshape(max(max(abs(S),1e-6),[],1),[n0 cbSz]);
            if S.ndim == 3:
                S_abs = np.abs(S)
                S_max = np.maximum(S_abs, 1e-6)
                sens_max = np.max(S_max, axis=0)  # (input_dim, batch)
                sens = sens_max.T  # (batch, input_dim)
            else:
                # Fallback: create default sensitivity
                sens = np.ones((cbSz, n0), dtype=xi.dtype)
                
                # Ensure S has correct shape: (output_dim, input_dim, batch)
                # calcSensitivity should return 3D array (nK, input_dim, bSz), but handle edge cases
                if S.ndim == 0:
                    # S is scalar, this is an error - should not happen
                    raise ValueError(f"calcSensitivity returned scalar S (shape={S.shape}), expected at least 2D. "
                                   f"This indicates a bug in evaluateSensitivity. xi.shape={xi.shape}")
                elif S.ndim == 1:
                    # S is 1D, reshape based on context
                    # If S.shape[0] == xi.shape[0], it's likely (input_dim,), reshape to (1, input_dim, 1)
                    # Otherwise, it might be (output_dim,), reshape to (output_dim, 1, 1)
                    if S.shape[0] == xi.shape[0]:
                        S = S.reshape(1, S.shape[0], 1)
                    else:
                        S = S.reshape(S.shape[0], 1, 1)
                elif S.ndim == 2:
                    # S is 2D, add batch dimension: (output_dim, input_dim) -> (output_dim, input_dim, 1)
                    S = S.reshape(S.shape[0], S.shape[1], 1)
                # If S.ndim == 3, it's already correct shape (output_dim, input_dim, batch)
            
            # MATLAB FGSM: grad = pagemtimes(A,S) or pagemtimes(-A,S) for safeSet
            # S has shape (nK, n0, cbSz) = (output_dim, input_dim, batch)
            # A has shape (p, nK) = (num_constraints, output_dim)
            # pagemtimes(A, S) computes A @ S for each batch: (p, n0, cbSz)
            p_orig = A.shape[0] if A.ndim == 2 else 1
            if useGpu and TORCH_AVAILABLE:
                if isinstance(S, np.ndarray):
                    S = torch.tensor(S, dtype=torch.float32, device=device)
                if isinstance(A, np.ndarray):
                    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
                else:
                    A_tensor = A
                # pagemtimes: for each batch b: grad[:,:,b] = A @ S[:,:,b]
                # S: (nK, n0, cbSz), A: (p, nK), result: (p, n0, cbSz)
                if safeSet:
                    grad = -torch.einsum('ij,jkl->ikl', A_tensor, S)  # (p_orig, n0, cbSz)
                    # MATLAB: We combine all constraints for a stronger attack.
                    # This means we sum over the constraint dimension
                    # grad has shape (p_orig, n0, cbSz), we sum over p_orig to get (1, n0, cbSz)
                    grad = torch.sum(grad, dim=0, keepdim=True)  # (1, n0, cbSz)
                    p = 1  # Combine all constraints for safe sets
                else:
                    grad = torch.einsum('ij,jkl->ikl', A_tensor, S)  # (p_orig, n0, cbSz)
                    p = p_orig
                # sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p])
                # sign(grad): (p, n0, cbSz) where p is the final value (1 for safeSet, p_orig otherwise)
                # permute([2 3 1]): (n0, cbSz, p)
                # reshape([n0 cbSz*p]): (n0, cbSz*p)
                sgrad = torch.sign(grad).permute(1, 2, 0).reshape(n0, cbSz * p)
                # xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad
                xi_repeated = xi.repeat(1, p)  # (n0, cbSz*p)
                ri_repeated = ri.repeat(1, p)  # (n0, cbSz*p)
                zi = xi_repeated + ri_repeated * sgrad
                zi = zi.cpu().numpy() if isinstance(xi, np.ndarray) else zi
            else:
                # NumPy implementation
                # pagemtimes: for each batch b: grad[:,:,b] = A @ S[:,:,b]
                # S: (nK, n0, cbSz), A: (p, nK), result: (p, n0, cbSz)
                if safeSet:
                    grad = -np.einsum('ij,jkl->ikl', A, S)  # (p_orig, n0, cbSz)
                    # MATLAB: We combine all constraints for a stronger attack.
                    # This means we sum over the constraint dimension
                    # grad has shape (p_orig, n0, cbSz), we sum over p_orig to get (1, n0, cbSz)
                    grad = np.sum(grad, axis=0, keepdims=True)  # (1, n0, cbSz)
                    p = 1  # Combine all constraints for safe sets
                else:
                    grad = np.einsum('ij,jkl->ikl', A, S)  # (p_orig, n0, cbSz)
                    p = p_orig
                # sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p])
                # sign(grad): (p, n0, cbSz) where p is the final value (1 for safeSet, p_orig otherwise)
                # permute([2 3 1]): (n0, cbSz, p)
                # reshape([n0 cbSz*p]): (n0, cbSz*p)
                sgrad = np.sign(grad).transpose(1, 2, 0).reshape(n0, cbSz * p)
                # xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad
                xi_repeated = np.repeat(xi, p, axis=1)  # (n0, cbSz*p)
                ri_repeated = np.repeat(ri, p, axis=1)  # (n0, cbSz*p)
                zi = xi_repeated + ri_repeated * sgrad
        # 2.2. Check the specification for adversarial examples.
        # MATLAB: [~,critVal,falsified,x_,y_] = aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);
        # Match MATLAB aux_checkPoints exactly (lines 985-1004)
        yi = nn.evaluate_(zi, options, idxLayer)
        # MATLAB: ld_ys = A*ys;
        ld_yi = A @ yi  # logit difference: A*yi, shape (num_constraints, batch_size)
        # MATLAB: critValPerConstr = ld_ys - b;
        critValPerConstr = ld_yi - b  # (num_constraints, batch_size)
        if safeSet:
            # MATLAB: falsified = any(ld_ys > b,1);
            # safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
            # Thus, unsafe if any(A*y > b).
            checkSpecs = np.any(ld_yi > b, axis=0)  # (batch_size,)
            critValPerConstr = -critValPerConstr
            critVal = np.min(critValPerConstr, axis=0)  # (batch_size,)
        else:
            # MATLAB: falsified = all(ld_ys <= b,1);
            # unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
            # Thus, unsafe if all(A*y <= b).
            # Note: For unsafe sets, if all constraints are satisfied (all A*y <= b),
            # it means the property is NOT satisfied (it's unsafe), so we found a counterexample.
            checkSpecs = np.all(ld_yi <= b, axis=0)  # (batch_size,)
            critVal = np.max(critValPerConstr, axis=0)  # (batch_size,)
        
        # MATLAB: Check if the batch was extended with multiple candidates.
        # if size(critVal,2) > cbSz
        #     critVal_ = reshape(critVal,1,cbSz,[]);
        #     critVal = min(critVal_,[],3);
        # end
        # This handles the case where falsification_method='fgsm' or 'zonotack' creates multiple candidates per batch
        # zi might have shape (n0, cbSz*p) where p is number of constraints
        if zi.shape[1] > cbSz:
            # Multiple candidates per original batch entry
            # critVal has shape (cbSz*p,) where p is number of constraints
            # Reshape to (1, cbSz, p) and take min over p dimension
            p_candidates = zi.shape[1] // cbSz
            critVal_reshaped = critVal.reshape(cbSz, p_candidates)  # (cbSz, p)
            critVal = np.min(critVal_reshaped, axis=1)  # (cbSz,)
        
        # Debug adversarial attack results
        if verbose:
            attack_result = ld_yi - b  # A*y - b, positive means violation (A*y > b)
            print(f"DEBUG Adversarial: safeSet={safeSet}, attack range=[{np.min(attack_result):.6f}, {np.max(attack_result):.6f}], counterexamples found: {np.sum(checkSpecs)}")
            print(f"DEBUG Adversarial: checkSpecs={checkSpecs}")
            sens_str = f"sens.shape={sens.shape}" if sens is not None else "sens=None"
            print(f"DEBUG Adversarial: xi.shape={xi.shape}, ri.shape={ri.shape}, {sens_str}")
            print(f"DEBUG Adversarial: zi.shape={zi.shape}, yi.shape={yi.shape}")
            print(f"DEBUG Adversarial: zi={zi.flatten()}")
            print(f"DEBUG Adversarial: yi={yi.flatten()}")
            
            if np.sum(checkSpecs) == 0:
                sens_range_str = f"sens range=[{np.min(sens):.6f}, {np.max(sens):.6f}]" if sens is not None else "sens=None"
                print(f"DEBUG Adversarial: ri range=[{np.min(ri):.6f}, {np.max(ri):.6f}], {sens_range_str}")
        
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
            # Gathering weights from gpu. There is are precision error when using single gpuArray.
            # In MATLAB: nn.castWeights(single(1)); y_ = nn.evaluate_(gather(x_),options,idxLayer);
            nn.castWeights(np.float32)
            # In MATLAB: gather(x_) moves data from GPU to CPU. For Python: x_ is already on CPU.
            y_ = nn.evaluate_(x_, options, idxLayer)
            break
        
        # 3. Refine input sets. -------------------------------------------
        # Extract refinement method
        refinement_method = options.get('nn', {}).get('refinement_method', 'naive')
        
        # Extract parameters needed for refinement
        nNeur = options.get('nn', {}).get('num_neuron_splits', 0)
        nReLU = options.get('nn', {}).get('num_relu_constraints', 0)
        numInitGens = options.get('nn', {}).get('train', {}).get('num_init_gens', n0)
        numInitGens = min(numInitGens, n0)
        numUnionConst = 1 if not safeSet else (safeSet if isinstance(safeSet, (int, np.integer)) and safeSet > 1 else A.shape[0])
        
        # Get input dimension indices (for zonotack refinement)
        # MATLAB: inputDimIdx is computed in aux_constructInputZonotope
        # For now, we'll use the first numInitGens dimensions
        inputDimIdx = np.arange(numInitGens).reshape(-1, 1)  # (numInitGens, 1)
        inputDimIdx = np.tile(inputDimIdx, (1, cbSz))  # (numInitGens, cbSz)
        
        # Get q (number of generators)
        q = numGen
        
        # Initialize nrXi (neuron split indices) - not used in naive mode
        nrXi = np.zeros((0, cbSz), dtype=xi.dtype)
        
        if refinement_method == 'naive':
            # The sets are not refined; split the input dimensions.
            xis = xi.copy()
            ris = ri.copy()
            # Store the indices of the split dimensions.
            dimIds = np.full((0, cbSz), np.nan, dtype=xi.dtype)
            
            # sens might not exist if falsification_method was 'center' or 'zonotack'
            if 'sens' not in locals() or sens is None:
                # Compute sensitivity for splitting
                S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
                # Compute sens from S (same logic as in falsification section)
                if S.ndim == 3:
                    S_abs = np.abs(S)
                    S_max = np.maximum(S_abs, 1e-6)
                    sens_max = np.max(S_max, axis=0)  # (input_dim, batch)
                    sens = sens_max.T  # (batch, input_dim)
                else:
                    # Fallback: create default sensitivity
                    sens = np.ones((cbSz, n0), dtype=xi.dtype)
            
            # Ensure sens has correct shape (input_dim, batch) for splitting
            if sens.ndim == 2 and sens.shape[0] == cbSz:
                sens = sens.T  # (input_dim, batch)
            elif sens.ndim == 1:
                sens = sens.reshape(-1, 1)  # (input_dim, 1)
                sens = np.tile(sens, (1, cbSz))  # (input_dim, batch)
            
            # Ensure sens matches batch size
            if sens.shape[1] != cbSz:
                if sens.shape[1] == 1:
                    sens = np.tile(sens, (1, cbSz))
                else:
                    # Broadcast or repeat as needed
                    sens = sens[:, :cbSz] if sens.shape[1] > cbSz else np.tile(sens, (1, (cbSz + sens.shape[1] - 1) // sens.shape[1]))[:, :cbSz]
            
            # grad is not used in naive mode, but initialize it
            grad = 0
            
            # Initialize critVal if it doesn't exist
            if 'critVal' not in locals():
                critVal = np.zeros((1, cbSz), dtype=xi.dtype)
            
            # Get input split heuristic
            inputSplitHeuristic = options.get('nn', {}).get('input_split_heuristic', 'most-sensitive-input-radius')
            
            for i in range(nDims):
                # Compute the heuristic.
                # MATLAB: his = aux_computeHeuristic(inputSplitHeuristic, 0, xis - ris, xis + ris, ris, sens, grad, [],[],[],false,1);
                his = _aux_computeHeuristic(
                    inputSplitHeuristic,
                    0,  # the input has layer index 0
                    xis - ris,  # lower bound
                    xis + ris,  # upper bound
                    ris,  # approximation error
                    sens,  # sensitivity
                    grad,  # zonotope norm gradient
                    [], [], [], False, 1
                )
                
                # Split the input sets along one dimension.
                # MATLAB: [xis,ris,dimId] = aux_split(xis,ris,his,nSplits);
                xis, ris, dimId = _aux_split_with_dim(xis, ris, his, nSplits)
                # Append the split dimension.
                # MATLAB: dimIds = [repmat(dimId,1,nSplits); dimIds];
                dimIds = np.vstack([np.tile(dimId, (1, nSplits)), dimIds])
                # Replicate sensitivity and criticality value.
                # MATLAB: sens = repmat(sens,1,nSplits);
                sens = np.tile(sens, (1, nSplits))
                # MATLAB: grad = repmat(grad,1,nSplits);
                grad = np.tile(grad, (1, nSplits)) if not isinstance(grad, (int, float)) else np.tile(np.array([grad]), (1, nSplits))
                # MATLAB: critVal = repmat(critVal,1,nSplits);
                critVal = np.tile(critVal, (1, nSplits))
            
            # There is no neuron splitting.
            nrXis = np.zeros((0, xis.shape[1]), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
            
        elif refinement_method in ['zonotack', 'zonotack-layerwise']:
            # Zonotack refinement - full implementation matching MATLAB
            # Initialize number of splitted sets.
            newSplits = 1
            
            # Construct neuron-split constraints.
            if nNeur > 0 and nSplits > 1:
                # Create split constraints for neurons within the network.
                neuronSplitHeuristic = options.get('nn', {}).get('neuron_split_heuristic', 'least-unstable')
                An, bn, newNrXi, hn = _aux_neuronConstraints(nn, options, None,
                                                              neuronSplitHeuristic, nSplits, nNeur,
                                                              numInitGens, nrXi)
                # Compute number of new splits.
                newSplits = (nSplits ** nNeur) * newSplits
            else:
                # There are no general-split constraints.
                An = np.zeros((0, q, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
                bn = np.zeros((nSplits-1, 0, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
                newNrXi = -np.ones((0, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
                hn = -np.ones((0, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
            
            # Identify dummy splits.
            isDummySplit = np.all(np.isinf(newNrXi), axis=0) & np.any(np.isinf(newNrXi), axis=0)
            
            # Construct input split constraints.
            if nDims > 0 and nSplits > 1:
                # When not all input dimensions get an assigned generator
                # we have to restrict and reorder the dimensions.
                # Compute indices.
                from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
                permIdx = sub2ind((n0, cbSz),
                                  inputDimIdx.flatten('F'),  # Column-major flatten, 1-based
                                  repelem(np.arange(1, cbSz + 1), numInitGens, 1).flatten('F'))  # 1-based
                permIdx = permIdx.reshape(numInitGens, cbSz) - 1  # Convert to 0-based for indexing
                # Permute the input and radius.
                xi_ = xi[permIdx]  # (numInitGens, cbSz)
                ri_ = ri[permIdx]  # (numInitGens, cbSz)
                # Permute the sensitivity.
                if 'sens' not in locals() or sens is None:
                    # Compute sensitivity for splitting
                    S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
                    # Compute sens from S
                    if S.ndim == 3:
                        S_abs = np.abs(S)
                        S_max = np.maximum(S_abs, 1e-6)
                        sens_max = np.max(S_max, axis=0)  # (input_dim, batch)
                        sens = sens_max.T  # (batch, input_dim)
                    else:
                        sens = np.ones((cbSz, n0), dtype=xi.dtype)
                
                # Ensure sens has correct shape (input_dim, batch)
                if sens.ndim == 2 and sens.shape[0] == cbSz:
                    sens = sens.T  # (input_dim, batch)
                elif sens.ndim == 1:
                    sens = sens.reshape(-1, 1)
                    sens = np.tile(sens, (1, cbSz))
                
                sens_ = sens[permIdx] if sens.shape[0] == n0 else sens  # (numInitGens, cbSz) or keep original
                
                # Compute gradient for input split heuristic
                # Check if gradient should be computed
                inputSplitHeuristic = options.get('nn', {}).get('input_split_heuristic', 'most-sensitive-input-radius')
                neuronSplitHeuristic = options.get('nn', {}).get('neuron_split_heuristic', 'least-unstable')
                reluConstrHeuristic = options.get('nn', {}).get('relu_constr_heuristic', 'least-unstable')
                storeGradients = (inputSplitHeuristic == 'zono-norm-gradient' or
                                 neuronSplitHeuristic == 'zono-norm-gradient' or
                                 reluConstrHeuristic == 'least-unstable-gradient')
                if inputSplitHeuristic == 'zono-norm-gradient' and 'Gyi' in locals() and Gyi is not None:
                    # Compute gradient using aux_updateGradients
                    grad_full = _aux_updateGradients(nn, options, idxLayer, Gyi, A, b, storeGradients)
                    if grad_full is not None:
                        # Extract gradient for input dimensions
                        # MATLAB: dimGenIdx = reshape(sub2ind(size(grad), inputDimIdx, repmat((1:numInitGens)',1,cbSz), repelem(1:cbSz,numInitGens,1)),[numInitGens cbSz]);
                        from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
                        dimGenIdx = sub2ind(grad_full.shape,
                                           inputDimIdx.flatten('F'),  # 1-based
                                           np.tile(np.arange(1, numInitGens + 1), (1, cbSz)).flatten('F'),  # 1-based
                                           repelem(np.arange(1, cbSz + 1), numInitGens, 1).flatten('F'))  # 1-based
                        dimGenIdx = dimGenIdx - 1  # Convert to 0-based
                        dimGenIdx = dimGenIdx.reshape(numInitGens, cbSz)
                        # MATLAB: grad = reshape(grad(dimGenIdx),[numInitGens cbSz]);
                        grad = grad_full.flatten()[dimGenIdx].reshape(numInitGens, cbSz)
                    else:
                        grad = 0
                else:
                    grad = 0
                
                # Compute the heuristic.
                hi = _aux_computeHeuristic(
                    inputSplitHeuristic,
                    0,  # the input has layer index 0
                    xi_ - ri_,  # lower bound
                    xi_ + ri_,  # upper bound
                    ri_,  # approximation error
                    sens_,  # sensitivity
                    grad,  # zonotope norm gradient
                    [], [], [], False, 1
                )
                
                # Compute input-split constraints.
                Ai, bi, dimIds, hi = _aux_dimSplitConstraints(hi, nSplits, nDims)
                
                # Update number of new splits.
                newSplits = (nSplits ** nDims) * newSplits
            else:
                # There are no input-split constraints.
                Ai = np.zeros((0, q, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
                bi = np.zeros((nSplits-1, 0, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
                dimIds = np.full((0, cbSz), np.nan, dtype=xi.dtype)
                hi = -np.ones((0, cbSz), dtype=Gyi.dtype if 'Gyi' in locals() else xi.dtype)
            
            # Pad offsets if there are different number of offsets in general
            # split and input split constraints.
            if bn.shape[1] != bi.shape[1]:
                max_offsets = max(bn.shape[1], bi.shape[1])
                if bn.shape[1] < max_offsets:
                    pad_size = max_offsets - bn.shape[1]
                    bn_pad = np.full((bn.shape[0], pad_size, bn.shape[2]), np.nan, dtype=bn.dtype)
                    bn = np.concatenate([bn, bn_pad], axis=1)
                if bi.shape[1] < max_offsets:
                    pad_size = max_offsets - bi.shape[1]
                    bi_pad = np.full((bi.shape[0], pad_size, bi.shape[2]), np.nan, dtype=bi.dtype)
                    bi = np.concatenate([bi, bi_pad], axis=1)
            
            # Append zeros for generators.
            An_ = np.concatenate([An, np.zeros((An.shape[0], q - An.shape[1], cbSz), dtype=An.dtype)], axis=1) if An.shape[1] < q else An
            Ai_ = np.concatenate([Ai, np.zeros((Ai.shape[0], q - Ai.shape[1], cbSz), dtype=Ai.dtype)], axis=1) if Ai.shape[1] < q else Ai
            # Concatenate input and neuron splits.
            As = np.concatenate([An_, Ai_], axis=0)  # (nNeur + nDims, q, cbSz)
            bs = np.concatenate([bn, bi], axis=0)  # (nSplits-1, nNeur + nDims, cbSz)
            # Pad the neuron split indices with NaN for the input dimensions.
            newNrXi = np.concatenate([newNrXi, np.full((dimIds.shape[0], cbSz), np.nan, dtype=newNrXi.dtype)], axis=0)
            
            # Handle input_xor_neuron_splitting option
            if options.get('nn', {}).get('input_xor_neuron_splitting', False) and nNeur > 0 and nDims > 0:
                # We only allow input xor neuron splitting.
                # Select the minimum of split dimensions.
                nDims_ = min(nNeur, nDims)
                
                # Only pick either neuron or input split.
                h_combined = np.concatenate([hn, hi], axis=0)  # (nNeur + nDims, cbSz)
                sort_idx = np.argsort(h_combined, axis=0)[::-1]  # Sort descending
                
                # Obtain the indices for the relevant constraints.
                from ..layers.linear.nnGeneratorReductionLayer import sub2ind
                sIdx = sub2ind((As.shape[0], As.shape[2]),
                               sort_idx[:nDims_, :].flatten('F'),  # 1-based
                               np.tile(np.arange(1, cbSz + 1), nDims_))  # 1-based
                sIdx = sIdx - 1  # Convert to 0-based
                
                # Extract the corresponding constraint.
                As_flat = As.reshape(As.shape[0], q, -1)
                As = As_flat[sIdx, :, :].reshape(nDims_, q, cbSz)
                bs_flat = bs.reshape(bs.shape[0], bs.shape[1], -1)
                bs = bs_flat[sIdx, :, :].reshape(nSplits-1, nDims_, cbSz)
                # Extract the corresponding neuron indices.
                newNrXi = newNrXi[sort_idx[:nDims_, :], np.arange(cbSz)]
                
                # Update number of new splits.
                newSplits = nSplits ** nDims_
            
            # Refine the input set based on the output specification.
            reluConstrHeuristic = options.get('nn', {}).get('relu_constraint_heuristic', 'most-sensitive-input-radius')
            l, u, nrXis = _aux_refineInputSet(nn, options, nReLU > 0 or nNeur > 0,
                                               cxi, Gxi, yi, Gyi, A, b, numUnionConst, safeSet,
                                               As, bs, newNrXi, nrXi, reluConstrHeuristic)
            
            # We enclose all unsafe outputs; therefore, a set is verified if it is empty.
            isVerified = np.any(np.isnan(l), axis=0) | np.any(np.isnan(u), axis=0)
            # Remove the verified sets...
            l = l[:, ~isVerified]
            u = u[:, ~isVerified]
            nrXis = nrXis[:, ~isVerified] if nrXis.size > 0 else nrXis
            
            # Compute center and radius of refined sets.
            xis = 0.5 * (u + l)  # (n0, remaining_batch)
            ris = 0.5 * (u - l)  # (n0, remaining_batch)
            
            # Identify which sets were refined to just being a point.
            isPoint = np.all(ris == 0, axis=0)
            # Check the specification for the points.
            if np.any(~isPoint):
                # Evaluate points that are not just points
                xis_nonpoint = xis[:, ~isPoint]
                # Check specification using aux_checkPoints logic
                yi_points = nn.evaluate_(xis_nonpoint, options, idxLayer)
                ld_yi_points = A @ yi_points
                critValPerConstr_points = ld_yi_points - b
                if safeSet:
                    falsified_points = np.any(ld_yi_points > b, axis=0)
                    critVal_points = np.min(-critValPerConstr_points, axis=0)
                else:
                    falsified_points = np.all(ld_yi_points <= b, axis=0)
                    critVal_points = np.max(critValPerConstr_points, axis=0)
                
                if np.any(falsified_points):
                    # Found a counterexample in point sets.
                    res = 'COUNTEREXAMPLE'
                    id_ = np.where(falsified_points)[0][0]
                    x_ = xis_nonpoint[:, id_].reshape(-1, 1)
                    nn.castWeights(np.float32)
                    y_ = nn.evaluate_(x_, options, idxLayer)
                    break
                else:
                    # Remove the point sets...
                    xis = xis_nonpoint
                    ris = ris[:, ~isPoint]
                    nrXis = nrXis[:, ~isPoint] if nrXis.size > 0 else nrXis
                    critVal = critVal_points
            else:
                critVal = np.zeros((1, 0), dtype=xis.dtype)
            
            # All removed subproblems are verified, i.e., empty set or point sets.
            numVerified = np.sum(isVerified) + np.sum(isPoint)
            # We have to subtract the number of dummy splits.
            numVerified = numVerified - np.sum(isDummySplit)
            verifiedPatches += numVerified
            
        else:
            raise ValueError(f"Invalid refinement_method: {refinement_method}. Must be one of ['naive', 'zonotack', 'zonotack-layerwise']")
        
        # Order remaining sets by their criticality.
        # MATLAB: [~,idx] = sort(critVal.*1./max(ris,[],1),'ascend');
        if 'critVal' in locals() and critVal.size > 0:
            ris_max = np.max(ris, axis=0)  # (batch,)
            # Avoid division by zero
            ris_max = np.maximum(ris_max, 1e-10)
            sort_key = critVal.flatten() / ris_max  # (batch,)
            idx = np.argsort(sort_key)  # Ascending order
            # Order sets.
            xis = xis[:, idx]
            ris = ris[:, idx]
            if nrXis.size > 0:
                nrXis = nrXis[:, idx]
        
        # Add new splits to the queue.
        xs = np.hstack([xis, xs])
        rs = np.hstack([ris, rs])
        
        totalNumSplits += xis.shape[1]
        # All remaining patches were unknown, so they were all split (none were verified in this iteration)
        # verifiedPatches was already updated earlier when we computed unknown
        
        # To save memory, we clear all variables that are no longer used.
        # MATLAB: batchVars = {'xi','ri','nrXi','S_','S','sens','cxi','Gxi','yic','yid','Gyi','ld_yi','ld_Gyi'};
        # MATLAB: clear(batchVars{:});
        # In Python, we rely on garbage collection, but we can explicitly delete variables
        # Note: dyi and dri were removed - we now use ld_yi and ld_ri instead
        # All these variables should exist at this point in the code
        del xi, ri, yi, Gyi, cxi, Gxi, yic, yid, ld_yi, ld_Gyi, ld_ri, ld_Gyi_err, zi, critVal, critValPerConstr, checkSpecs, xis, ris
    
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


def _aux_split_with_dim(xi: np.ndarray, ri: np.ndarray, his: np.ndarray, nSplits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the input for verification and return the dimension ID (MATLAB aux_split equivalent for naive refinement)
    
    MATLAB signature:
    function [xis,ris,dimId] = aux_split(xis,ris,his,nSplits)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    n, bs = xi.shape
    # Find the input dimension with the largest heuristic.
    # MATLAB: [~,sortDims] = sort(abs(his),1,'descend');
    sortDims = np.argsort(np.abs(his), axis=0)[::-1]  # Sort descending along axis=0 (rows)
    # MATLAB: dimId = sortDims(1,:);
    dimId = sortDims[0, :]  # Shape: (batch,), 1-based dimension indices
    
    # MATLAB: splitsIdx = repmat(1:nSplits,1,bs);
    splitsIdx = np.tile(np.arange(1, nSplits + 1), bs)  # 1-based like MATLAB: (nSplits*bs,)
    # MATLAB: bsIdx = repelem((1:bs)',nSplits);
    bsIdx = repelem(np.arange(1, bs + 1), nSplits)  # 1-based: (bs*nSplits,)
    
    # MATLAB: linIdx = sub2ind([n bs nSplits], repelem(dimId,nSplits),bsIdx(:)',splitsIdx(:)');
    dim_repeated = repelem(dimId, nSplits)  # Shape: (batch*nSplits,), 1-based
    linIdx = sub2ind((n, bs, nSplits), dim_repeated, bsIdx, splitsIdx)  # 1-based linear indices
    
    # 2. Split the selected dimension.
    xi_ = xi.copy()
    ri_ = ri.copy()
    # Shift to the lower bound.
    dimIdx = sub2ind((n, bs), dimId, np.arange(1, bs + 1))  # 1-based linear indices
    dimIdx_0based = dimIdx - 1
    xi_flat = xi_.flatten()
    ri_flat = ri_.flatten()
    xi_flat[dimIdx_0based] = xi_flat[dimIdx_0based] - ri_flat[dimIdx_0based]
    xi_ = xi_flat.reshape(n, bs)
    ri_flat = ri_.flatten()
    ri_flat[dimIdx_0based] = ri_flat[dimIdx_0based] / nSplits
    ri_ = ri_flat.reshape(n, bs)
    
    # MATLAB: xis = repmat(xi_,1,1,nSplits);
    xis = np.tile(xi_.reshape(n, bs, 1), (1, 1, nSplits))  # Shape: (n, bs, nSplits)
    # MATLAB: ris = repmat(ri_,1,1,nSplits);
    ris = np.tile(ri_.reshape(n, bs, 1), (1, 1, nSplits))  # Shape: (n, bs, nSplits)
    
    # MATLAB: xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    linIdx_0based = linIdx - 1
    xis_flat = xis.flatten()
    ris_flat = ris.flatten()
    splitsIdx_0based = splitsIdx - 1
    xis_flat[linIdx_0based] = xis_flat[linIdx_0based] + (2 * splitsIdx_0based - 1) * ris_flat[linIdx_0based]
    xis = xis_flat.reshape(n, bs, nSplits)
    
    # MATLAB: xis = xis(:,:); ris = ris(:,:);
    xis = xis.reshape(n, -1)
    ris = ris.reshape(n, -1)
    
    return xis, ris, dimId


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


# ============================================================================
# Helper functions for zonotack refinement
# ============================================================================

def _aux_enumerateLayers(layers: List, ancIdx: List[int]) -> Tuple[List, List[int], List[int], List[int]]:
    """
    Recursively enumerate layers of a neural network, handling composite layers.
    
    MATLAB: function [layersEnum,ancIdx,predIdx,succIdx] = aux_enumerateLayers(layers,ancIdx)
    """
    # Initialize result.
    layersEnum = []
    ancIdxIds = []  # Indices into ancIdx.
    predIdx = []  # Indices to predecessors.
    succIdx = []  # Indices to successors.
    
    # Recursive iteration over the layers.
    for i in range(len(layers)):
        # Obtain i-th layer.
        layeri = layers[i]
        
        # Check if layer is composite (has nested layers)
        # In Python, we check if layer has a 'layers' attribute that is a list
        if hasattr(layeri, 'layers') and isinstance(getattr(layeri, 'layers', None), list):
            # Store index for predecessor layer.
            predi = len(layersEnum)
            # Enumerate layers of computation paths.
            for j in range(len(layeri.layers)):
                # Obtain the top-level layers of the j-th computation path.
                layersij = layeri.layers[j]
                if not layersij or len(layersij) == 0:
                    # This is a residual connection. There are no layers here.
                    continue
                # All layers in this computation path have the same ancestor.
                ancIdxIdij = [i] * len(layersij)
                # Enumerate layers of the j-th computation path.
                layersijEnum, ancIdxIdij, predIdxij, succIdxij = _aux_enumerateLayers(layersij, ancIdxIdij)
                # Offset the predecessor and successor indices.
                predIdxij = [p + len(layersEnum) for p in predIdxij]
                if len(predIdxij) > 0:
                    predIdxij[0] = predi
                succIdxij = [s + len(layersEnum) if not np.isnan(s) else np.nan for s in succIdxij]
                if len(succIdxij) > 0:
                    succIdxij[-1] = np.nan
                # Append layers.
                layersEnum.extend(layersijEnum)
                # Replicate ancestor index for all new layers.
                ancIdxIds.extend(ancIdxIdij)
                # Append the predecessor and successor indices.
                predIdx.extend(predIdxij)
                succIdx.extend(succIdxij)
            # Store index of the successor layer.
            succIdx = [len(layersEnum) + 1 if np.isnan(s) else s for s in succIdx]
        else:
            # Append layer.
            layersEnum.append(layeri)
            ancIdxIds.append(i)
            predIdx.append(len(layersEnum) - 1)
            succIdx.append(len(layersEnum) + 1)
    
    # Construct ancestor indices.
    ancIdx_result = [ancIdx[i] for i in ancIdxIds]
    
    return layersEnum, ancIdx_result, predIdx, succIdx


def _enumerateLayers(nn: 'NeuralNetwork') -> Tuple[List, List[int], List[int], List[int]]:
    """
    Enumerate the layers of a neural network.
    
    MATLAB: function [layersEnum,ancIdx,predIdx,succIdx] = enumerateLayers(nn)
    """
    # Obtain the layers of the neural network.
    layers = nn.layers
    # Initialize ancestor indices.
    ancIdx = list(range(len(layers)))
    
    # Enumerate all layers of the neural network.
    layersEnum, ancIdx_result, predIdx, succIdx = _aux_enumerateLayers(layers, ancIdx)
    
    return layersEnum, ancIdx_result, predIdx, succIdx

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
                          layerDiscount: float = 1.0) -> np.ndarray:
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
            h = np.where(wasSplit, h, -np.inf)
            if sim is not None and sim.size > 0:
                # Specify a tolerance for similarity.
                tol = 1e-3
                # Reduce the heuristic based on the similarity to already split neurons.
                simSplit = np.any((sim > 1 - tol) & wasSplit.reshape(1, -1, 1), axis=1)
                h = np.where(simSplit, -np.inf, h)
    
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
    dimIds = sortDims[:nDims, :]  # (nDims, batch) - 0-based indices
    hi = hi_sorted[:nDims, :]  # (nDims, batch)
    
    # Convert to 1-based for sub2ind (MATLAB uses 1-based)
    dimIds_1based = dimIds + 1
    
    # Compute dimension indices.
    dimIdx = sub2ind((nDims, n, bSz),
                     repelem(np.arange(1, nDims + 1), 1, bSz),  # 1-based
                     dimIds_1based.flatten('F'),  # Column-major flatten, 1-based
                     repelem(np.arange(1, bSz + 1), nDims, 1).flatten('F'))  # 1-based
    
    # 2. Construct the constraints.
    Ai = np.zeros((nDims, n, bSz), dtype=hi.dtype)
    # Set non-zero entries
    dimIdx_0based = dimIdx - 1  # Convert to 0-based for Python indexing
    Ai_flat = Ai.flatten()
    Ai_flat[dimIdx_0based] = 1
    Ai = Ai_flat.reshape(nDims, n, bSz)
    
    # Specify offsets: repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz)
    offsets = -1 + np.arange(1, nSplits) * (2.0 / nSplits)  # (nSplits-1,)
    bi = np.tile(offsets.reshape(-1, 1, 1), (1, nDims, bSz))  # (nSplits-1, nDims, bSz)
    bi = np.transpose(bi, (1, 0, 2))  # (nDims, nSplits-1, bSz)
    
    return Ai, bi, dimIds, hi


def _aux_convertSplitConstraints(As: np.ndarray, bs: np.ndarray, nrXis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Consider all combinations between the given constraints, A*x <= b.
    
    MATLAB: function [A,b,newSplits,constNrIdx] = aux_convertSplitConstraints(As,bs,nrXis)
    """
    from ..layers.linear.nnGeneratorReductionLayer import repelem
    
    if As.size > 0:
        # Obtain the number of split-constraints.
        ps, q, bSz = As.shape
        # Obtain the number of pieces.
        _, pcs, _ = bs.shape
        # Compute number of new splits.
        newSplits = (pcs + 1) ** ps
        
        # We flip the signs of the constraints to realize splitting; -1
        # represents a lower bound, i.e., -A*x > -b, whereas 1 represents
        # an upper bound, i.e., A*x <= b.
        constrSign = np.array([-1, 1])
        
        # Duplicate each halfspace for a lower and an upper bound.
        As_ = repelem(As, 2, 1, 1)  # (2*ps, q, bSz)
        # Duplicate offsets for lower and upper bound.
        bs_ = repelem(bs, 1, 2, 1)  # (ps, 2*pcs, bSz)
        # Duplicate the indices for the split neurons.
        constNrIdx = repelem(nrXis, 2, 1)  # (2*ps, bSz)
        
        # Scale the constraints; -1 for upper bound and 1 for lower bound.
        As_ = np.tile(constrSign.reshape(-1, 1, 1), (ps, 1, 1)) * As_  # (2*ps, q, bSz)
        
        # Duplicate the constraint for the new splits.
        # MATLAB: A_ = permute(repelem(As_,1,1,1,newSplits),[2 1 4 3]);
        As_4d = np.tile(As_.reshape(2*ps, q, bSz, 1), (1, 1, 1, newSplits))  # (2*ps, q, bSz, newSplits)
        A_ = np.transpose(As_4d, (1, 0, 3, 2))  # (q, 2*ps, newSplits, bSz)
        
        # Mark unused bounds by NaN.
        bs_nan_lower = np.full((ps, 1, bSz), np.nan, dtype=bs.dtype)
        bs_nan_upper = np.full((ps, 1, bSz), np.nan, dtype=bs.dtype)
        bs_ = np.concatenate([bs_nan_lower, bs_, bs_nan_upper], axis=1)  # (ps, 2*pcs+2, bSz)
        
        # Scale the offsets; -1 for upper bound and 1 for lower bound.
        bs_ = np.tile(constrSign.reshape(1, -1, 1), (ps, pcs+1, 1)) * bs_  # (ps, 2*pcs+2, bSz)
        
        # Reshape and combine the lower and upper bounds.
        bs_reshaped = bs_.reshape(2, pcs+1, ps, bSz)
        bs_ = bs_reshaped.transpose(2, 0, 1, 3).reshape(2*ps, pcs+1, bSz)
        
        # Extend the offsets.
        b_ = np.concatenate([bs_, np.zeros((2*ps, newSplits - (pcs+1), bSz), dtype=bs.dtype)], axis=1)
        
        # Compute all combinations of the splits.
        idx = pcs + 1
        for i in range(ps - 1):
            # Increase the index.
            idx_ = idx * (pcs + 1)
            # Repeat the current combined splits.
            b_[:2*(i+1), :idx_, :] = np.tile(b_[:2*(i+1), :idx, :], (1, pcs+1, 1))
            # Repeat the elements of the next split and append them.
            next_slice = b_[2*i:2*(i+1), :(pcs+1), :]
            b_[2*i:2*(i+1), :idx_, :] = repelem(next_slice, 1, (pcs+1)**i, 1)
            # Update the index of the combined splits.
            idx = idx_
        
        # Compute the neuron indices of each constraint.
        # Scale the indices; -1 for upper bound and 1 for lower bound.
        constNrIdx = np.tile(constrSign.reshape(-1, 1), (ps, 1)) * constNrIdx  # (2*ps, bSz)
        # Duplicate the indices for the new splits.
        constNrIdx = repelem(constNrIdx, 1, newSplits)  # (2*ps, bSz*newSplits)
        
        # Find all unused constraints.
        nanIdx = np.isnan(b_)
        # Set all not needed constraints to zero.
        A_flat = A_.reshape(q, -1)
        A_flat[:, nanIdx.flatten()] = 0
        A_ = A_flat.reshape(q, 2*ps, newSplits, bSz)
        b_[nanIdx] = 0
        # Mark all unused constraints.
        constNrIdx_flat = constNrIdx.flatten()
        constNrIdx_flat[nanIdx.flatten()] = np.nan
        constNrIdx = constNrIdx_flat.reshape(2*ps, bSz*newSplits)
        
        # Reshape the constraint matrix and offset.
        A = A_.transpose(1, 0, 3, 2).reshape(2*ps, q, newSplits*bSz)
        b = b_.reshape(2*ps, newSplits*bSz)
    else:
        # There are no additional constraints.
        newSplits = 1
        A = np.zeros((0, As.shape[1] if As.size > 0 else 0, As.shape[2] if As.size > 0 else 0), dtype=As.dtype if As.size > 0 else np.float32)
        b = np.zeros((0, bs.shape[1] if bs.size > 0 else 0), dtype=bs.dtype if bs.size > 0 else np.float32)
        constNrIdx = np.zeros((0, nrXis.shape[1] if nrXis.size > 0 else 0), dtype=nrXis.dtype if nrXis.size > 0 else np.float32)
    
    return A, b, newSplits, constNrIdx


def _aux_boundsOfBoundedPolytope(A: np.ndarray, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounds [bl,bu] of a bounded polytope P:
    Given P=(A,b) \cap [-1,1], compute its bounds.
    
    MATLAB: function [bl,bu] = aux_boundsOfBoundedPolytope(A,b,options)
    """
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Specify a numerical tolerance to avoid numerical instability.
    tol = 1e-8
    
    # Initialize bounds of the factors.
    q = A.shape[1]
    bSz = A.shape[2] if A.ndim > 2 else 1
    bl = -np.ones((q, bSz), dtype=A.dtype)
    bu = np.ones((q, bSz), dtype=A.dtype)
    
    if not options.get('nn', {}).get('exact_conzonotope_bounds', False):
        # Efficient approximation by isolating the i-th variable.
        # Specify maximum number of iterations.
        maxIter = options.get('nn', {}).get('polytope_bound_approx_max_iter', 10)
        
        # Permute the dimension of the constraints for easier handling.
        A_ = np.transpose(A, (1, 0, 2))  # (q, p, bSz)
        b_ = np.transpose(b, (2, 0, 1)) if b.ndim > 1 else b.reshape(1, -1, 1)  # (1, p, bSz)
        # Reshape factor bounds for easier multiplication.
        bl_ = bl.reshape(q, 1, bSz)  # (q, 1, bSz)
        bu_ = bu.reshape(q, 1, bSz)  # (q, 1, bSz)
        
        # Extract a mask for the sign of the coefficient.
        nMsk = (A_ < 0)  # (q, p, bSz)
        pMsk = (A_ > 0)  # (q, p, bSz)
        # Decompose the matrix into positive and negative entries.
        An = A_ * nMsk  # (q, p, bSz)
        Ap = A_ * pMsk  # (q, p, bSz)
        
        # Do summation with matrix multiplication: sum all but the i-th entry.
        sM = 1 - np.eye(q, dtype=A.dtype)  # (q, q)
        
        # Initialize iteration counter.
        iter = 1
        tighterBnds = True
        while tighterBnds and iter <= maxIter:
            # Scale the matrix entries with the current bounds.
            ABnd = Ap * bl_ + An * bu_  # (q, p, bSz)
            # Isolate the i-th variable of the j-th constraint.
            sABnd = pagemtimes(sM, 'none', ABnd, 'none')  # (q, p, bSz)
            # Compute right-hand side of the inequalities.
            # Avoid division by zero
            A_safe = np.where(np.abs(A_) > tol, A_, np.sign(A_) * tol)
            rh = np.clip((b_ - sABnd) / A_safe, bl_, bu_)  # (q, p, bSz)
            # Update the bounds.
            bl_ = np.max(np.where(nMsk, rh, bl_), axis=1, keepdims=True)  # (q, 1, bSz)
            bu_ = np.min(np.where(pMsk, rh, bu_), axis=1, keepdims=True)  # (q, 1, bSz)
            
            # Check if the bounds could be tightened.
            bl_2d = bl_.squeeze(1)  # (q, bSz)
            bu_2d = bu_.squeeze(1)  # (q, bSz)
            tighterBnds = np.any((bl + tol < bl_2d) | (bu_2d < bu - tol)) & np.all(bl_2d <= bu_2d)
            bl = bl_2d
            bu = bu_2d
            # Increment iteration counter.
            iter += 1
    else:
        # Slow implementation with exact bounds for validation.
        # Obtain the batch size.
        p, q, bSz = A.shape
        
        for i in range(bSz):
            # Obtain parameters of the i-th batch entry.
            Ai = A[:, :, i].astype(np.float64)
            bi = b[:, i].astype(np.float64)
            if np.any(np.isnan(Ai)) or np.any(np.isnan(bi)):
                # The given set is already marked as empty.
                bl[:, i] = np.nan
                bu[:, i] = np.nan
            else:
                # Loop over the dimensions.
                for j in range(q):
                    # Construct linear program.
                    # MATLAB: prob = struct('Aineq',Ai,'bineq',bi, 'lb',-ones(q,1),'ub',ones(q,1));
                    # For Python, we'll use scipy.optimize.linprog if available, otherwise use approximation
                    try:
                        from scipy.optimize import linprog
                        # Find the lower bound for the j-th dimension.
                        c = np.zeros(q)
                        c[j] = 1
                        # Solve the linear program: minimize c^T x subject to Ai*x <= bi, -1 <= x <= 1
                        result_lower = linprog(c, A_ub=Ai, b_ub=bi, bounds=[(-1, 1)] * q, method='highs')
                        # Find the upper bound for the j-th dimension.
                        c[j] = -1
                        result_upper = linprog(c, A_ub=Ai, b_ub=bi, bounds=[(-1, 1)] * q, method='highs')
                        if result_lower.success and result_upper.success:
                            # Solutions found; assign values.
                            bl[j, i] = result_lower.fun
                            bu[j, i] = -result_upper.fun
                        else:
                            # No solution; the polytope is empty.
                            bl[j, i] = np.nan
                            bu[j, i] = np.nan
                            continue
                    except ImportError:
                        # scipy not available, use approximation (fallback to iterative method)
                        # This is a simplified fallback - use the iterative method instead
                        bl[j, i] = -1.0
                        bu[j, i] = 1.0
                # Check if the sets are empty.
                if np.any(np.isnan(bl[:, i])) or np.any(np.isnan(bu[:, i])):
                    # No solution; the polytope is empty.
                    bl[:, i] = np.nan
                    bu[:, i] = np.nan
    
    return bl, bu


def _aux_boundsOfConZonotope(cZs: Dict[str, np.ndarray], numUnionConst: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bounds of constraint zonotope.
    
    MATLAB: function [l,u,bl,bu] = aux_boundsOfConZonotope(cZs,numUnionConst,options)
    """
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Extract parameters of the constraint zonotope.
    c = cZs['c']
    G = cZs['G']
    r = cZs.get('r', np.zeros((c.shape[0], 1, G.shape[2]) if G.ndim > 2 else (c.shape[0], 1)))
    A = cZs.get('A', np.array([]))
    b = cZs.get('b', np.array([]))
    
    # Obtain number of dimensions, generators, and batch size.
    n, q, bSz = G.shape
    
    if A.size == 0:
        # There are no constraints. Just compute the bounds of the zonotope.
        r = np.sum(np.abs(G), axis=1).reshape(n, bSz)
        l = c - r
        u = c + r
        # The bounds of the hypercube are just -1 and 1;
        bl = -np.ones((q, bSz), dtype=G.dtype)
        bu = np.ones((q, bSz), dtype=G.dtype)
        return l, u, bl, bu
    
    # Specify indices of intersection constraints.
    intConIdx = np.arange(numUnionConst, A.shape[0])
    
    if options.get('nn', {}).get('batch_union_conzonotope_bounds', False):
        # The safe set is the union of all constraints.
        # Move union constraints into the batch.
        Au = A[:numUnionConst, :, :].transpose(2, 1, 0).reshape(1, q, bSz*numUnionConst)
        bu_vals = b[:numUnionConst, :].transpose(1, 0).reshape(1, bSz*numUnionConst)
        # Replicate intersection constraints.
        Ai = np.tile(A[intConIdx, :, :], (1, 1, numUnionConst))
        bi = np.tile(b[intConIdx, :], (1, numUnionConst))
        # Append intersection constraints.
        A_combined = np.concatenate([Au, Ai], axis=0)
        b_combined = np.concatenate([bu_vals, bi], axis=0)
        
        # Approximate the bounds of the hypercube (bounded polytope).
        bl, bu = _aux_boundsOfBoundedPolytope(A_combined, b_combined, options)
        
        if numUnionConst > 1:
            # Unify sets if a safe set is specified.
            bl = bl.reshape(q, bSz, numUnionConst)
            bu = bu.reshape(q, bSz, numUnionConst)
            bl = np.min(bl, axis=2)
            bu = np.max(bu, axis=2)
        else:
            # There are no constraints to unify.
            bl = bl.reshape(q, bSz)
            bu = bu.reshape(q, bSz)
    else:
        bl = None
        bu = None
        # Loop over the union constraints.
        for k in range(numUnionConst):
            # Use the k-th union constraint and all intersection constraints.
            Ak = np.concatenate([A[k:k+1, :, :], A[intConIdx, :, :]], axis=0)
            bk = np.concatenate([b[k:k+1, :], b[intConIdx, :]], axis=0)
            # Approximate the bounds of the hypercube.
            blk, buk = _aux_boundsOfBoundedPolytope(Ak, bk, options)
            # Unify constraints.
            if bl is None:
                bl = blk
                bu = buk
            else:
                bl = np.minimum(bl, blk)
                bu = np.maximum(bu, buk)
    
    # Map bounds of the factors to bounds of the constraint zonotope.
    # We use interval arithmetic for that.
    bc = 0.5 * (bu + bl).transpose(1, 0)  # (q, bSz) -> (bSz, q) -> transpose to (q, 1, bSz)?
    br = 0.5 * (bu - bl).transpose(1, 0)
    
    # Ensure correct shape for pagemtimes
    if bc.ndim == 2:
        bc = bc.reshape(q, 1, bSz)
    if br.ndim == 2:
        br = br.reshape(q, 1, bSz)
    
    # Map bounds of the factors to bounds of the constraint zonotope.
    c_offset = pagemtimes(G, 'none', bc, 'none').reshape(n, bSz)
    r_offset = pagemtimes(np.abs(G), 'none', br, 'none').reshape(n, bSz)
    c = c + c_offset
    r = r.reshape(n, bSz) + r_offset
    l = c - r
    u = c + r
    
    # Identify empty sets.
    isEmpty = np.any(bl > bu, axis=0)
    l[:, isEmpty] = np.nan
    u[:, isEmpty] = np.nan
    bl[:, isEmpty] = 0
    bu[:, isEmpty] = 0
    
    return l, u, bl, bu


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


def _aux_neuronConstraints(nn: 'NeuralNetwork', options: Dict[str, Any], idxLayer: Optional[List[int]],
                           heuristic: str, nSplits: int, nNeur: int, numInitGens: int,
                           prevNrXs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct neuron-split constraints.
    
    MATLAB: function [As,bs,constNrIdx,h] = aux_neuronConstraints(nn,options,idxLayer,heuristic,nSplits,nNeur,numInitGens,prevNrXs)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem, pagemtimes
    
    # Initialize constraints; we insert dummy splits to prevent splitting a dimension twice.
    As = np.zeros((0, nNeur, 1), dtype=prevNrXs.dtype)
    bs = np.zeros((nSplits-1, nNeur, 1), dtype=prevNrXs.dtype)
    q = 0  # Number of considered generators.
    # Initial heuristics.
    h = -np.ones((nNeur, 1), dtype=prevNrXs.dtype)
    # Initialize indices of neuron split.
    constNrIdx = np.full((nNeur, 1), np.inf, dtype=prevNrXs.dtype)
    
    # Enumerate the layers of the neural networks.
    # MATLAB: [layers,~,~,succIdx] = nn.enumerateLayers();
    layers, _, _, succIdx = _enumerateLayers(nn)
    
    if idxLayer is None or len(idxLayer) == 0:
        idxLayer = list(range(len(layers)))
    
    # Compute the indices of ReLU layers (activation layers).
    # MATLAB: idxLayer = idxLayer(arrayfun(@(i) isa(layers{i},'nnActivationLayer'),idxLayer));
    from ..layers.nonlinear.nnActivationLayer import nnActivationLayer
    idxLayer = [i for i in idxLayer if isinstance(layers[i], nnActivationLayer)]
    
    # Iterate through the layers and find max heuristics and propagate constraints.
    for i in idxLayer:
        # Obtain i-th layer.
        layeri = layers[i]
        
        # Obtain the input set and compute the bounds.
        # MATLAB: [li,ui,cil,ciu,Gi] = aux_computeBoundsOfInputSet(layeri,options,[],[],[],[],prevNrXs);
        # For now, we need to get bounds from layer's backprop store if available
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            if 'l' in store and 'u' in store:
                li = store['l']
                ui = store['u']
                # Get input center and generators from store
                if 'inc' in store and 'inG' in store:
                    cil = store['inc']
                    ciu = store['inc']  # For non-interval center, same as cil
                    Gi = store['inG']
                    if 'genIds' in store:
                        genIds = store['genIds']
                        if isinstance(genIds, (list, np.ndarray)):
                            Gi = Gi[:, genIds, :] if Gi.ndim == 3 else Gi
                else:
                    cil = None
                    ciu = None
                    Gi = None
            else:
                # No stored bounds, use default
                li = np.array([-np.inf])
                ui = np.array([np.inf])
                cil = None
                ciu = None
                Gi = None
        else:
            li = np.array([-np.inf])
            ui = np.array([np.inf])
            cil = None
            ciu = None
            Gi = None
        
        # Compute the center.
        if cil is not None and ciu is not None:
            ci = 0.5 * (ciu + cil)
        else:
            ci = np.zeros((1, 1), dtype=prevNrXs.dtype)
        
        # Obtain number of hidden neurons.
        if Gi is not None:
            nk, qi, bSz = Gi.shape
        else:
            # Default values
            nk = 1
            qi = 0
            bSz = prevNrXs.shape[1] if prevNrXs.ndim > 1 else 1
        
        # Obtain the indices of the neurons of the current layer.
        neuronIds = getattr(layeri, 'neuronIds', np.arange(nk))
        
        # Obtain the approximation errors.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            dl = store.get('dl', np.zeros((nk, bSz), dtype=prevNrXs.dtype))
            du = store.get('du', np.zeros((nk, bSz), dtype=prevNrXs.dtype))
        else:
            dl = np.zeros((nk, bSz), dtype=prevNrXs.dtype)
            du = np.zeros((nk, bSz), dtype=prevNrXs.dtype)
        
        # Compute center and radius of approximation errors.
        dr = 0.5 * (du - dl)
        
        # Obtain the sensitivity for heuristic.
        if hasattr(layeri, 'sensitivity') and layeri.sensitivity is not None:
            Si_ = np.maximum(np.abs(layeri.sensitivity), 1e-6)
            sens = np.reshape(np.max(Si_, axis=0), (nk, -1))
        else:
            sens = np.ones((nk, bSz), dtype=prevNrXs.dtype)
        
        if As.shape[2] < bSz:
            padBSz = bSz - As.shape[2]
            # Pad to the correct batch size.
            As = np.concatenate([As, np.zeros((q, nNeur, padBSz), dtype=As.dtype)], axis=2)
            bs = np.concatenate([bs, np.zeros((nSplits-1, nNeur, padBSz), dtype=bs.dtype)], axis=2)
            h = np.concatenate([h, -np.ones((nNeur, padBSz), dtype=h.dtype)], axis=1)
            constNrIdx = np.concatenate([constNrIdx, np.full((nNeur, padBSz), np.inf, dtype=constNrIdx.dtype)], axis=1)
        
        if q < qi:
            # Pad constraints with zeros.
            As = np.concatenate([As, np.zeros((qi - q, nNeur, bSz), dtype=As.dtype)], axis=0)
            # Update number of constraints.
            q = qi
        else:
            # Pad generators with zeros.
            if Gi is not None:
                Gi = np.concatenate([Gi, np.zeros((nk, qi - q, bSz), dtype=Gi.dtype)], axis=1)
        
        # Append new constraints.
        if Gi is not None:
            Asi = Gi  # (nk, qi, bSz)
            As = np.concatenate([As, np.transpose(Asi, (1, 0, 2))], axis=1)  # (q, nNeur + nk, bSz)
        
        # Compute split offsets based on split_position
        split_position = options.get('nn', {}).get('split_position', 'middle')
        if split_position == 'zero':
            # Split into #nSplits pieces around 0.
            nSplits_ = (nSplits - 1) // 2
            splitEnum = 1.0 / (nSplits_ + 1) * np.arange(1, (nSplits - 1) // 2 + 1)
            # bil = flip(splitEnum).*permute(li,[3 1 2]);
            bil = np.flip(splitEnum).reshape(-1, 1, 1) * li.reshape(1, nk, bSz)
            # biu = splitEnum.*permute(ui,[3 1 2]);
            biu = splitEnum.reshape(-1, 1, 1) * ui.reshape(1, nk, bSz)
            if nSplits % 2 == 0:
                # Include the center in the lower bounds.
                bil = np.concatenate([bil, np.zeros((1, nk, bSz), dtype=ci.dtype)], axis=0)
            # Combine the bounds.
            bsi = np.concatenate([bil, biu], axis=0)  # (nSplits-1, nk, bSz)
            # Subtract the center.
            bsi = bsi - ci.reshape(1, nk, bSz)
        elif split_position == 'middle':
            # Split into #nSplits pieces around the middle.
            splitEnum = np.linspace(-1, 1, nSplits + 1)[1:-1]  # Remove endpoints
            # bsi = splitEnum.*permute(ri,[3 1 2]);
            # ri is not directly available, use dr as approximation
            bsi = splitEnum.reshape(-1, 1, 1) * dr.reshape(1, nk, bSz)
        else:
            raise ValueError(f"Invalid split_position: {split_position}. Must be 'zero' or 'middle'")
        
        # Append the new offsets.
        bs = np.concatenate([bs, bsi], axis=1)  # (nSplits-1, nNeur + nk, bSz)
        
        # Obtain the gradient of the zonotope norm.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            grad = store.get('approx_error_gradients', 0)
        else:
            grad = 0
        
        # Obtain the neuron similarity based on the sensitivity.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            sim = store.get('similarity', None)
        else:
            sim = None
        
        # Compute the heuristic.
        hi = _aux_computeHeuristic(heuristic, i, li, ui, dr, sens, grad,
                                   sim, prevNrXs, neuronIds, True, 0.7)
        
        # Append heuristic and sort.
        h_combined = np.concatenate([h, hi], axis=0)  # (nNeur + nk, bSz)
        sort_idx = np.argsort(h_combined, axis=0)[::-1]  # Sort descending
        h = h_combined[sort_idx[:nNeur, :], np.arange(bSz)]  # (nNeur, bSz)
        
        # Obtain the indices for the relevant constraints.
        sIdx = sub2ind((As.shape[1], As.shape[2]),
                       sort_idx[:nNeur, :].flatten('F'),  # 1-based
                       np.tile(np.arange(1, bSz + 1), nNeur))  # 1-based
        sIdx = sIdx - 1  # Convert to 0-based
        
        # Extract constraints.
        As_flat = As.reshape(q, -1)
        As = As_flat[:, sIdx].reshape(q, nNeur, bSz)
        bs_flat = bs.reshape(nSplits-1, -1)
        bs = bs_flat[:, sIdx].reshape(nSplits-1, nNeur, bSz)
        
        # Update indices.
        constNrIdx_new = np.tile(neuronIds.reshape(-1, 1), (1, bSz))  # (nk, bSz)
        constNrIdx = np.concatenate([constNrIdx, constNrIdx_new], axis=0)  # (nNeur + nk, bSz)
        constNrIdx = constNrIdx[sort_idx[:nNeur, :], np.arange(bSz)]  # (nNeur, bSz)
    
    # Transpose constraint matrix.
    As = np.transpose(As, (1, 0, 2))  # (nNeur, q, bSz)
    bs = np.transpose(bs, (1, 0, 2))  # (nNeur, nSplits-1, bSz)
    
    if options.get('nn', {}).get('add_orth_neuron_splits', False):
        # Add the orthogonal neuron splits.
        # Obtain the number of constraints.
        p, _, _ = As.shape
        
        # Extract the most important input dimension.
        As_ = np.maximum(np.abs(As[:, :numInitGens, :]), 1e-6)  # (p, numInitGens, bSz)
        dimIds = np.argmax(As_, axis=1)  # (p, bSz)
        
        # 1. Generate unit vector along most important dimension.
        v = np.zeros((p, numInitGens, bSz), dtype=As_.dtype)
        dimIdx = sub2ind((p, numInitGens, bSz),
                         np.tile(np.arange(1, p + 1).reshape(-1, 1), (1, bSz)).flatten('F'),  # 1-based
                         dimIds.flatten('F'),  # 1-based (after adding 1)
                         np.tile(np.arange(1, bSz + 1), p))  # 1-based
        dimIdx = dimIdx - 1  # Convert to 0-based
        v_flat = v.flatten()
        v_flat[dimIdx] = 1
        v = v_flat.reshape(p, numInitGens, bSz)
        
        # 2. Make the vector orthogonal to the input dimensions of the split constraints.
        As_flat = As_.transpose(1, 0, 2).reshape(1, p * numInitGens * bSz)
        v_flat = v.transpose(1, 0, 2).reshape(1, p * numInitGens * bSz)
        
        # Compute projection
        proj = As_flat * pagemtimes(v_flat, 'none', As_flat, 'transpose') / \
               pagemtimes(As_flat, 'none', As_flat, 'transpose')
        vOrth = (v_flat - proj).reshape(numInitGens, p, bSz).transpose(1, 0, 2)
        
        # 3. Normalize the orthogonal vector and embed into the full space.
        AsOrth = As.copy()
        vOrthNorm = np.sqrt(np.sum(np.sum(vOrth**2, axis=1), axis=1))  # (p, bSz)
        AsOrth[:, :numInitGens, :] = vOrth / vOrthNorm.reshape(p, 1, bSz)
        
        # 4. Append the orthogonal constraints.
        As = np.concatenate([As, AsOrth], axis=0)
        bs = np.tile(bs, (2, 1, 1))
        
        # Append NaN for the orthogonal constraints.
        constNrIdx = np.concatenate([constNrIdx, np.full(constNrIdx.shape, np.nan, dtype=constNrIdx.dtype)], axis=0)
        # Append -1 for the orthogonal constraints.
        h = np.concatenate([h, -np.ones(h.shape, dtype=h.dtype)], axis=0)
    
    return As, bs, constNrIdx, h


def _aux_computeBoundsZonotope(c: np.ndarray, G: np.ndarray, options: Dict[str, Any],
                                bc: np.ndarray, br: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the bounds of a batch of zonotopes.
    
    MATLAB: function [l,u,c,G,cl_,cu_,G_] = aux_computeBoundsZonotope(c,G,options,bc,br)
    """
    if bc.size > 1:
        # Obtain the batch size.
        bSz = bc.shape[1]
        # Replicate sets to match the batch size.
        c, G = _aux_matchBatchSize(c, G, bSz, options)
    
    if bc.size > 0:
        # Scale and offset the input set.
        c_, G_ = _aux_scaleAndOffsetZonotope(c, G, bc, br)
    else:
        # The input set is not scaled.
        c_ = c
        G_ = G
    
    # Obtain number of hidden neurons.
    nk, _, bSz = G_.shape
    
    # Compute the radius of the zonotope.
    r_ = np.sum(np.abs(G_), axis=1).reshape(nk, bSz)
    
    if options.get('nn', {}).get('interval_center', False):
        # Compute center and center radius.
        cl_ = c_[:, 0, :].reshape(nk, bSz)
        cu_ = c_[:, 1, :].reshape(nk, bSz)
        # Compute the un-scaled center.
        c = 0.5 * np.sum(c, axis=1).reshape(nk, bSz) if c.ndim > 2 else c
    else:
        # The radius is zero.
        cl_ = c_
        cu_ = c_
    
    # Compute the bounds.
    l = cl_ - r_
    u = cu_ + r_
    
    return l, u, c, G, cl_, cu_, G_


def _aux_computeBoundsOfInputSet(layer: Any, options: Dict[str, Any], bc: np.ndarray, br: np.ndarray,
                                 constNrIdx: np.ndarray, d: np.ndarray, prevNrXs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given layer, compute the bounds w.r.t. the current hypercube.
    
    MATLAB: function [l,u,cl_,cu_,G_] = aux_computeBoundsOfInputSet(layer,options,bc,br,constNrIdx,d,prevNrXs)
    """
    # Obtain the batch size.
    bSz = max(constNrIdx.shape[1] if constNrIdx.size > 0 else 0,
              prevNrXs.shape[1] if prevNrXs.size > 0 else 0)
    if bSz == 0:
        bSz = 1
    
    # Obtain the number of input generators.
    if hasattr(layer, 'backprop') and isinstance(layer.backprop, dict):
        store = layer.backprop.get('store', {})
        qIds = store.get('genIds', slice(None))
    else:
        qIds = slice(None)
    
    if hasattr(layer, 'backprop') and isinstance(layer.backprop, dict):
        store = layer.backprop.get('store', {})
        if 'inc' in store and 'inG' in store:
            # Obtain the input set.
            c = store['inc']
            G = store['inG']
            if isinstance(qIds, (list, np.ndarray)):
                G = G[:, qIds, :] if G.ndim == 3 else G
            
            # Compute bound based on the refinement.
            l, u, c, _, cl_, cu_, G_ = _aux_computeBoundsZonotope(c, G, options, bc, br)
        else:
            # There is no stored input set.
            l = np.full((1, bSz), -np.inf, dtype=bc.dtype if bc.size > 0 else np.float32)
            u = np.full((1, bSz), np.inf, dtype=bc.dtype if bc.size > 0 else np.float32)
            cl_ = np.array([])
            cu_ = np.array([])
            G_ = np.array([])
    else:
        l = np.full((1, bSz), -np.inf, dtype=bc.dtype if bc.size > 0 else np.float32)
        u = np.full((1, bSz), np.inf, dtype=bc.dtype if bc.size > 0 else np.float32)
        cl_ = np.array([])
        cu_ = np.array([])
        G_ = np.array([])
    
    # Use bounds based on splitting.
    # Obtain the indices of the neurons of the current layer.
    neuronIds = getattr(layer, 'neuronIds', np.array([]))
    
    if constNrIdx.size > 0 and np.any(~np.isnan(constNrIdx)):
        # Compute bounds based on current splits.
        l_split, u_split = _aux_obtainBoundsFromSplits(neuronIds, bSz, constNrIdx, np.array([]), np.array([]))
        # Update bounds with split bounds (take intersection)
        l = np.maximum(l, l_split) if l_split.size > 0 else l
        u = np.minimum(u, u_split) if u_split.size > 0 else u
    
    if prevNrXs.size > 0 and options.get('nn', {}).get('num_splits', 2) == 2 and \
       options.get('nn', {}).get('split_position', 'zero') == 'zero':
        # Apply bounds from previous splits.
        # For zero splits, d and c are empty, so we use empty arrays
        l_split, u_split = _aux_obtainBoundsFromSplits(neuronIds, bSz, np.array([]), np.array([]), np.array([]))
        # Update bounds with split bounds (take intersection)
        l = np.maximum(l, l_split) if l_split.size > 0 else l
        u = np.minimum(u, u_split) if u_split.size > 0 else u
    
    # Identify NaN (indicate empty sets).
    isEmpty = np.isnan(l) | np.isnan(u)
    
    # Remove empty sets.
    l[isEmpty] = np.nan
    u[isEmpty] = np.nan
    
    return l, u, cl_, cu_, G_


def _aux_refineInputSet(nn: 'NeuralNetwork', options: Dict[str, Any], storeInputs: bool,
                        x: np.ndarray, Gx: np.ndarray, y: np.ndarray, Gy: np.ndarray,
                        A: np.ndarray, b: np.ndarray, numUnionConst: int, safeSet: bool,
                        As: np.ndarray, bs: np.ndarray, newNrXs: np.ndarray, prevNrXs: np.ndarray,
                        reluConstrHeuristic: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine input set based on output specification.
    
    MATLAB: function [l,u,nrXis] = aux_refineInputSet(nn,options,storeInputs,x,Gx,y,Gy,A,b,numUnionConst,safeSet,As,bs,newNrXs,prevNrXs,reluConstrHeuristic)
    """
    # Specify improvement tolerance.
    improveTol = 1e-2
    
    # Specify the minimum and maximum number of refinement iterations per layer.
    minRefIter = options.get('nn', {}).get('refinement_min_iter', 1)
    maxRefIter = options.get('nn', {}).get('refinement_max_iter', 5)
    
    # Extract type of refinement.
    layerwise = (options.get('nn', {}).get('refinement_method', 'zonotack') == 'zonotack-layerwise')
    
    # Specify whether approximation slopes are gradient optimized.
    gradientSlopeOptimization = (
        options.get('nn', {}).get('input_split_heuristic', '') == 'zono-norm-gradient' or
        (options.get('nn', {}).get('num_neuron_splits', 0) > 0 and
         options.get('nn', {}).get('neuron_split_heuristic', '') in ['zono-norm-gradient', 'least-unstable-gradient']) or
        (options.get('nn', {}).get('num_relu_constraints', 0) > 0 and
         options.get('nn', {}).get('relu_constraint_heuristic', '') in ['zono-norm-gradient', 'least-unstable-gradient'])
    )
    
    # Enumerate the layers of the neural networks.
    # MATLAB: [layers,ancIdx] = nn.enumerateLayers();
    layers, ancIdx, _, _ = _enumerateLayers(nn)
    
    # Obtain the indices of the activation layers.
    # MATLAB: actIdxLayer = actIdxLayer(arrayfun(@(i) isa(layers{i},'nnActivationLayer'),actIdxLayer));
    from ..layers.nonlinear.nnActivationLayer import nnActivationLayer
    actIdxLayer = [i for i in range(len(layers)) if isinstance(layers[i], nnActivationLayer)]
    
    if layerwise:
        # We refine all activation layers.
        refIdxLayer = [0] + actIdxLayer  # 0 is input layer
        # Flip the layers for a backward refinement.
        refIdxLayer = refIdxLayer[::-1]
    else:
        # We only refine the input.
        refIdxLayer = [0]  # 0 represents input layer
    
    # Obtain number of generators and batchsize.
    nK, q, bSz = Gy.shape
    
    # Convert and join the general- & input-split constraints.
    C, d, newSplits, constNrIdx = _aux_convertSplitConstraints(As, bs, newNrXs)
    
    # Replicate set for split constraints.
    x, Gx = _aux_matchBatchSize(x, Gx, bSz * newSplits, options)
    y, Gy = _aux_matchBatchSize(y, Gy, bSz * newSplits, options)
    # Update the batch size.
    bSz = bSz * newSplits
    
    if constNrIdx.size > 0 and bs.shape[1] == 1:
        # Compute the indices of the split neurons.
        nrXis = constNrIdx
        # Remove all indices that do not correspond to the split of a neuron.
        nrXis = nrXis[~np.all(np.isnan(nrXis), axis=0), :] if nrXis.ndim > 1 else nrXis
        nrXis = nrXis.reshape(-1, bSz)
        # Duplicate the indices for previously split neurons.
        from ..layers.linear.nnGeneratorReductionLayer import repelem
        prevNrXs = repelem(prevNrXs, 1, newSplits)
        # Combine the newly split neurons with the previously split ones.
        nrXis = np.concatenate([prevNrXs, nrXis], axis=0) if prevNrXs.size > 0 else nrXis
    else:
        # Currently, we can only remember splits into two pieces.
        nrXis = np.zeros((0, bSz), dtype=Gy.dtype)
    
    # Initialize scale and offset of the generators.
    bc = np.zeros((q, bSz), dtype=Gy.dtype)
    br = np.ones((q, bSz), dtype=Gy.dtype)
    
    # Initialize loop variables.
    refIdx = 0  # index into refIdxLayer
    refIter = 1  # Counter for number of refinement iterations of the current layer.
    
    # Keep track of empty sets.
    isEmpty = np.zeros((1, bSz), dtype=bool)
    
    # Keep track of which inputs sets of which layers need scaling.
    scaleInputSets = np.ones((1, len(layers)), dtype=bool)
    
    # Iterate layers in a backward fashion to propagate the constraints through the layers.
    while refIdx < len(refIdxLayer):
        # Obtain layer index.
        i = refIdxLayer[refIdx]
        
        # Append the index of the current layer to update its input set in the next iterations.
        idxLayer = list(range(ancIdx[i], len(nn.layers)))
        
        # Construct the unsafe output set.
        uYi = _aux_constructUnsafeOutputSet(options, y, Gy, A, b, safeSet, numUnionConst)
        
        # Scale and offset constraints with current hypercube.
        d_, C_ = _aux_scaleAndOffsetZonotope(d, C, -bc, br)
        
        if C_.size > 0:
            # Append split constraints.
            if uYi['A'].size > 0:
                uYi['A'] = np.concatenate([uYi['A'], C_], axis=0)
                uYi['b'] = np.concatenate([uYi['b'], d_], axis=0)
            else:
                uYi['A'] = C_
                uYi['b'] = d_
        
        if options.get('nn', {}).get('num_relu_constraints', 0) > 0:
            # Compute ReLU tightening constraints
            reluConstrHeuristic = options.get('nn', {}).get('relu_constr_heuristic', 'least-unstable')
            At, bt, nrCt = _aux_reluTightenConstraints(nn, options, idxLayer, reluConstrHeuristic,
                                                        bc, br, scaleInputSets, prevNrXs)
            # Append ReLU constraints to unsafe output set
            if At.size > 0:
                if uYi['A'].size > 0:
                    uYi['A'] = np.concatenate([uYi['A'], At], axis=0)
                    uYi['b'] = np.concatenate([uYi['b'], bt], axis=0)
                else:
                    uYi['A'] = At
                    uYi['b'] = bt
        
        # Compute the bounds of the unsafe inputs (hypercube).
        ly, uy, bli, bui = _aux_boundsOfConZonotope(uYi, numUnionConst, options)
        # Update empty sets.
        isEmpty = isEmpty | np.any(np.isnan(ly), axis=0) | np.any(np.isnan(uy), axis=0)
        
        # Compute the center and radius of the new inner hypercube.
        bci = 0.5 * (bui + bli)
        bri = 0.5 * (bui - bli)
        
        # Update the hypercube.
        bc = bc + br * bci
        br = br * bri
        
        # We have to refine the input set of an ancestor layer.
        # Obtain the input set of the ancestor of the i-th layer.
        if i < len(layers):
            layerAnc = layers[ancIdx[i]]
        else:
            # Input layer
            layerAnc = None
        
        # Obtain number of input generators of the i-th layer.
        if layerAnc is not None and hasattr(layerAnc, 'backprop') and isinstance(layerAnc.backprop, dict):
            store = layerAnc.backprop.get('store', {})
            ancQiIds = store.get('genIds', slice(None))
        else:
            ancQiIds = slice(None)
        
        if layerwise and layerAnc is not None:
            # We refine the input set of the ancestor layer.
            if hasattr(layerAnc, 'backprop') and isinstance(layerAnc.backprop, dict):
                store = layerAnc.backprop.get('store', {})
                cAnc = store.get('inc', x)
                GAnc = store.get('inG', Gx)
                if isinstance(ancQiIds, (list, np.ndarray)):
                    GAnc = GAnc[:, ancQiIds, :] if GAnc.ndim == 3 else GAnc
            else:
                cAnc = x
                GAnc = Gx
            # Replicate sets to match the batch size.
            cAnc, GAnc = _aux_matchBatchSize(cAnc, GAnc, bSz, options)
        else:
            # We only refine the input set.
            cAnc = x
            GAnc = Gx[:, ancQiIds, :] if isinstance(ancQiIds, (list, np.ndarray)) and Gx.ndim == 3 else Gx
        
        # Update scale and offset of the input set to compute a smaller output set.
        if scaleInputSets[0, i] if i < scaleInputSets.shape[1] else True or not layerwise:
            cAnc, GAnc = _aux_scaleAndOffsetZonotope(cAnc, GAnc, bc, br)
        else:
            cAnc, GAnc = _aux_scaleAndOffsetZonotope(cAnc, GAnc, bci, bri)
        
        # Store computed bounds in the layers for tighter approximations.
        bndIdxLayer = [j for j in actIdxLayer if ancIdx[i] <= ancIdx[j]]
        for j in bndIdxLayer:
            # Obtain the j-th layer.
            layerj = layers[j]
            
            # Compute the bounds of the j-th layer.
            if scaleInputSets[0, j] if j < scaleInputSets.shape[1] else True:
                lj, uj, _, _, _ = _aux_computeBoundsOfInputSet(layerj, options, bc, br, constNrIdx, d, prevNrXs)
            else:
                lj, uj, _, _, _ = _aux_computeBoundsOfInputSet(layerj, options, bci, bri, constNrIdx, d, prevNrXs)
            
            # Store the computed bounds in the layers.
            if not hasattr(layerj, 'backprop'):
                layerj.backprop = {'store': {}}
            elif not isinstance(layerj.backprop, dict):
                layerj.backprop = {'store': {}}
            layerj.backprop['store']['l'] = lj
            layerj.backprop['store']['u'] = uj
        
        # Store inputs for each layer by enabling backpropagation.
        options['nn']['train'] = options.get('nn', {}).get('train', {})
        options['nn']['train']['backprop'] = storeInputs
        # Compute a new output enclosure.
        y, Gy = nn.evaluateZonotopeBatch_(cAnc, GAnc, options, idxLayer)
        if gradientSlopeOptimization:
            # Update the gradients to optimize slope.
            storeGradients = True
            _aux_updateGradients(nn, options, idxLayer, Gy, A, b, storeGradients)
        options['nn']['train']['backprop'] = False
        
        if storeInputs:
            # New input sets are computed for the layers; update scaling index.
            scaleInputSets[0, ancIdx >= ancIdx[i]] = False
        
        # Check if we can further refine the current layer.
        if refIter <= minRefIter or (refIter < maxRefIter and
                                     np.any(~isEmpty & (np.min(bri, axis=0) < 1 - improveTol))):
            # Do another refinement iteration on the current layer.
            refIter += 1
        else:
            # No more refinement possible.
            refIdx += 1
            # Reset refinement iteration counter.
            refIter = 1
    
    # Remove the stored bounds.
    for i in actIdxLayer:
        # Obtain the i-th layer.
        layeri = layers[i]
        # Clear the stored bounds and gradients.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            store.pop('slope_gradients', None)
            store.pop('l', None)
            store.pop('u', None)
            store.pop('dm', None)
    
    # Compute bounds of the refined input set.
    l, u, _, _, _, _, _ = _aux_computeBoundsZonotope(x, Gx, options, bc, br)
    
    # Update bounds to represent empty sets.
    l[:, isEmpty.flatten()] = np.nan
    u[:, isEmpty.flatten()] = np.nan
    
    return l, u, nrXis


def _aux_updateGradients(nn: 'NeuralNetwork', options: Dict[str, Any], idxLayer: List[int],
                        Yi: np.ndarray, A: np.ndarray, b: np.ndarray, storeGradients: bool) -> np.ndarray:
    """
    Update the gradient of the f-radius stored in the layers of the neural network.
    The gradients are used to optimize the approximation slope as well as for splitting heuristics.
    
    MATLAB: function grad = aux_updateGradients(nn,options,idxLayer,Yi,A,b,storeGradients)
    """
    # Store the gradients of the approximation errors.
    if 'nn' not in options:
        options['nn'] = {}
    options['nn']['store_approx_error_gradients'] = storeGradients
    
    # Check if Yi is an interval (has inf and sup) or a zonotope (3D array)
    if isinstance(Yi, dict) and 'inf' in Yi and 'sup' in Yi:
        # We compute the gradient of an interval.
        # Obtain the bounds.
        yli = Yi['inf']
        yui = Yi['sup']
        
        # Compute the gradient of the f-radius.
        frad = np.sqrt(np.sum((yui - yli)**2, axis=0))
        gyli = -yli / (frad + 1e-6)
        gyui = -yui / (frad + 1e-6)
        # Backpropagate the gradients to identify which input dimensions
        # create the largest interval approximation error.
        # For now, we do not store the gradient, because we only use the
        # gradient for the construction of the input zonotope.
        # MATLAB: [~,grad] = nn.backpropIntervalBatch(gyli,gyui,options,idxLayer,false);
        gl, gu = gyli.copy(), gyui.copy()
        # Iterate through layers in reverse order (from output to input)
        for i in range(len(idxLayer) - 1, -1, -1):
            k = idxLayer[i]
            layer_k = nn.layers[k]
            if hasattr(layer_k, 'backpropIntervalBatch'):
                # Get stored input bounds from forward pass
                if hasattr(layer_k, 'backprop') and isinstance(layer_k.backprop, dict):
                    store = layer_k.backprop.get('store', {})
                    # Try to get input from stored input (could be interval dict or bounds)
                    if 'input' in store:
                        input_data = store['input']
                        if isinstance(input_data, dict) and 'inf' in input_data and 'sup' in input_data:
                            l_in = input_data['inf']
                            u_in = input_data['sup']
                        elif 'l' in store and 'u' in store:
                            l_in = store['l']
                            u_in = store['u']
                        else:
                            # Fallback: use current gradients as bounds (approximation)
                            l_in = gl
                            u_in = gu
                    elif 'l' in store and 'u' in store:
                        l_in = store['l']
                        u_in = store['u']
                    else:
                        # Fallback: use current gradients as bounds (approximation)
                        l_in = gl
                        u_in = gu
                else:
                    # No stored input, use current gradients as bounds (approximation)
                    l_in = gl
                    u_in = gu
                gl, gu = layer_k.backpropIntervalBatch(l_in, u_in, gl, gu, options)
        # The gradient at the input is the result
        # For interval backprop, grad typically represents the gradient w.r.t. input
        # We combine gl and gu (taking the maximum absolute value for each dimension)
        if gl.ndim > 0 and gu.ndim > 0:
            grad = np.maximum(np.abs(gl), np.abs(gu))
        else:
            grad = None
    else:
        # We compute the gradient of a zonotope.
        Gyi = Yi
        # Obtain the number of output dimensions and batch size.
        nK, _, bSz = Gyi.shape
        
        # Compute the gradient of the f-radius.
        frad = np.sqrt(np.sum(Gyi**2, axis=(0, 1)))  # Sum over first two dimensions
        gGyi = Gyi / (frad.reshape(1, 1, bSz) + 1e-6)
        
        # Compute a dummy center gradient.
        if options.get('nn', {}).get('interval_center', False):
            gcyi = np.zeros((nK, 2, bSz), dtype=Gyi.dtype)
        else:
            gcyi = np.zeros((nK, bSz), dtype=Gyi.dtype)
        
        # Compute gradient of the f-radius of the output set; the gradient 
        # is used to split the neuron in the network as well as input dimensions.
        # MATLAB: [~,grad] = nn.backpropZonotopeBatch_(gcyi,gGyi,options,idxLayer,false);
        gc, gG = gcyi.copy(), gGyi.copy()
        # Iterate through layers in reverse order (from output to input)
        for i in range(len(idxLayer) - 1, -1, -1):
            k = idxLayer[i]
            layer_k = nn.layers[k]
            if hasattr(layer_k, 'backpropZonotopeBatch'):
                # Get stored input from forward pass
                if hasattr(layer_k, 'backprop') and isinstance(layer_k.backprop, dict):
                    store = layer_k.backprop.get('store', {})
                    c_in = store.get('inc', np.zeros_like(gc))
                    G_in = store.get('inG', np.zeros_like(gG))
                else:
                    c_in = np.zeros_like(gc)
                    G_in = np.zeros_like(gG)
                gc, gG = layer_k.backpropZonotopeBatch(c_in, G_in, gc, gG, options)
        # The gradient at the input is the result
        # gG has shape (input_dim, numGen, batch) - this is the gradient w.r.t. generators
        grad = gG
    
    return grad


def _aux_obtainBoundsFromSplits(neuronIds: np.ndarray, bSz: int, constNrIdx: np.ndarray,
                                d: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the bounds from constraints. The argument d represents the
    offset of a split constraint; together with the center c we can
    recompute the split-bound.
    
    MATLAB: function [l,u] = aux_obtainBoundsFromSplits(neuronIds,bSz,constNrIdx,d,c)
    """
    # Initialize the bounds.
    l = np.full((len(neuronIds), bSz), -np.inf, dtype=d.dtype if d.size > 0 else np.float32)
    u = np.full((len(neuronIds), bSz), np.inf, dtype=d.dtype if d.size > 0 else np.float32)
    
    # Handle multiple splits per dimension by aggregating bounds to get the best (tightest) bounds for each dimension.
    # MATLAB TODO: implement handling of multiple splits per dimension. Use aggregation to get the best bounds for each dimension.
    # Python implementation: Fully implemented - when multiple constraints exist for the same neuron, we aggregate by taking
    # the maximum lower bound (tightest lower bound) and minimum upper bound (tightest upper bound) across all constraints.
    if constNrIdx.size > 0 and np.any(~np.isnan(constNrIdx)):
        # Check which constraints contain bounds for the current layer.
        # MATLAB: isBnd = (permute(abs(constNrIdx),[3 2 1]) == repmat(neuronIds',1,bSz));
        constNrIdx_abs = np.abs(constNrIdx)  # (num_constraints, bSz)
        neuronIds_2d = neuronIds.reshape(-1, 1)  # (num_neurons, 1)
        isBnd = (constNrIdx_abs.T.reshape(1, bSz, -1) == neuronIds_2d.reshape(-1, 1, 1))  # (num_neurons, bSz, num_constraints)
        
        if not np.any(isBnd):
            # There are no bounds from splits.
            return l, u
        
        # Check which constraints contain lower or upper bounds.
        # MATLAB: isLBnd = (permute(sign(constNrIdx),[3 2 1]) == -1);
        # MATLAB: isUBnd = (permute(sign(constNrIdx),[3 2 1]) == 1);
        constNrIdx_sign = np.sign(constNrIdx)  # (num_constraints, bSz)
        isLBnd = (constNrIdx_sign.T.reshape(1, bSz, -1) == -1)  # (num_neurons, bSz, num_constraints)
        isUBnd = (constNrIdx_sign.T.reshape(1, bSz, -1) == 1)  # (num_neurons, bSz, num_constraints)
        
        # Compute the indices into the bounds.
        # MATLAB: ljIdx = any(isBnd & isLBnd,3);
        ljIdx = np.any(isBnd & isLBnd, axis=2)  # (num_neurons, bSz)
        # MATLAB: ujIdx = any(isBnd & isUBnd,3);
        ujIdx = np.any(isBnd & isUBnd, axis=2)  # (num_neurons, bSz)
        
        if c.size > 0 and d.size > 0:
            # Handle multiple splits per dimension by aggregating bounds.
            # For each neuron and batch, collect all bounds and take the tightest (most restrictive).
            num_neurons = len(neuronIds)
            bndLj_all = np.full((num_neurons, bSz), np.nan, dtype=d.dtype)
            bndUj_all = np.full((num_neurons, bSz), np.nan, dtype=d.dtype)
            
            # Iterate over each constraint to extract bounds
            for constr_idx in range(constNrIdx.shape[0]):
                for batch_idx in range(bSz):
                    const_nr = constNrIdx[constr_idx, batch_idx]
                    if np.isnan(const_nr):
                        continue
                    neuron_idx = np.where(neuronIds == np.abs(const_nr))[0]
                    if neuron_idx.size == 0:
                        continue
                    neuron_idx = neuron_idx[0]
                    
                    # Compute bound value
                    if const_nr < 0:  # Lower bound
                        bnd_val = d[constr_idx, batch_idx] - (c[neuron_idx, batch_idx] if c.ndim > 1 else c[neuron_idx])
                        # Take maximum (tightest lower bound)
                        if np.isnan(bndLj_all[neuron_idx, batch_idx]) or bnd_val > bndLj_all[neuron_idx, batch_idx]:
                            bndLj_all[neuron_idx, batch_idx] = bnd_val
                    elif const_nr > 0:  # Upper bound
                        bnd_val = d[constr_idx, batch_idx] + (c[neuron_idx, batch_idx] if c.ndim > 1 else c[neuron_idx])
                        # Take minimum (tightest upper bound)
                        if np.isnan(bndUj_all[neuron_idx, batch_idx]) or bnd_val < bndUj_all[neuron_idx, batch_idx]:
                            bndUj_all[neuron_idx, batch_idx] = bnd_val
            
            # Update bounds based on the aggregated splits.
            valid_lower = ~np.isnan(bndLj_all) & ljIdx
            valid_upper = ~np.isnan(bndUj_all) & ujIdx
            l[valid_lower] = bndLj_all[valid_lower]
            u[valid_upper] = bndUj_all[valid_upper]
        else:
            # The bounds are 0, because we split a ReLU at 0.
            # When c and d are empty, we still need to set bounds for indices where splits exist
            if np.any(ljIdx):
                l[ljIdx] = 0
            if np.any(ujIdx):
                u[ujIdx] = 0
    
    return l, u


def _aux_reluTightenConstraints(nn: 'NeuralNetwork', options: Dict[str, Any],
                                idxLayer: Optional[List[int]], heuristic: str,
                                bc: np.ndarray, br: np.ndarray, scaleInputSets: np.ndarray,
                                prevNrXs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tightening constraints for unstable ReLU neurons.
    
    MATLAB: function [At,bt,nrCt] = aux_reluTightenConstraints(nn,options,idxLayer,heuristic,bc,br,scaleInputSets,prevNrXs)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem, pagemtimes
    from ..layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Specify a tolerance to avoid numerical issues.
    tol = 1e-8
    
    # Obtain the number of constraints.
    numConstr = options.get('nn', {}).get('num_relu_constraints', 0)
    
    # Initialize constraints.
    q = 0  # Number of considered generators.
    p = 0  # Number of constraints.
    # (i) ReLU(x) >= 0
    At0 = np.array([])
    bt0 = np.array([])
    # (ii) ReLU(x) >= x
    Atd = np.array([])
    btd = np.array([])
    # (iii) ReLU(x) <= u/(u-l)*(x-l)
    Atd2 = np.array([])
    btd2 = np.array([])
    # Initial heuristics.
    h = np.array([])
    # Initialize indices of neuron constraints.
    nrCt = np.array([])
    
    # Enumerate the layers of the neural networks.
    layers, _, _, succIdx = _enumerateLayers(nn)
    
    if scaleInputSets.size == 0:
        scaleInputSets = np.zeros((1, len(layers)), dtype=bool)
    
    if idxLayer is None or len(idxLayer) == 0:
        idxLayer = list(range(len(layers)))
    
    # Compute the indices of ReLU layers.
    idxLayer = [i for i in idxLayer if isinstance(layers[i], nnReLULayer)]
    
    # Obtain the batch size.
    bSz = bc.shape[1] if bc.ndim > 1 else 1
    
    # Iterate through the layers and find maximal unstable neurons.
    for i in idxLayer:
        # Obtain i-th layer.
        layeri = layers[i]
        # Obtain the input set and compute the bounds.
        if scaleInputSets[0, i] if i < scaleInputSets.shape[1] else False:
            li, ui, cil, ciu, Gi = _aux_computeBoundsOfInputSet(layeri, options, bc, br, np.array([]), np.array([]), prevNrXs)
        else:
            li, ui, cil, ciu, Gi = _aux_computeBoundsOfInputSet(layeri, options,
                                                                  np.zeros_like(bc), np.ones_like(br),
                                                                  np.array([]), np.array([]), prevNrXs)
        # Obtain the indices for the generators containing the approximation errors.
        approxErrGenIds = getattr(layeri, 'approxErrGenIds', np.array([]))
        # Obtain the indices of the neurons of the current layer.
        neuronIds = getattr(layeri, 'neuronIds', np.arange(li.shape[0]))
        # Check if neuron is stable.
        isUnstable = (li < 0) & (0 < ui)
        
        # Obtain the slope and approximation errors.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            dl = store.get('dl', np.zeros_like(li))
            du = store.get('du', np.zeros_like(ui))
        else:
            dl = np.zeros_like(li)
            du = np.zeros_like(ui)
        # Compute center and radius of approximation errors.
        dr = 0.5 * (du - dl)
        
        if dr.shape[1] < bSz:
            # Duplicate the approximation error (there was neuron splitting involved).
            newSplits = bSz // dr.shape[1]
            dr = repelem(dr, 1, newSplits)
        
        # Obtain successor of the i-th layer.
        if i < len(succIdx) and succIdx[i] < len(layers):
            layerj = layers[succIdx[i]]
            # Obtain the input set and compute the bounds.
            if scaleInputSets[0, succIdx[i]] if succIdx[i] < scaleInputSets.shape[1] else False:
                _, _, cjl, cju, Gj = _aux_computeBoundsOfInputSet(layerj, options, bc, br, np.array([]), np.array([]), prevNrXs)
            else:
                _, _, cjl, cju, Gj = _aux_computeBoundsOfInputSet(layerj, options,
                                                                    np.zeros_like(bc), np.ones_like(br),
                                                                    np.array([]), np.array([]), prevNrXs)
            # Obtain the number of generators and the batch size.
            nk, qj, _ = Gj.shape
        else:
            # No successor layer
            continue
        
        # Obtain the sensitivity for heuristic.
        if hasattr(layeri, 'sensitivity') and layeri.sensitivity is not None:
            Si_ = np.maximum(np.abs(layeri.sensitivity), 1e-6)
            sens = np.reshape(np.max(Si_, axis=0), (nk, -1))
        else:
            sens = np.ones((nk, bSz), dtype=li.dtype)
        
        if sens.size > 0 and sens.shape[1] < bSz:
            # Duplicate the sensitivity (there was neuron splitting involved).
            newSplits = bSz // sens.shape[1]
            sens = repelem(sens, 1, newSplits)
        
        if At0.size == 0 or (At0.ndim == 3 and At0.shape[2] < bSz):
            # Match the batch size.
            if At0.size > 0:
                newSplits = bSz // At0.shape[2]
                At0 = repelem(At0, 1, 1, newSplits)
                Atd = repelem(Atd, 1, 1, newSplits)
                Atd2 = repelem(Atd2, 1, 1, newSplits)
            else:
                At0 = np.zeros((0, 0, bSz), dtype=Gi.dtype)
                Atd = np.zeros((0, 0, bSz), dtype=Gi.dtype)
                Atd2 = np.zeros((0, 0, bSz), dtype=Gi.dtype)
        
        if q < qj:
            # Pad constraints with zeros.
            pad_size = qj - q
            At0 = np.concatenate([At0, np.zeros((pad_size, p, bSz), dtype=At0.dtype)], axis=0) if At0.size > 0 else np.zeros((qj, 0, bSz), dtype=Gi.dtype)
            Atd = np.concatenate([Atd, np.zeros((pad_size, p, bSz), dtype=Atd.dtype)], axis=0) if Atd.size > 0 else np.zeros((qj, 0, bSz), dtype=Gi.dtype)
            Atd2 = np.concatenate([Atd2, np.zeros((pad_size, p, bSz), dtype=Atd2.dtype)], axis=0) if Atd2.size > 0 else np.zeros((qj, 0, bSz), dtype=Gi.dtype)
            # Update number of constraints.
            q = qj
        
        if q > Gi.shape[1]:
            # Pad generators with zeros.
            pad_size = q - Gi.shape[1]
            Gi = np.concatenate([Gi, np.zeros((nk, pad_size, bSz), dtype=Gi.dtype)], axis=1)
        if q > qj:
            # Pad generators with zeros.
            pad_size = q - qj
            Gj = np.concatenate([Gj, np.zeros((nk, pad_size, bSz), dtype=Gj.dtype)], axis=1)
        
        # Reverse the sign of the approximation errors.
        Gj_ = Gj.copy()
        if approxErrGenIds.size > 0:
            Gj_[:, approxErrGenIds, :] = -Gj_[:, approxErrGenIds, :]
        
        # (i) ReLU(x) >= 0 
        # --> cj + Gj*\beta - dr >= ReLU(x) >= 0 
        # <--> -Gj*\beta + dr <= cj
        Ati0 = -Gj_  # (nk, q, bSz)
        bti0 = cju + tol  # (nk, bSz)
        # Permute the constraints s.t. the generators are in the first dimension.
        Ati0 = np.transpose(Ati0, (1, 0, 2))  # (q, nk, bSz)
        # Invalidate constraints for stable neurons.
        Ati0[:, ~isUnstable, :] = np.nan
        bti0[~isUnstable, :] = np.nan
        # Append new constraints.
        At0 = np.concatenate([At0, Ati0], axis=1) if At0.size > 0 else Ati0
        bt0 = np.concatenate([bt0, bti0.reshape(-1, bSz)], axis=0) if bt0.size > 0 else bti0.reshape(-1, bSz)
    
        # (ii) ReLU(x) >= x 
        # --> cj + Gj*\beta - dr >= ReLU(x) >= x = ci + Gi*\beta
        # <--> (Gi-Gj)*\beta + dr <= cj - ci
        # Compute difference of generator matrices.
        Atid = Gi - Gj_  # (nk, q, bSz)
        btid = cju - cil + tol  # (nk, bSz)
        # Permute the constraints s.t. the generators are in the first dimension.
        Atid = np.transpose(Atid, (1, 0, 2))  # (q, nk, bSz)
        # Invalidate constraints for stable neurons.
        Atid[:, ~isUnstable, :] = np.nan
        btid[~isUnstable, :] = np.nan
        # Append new constraints.
        Atd = np.concatenate([Atd, Atid], axis=1) if Atd.size > 0 else Atid
        btd = np.concatenate([btd, btid.reshape(-1, bSz)], axis=0) if btd.size > 0 else btid.reshape(-1, bSz)
    
        # (iii) ReLU(x) <= m*(x-li) + ReLU(li)
        # --> cj + Gj*\beta + dr <= ReLU(x) <= m*(x-li) + ReLU(li)
        # <--> (Gj - m*Gi)*\beta + dr <= m*(ci-li) - cj + ReLU(li),
        # where m = (ReLU(ui) - ReLU(li))/(ui-li).
        # Compute the approximation slope; add 1e-10 to avoid numerical issues.
        m_ = (np.maximum(ui, 0) - np.maximum(li, 0)) / (ui - li + 1e-10)  # (nk, bSz)
        Atid2 = Gj - m_.reshape(nk, 1, bSz) * Gi  # (nk, q, bSz)
        btid2 = m_ * (ciu - li) - cjl + np.maximum(li, 0) + tol  # (nk, bSz)
        # Permute the constraints s.t. the generators are in the first dimension.
        Atid2 = np.transpose(Atid2, (1, 0, 2))  # (q, nk, bSz)
        # Avoid numerical issues if bounds are too close.
        boundsAreEqual = (ui - li) < tol
        # Invalidate constraints for stable neurons.
        Atid2[:, ~isUnstable | boundsAreEqual, :] = np.nan
        btid2[~isUnstable | boundsAreEqual, :] = np.nan
        # Append new constraints.
        Atd2 = np.concatenate([Atd2, Atid2], axis=1) if Atd2.size > 0 else Atid2
        btd2 = np.concatenate([btd2, btid2.reshape(-1, bSz)], axis=0) if btd2.size > 0 else btid2.reshape(-1, bSz)
        
        # Obtain the gradient of the zonotope norm.
        if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
            store = layeri.backprop.get('store', {})
            grad = store.get('approx_error_gradients', 0)
            if not isinstance(grad, (int, float)) and grad.shape[1] < bSz:
                # Duplicate the gradient (there was neuron splitting involved).
                newSplits = bSz // grad.shape[1]
                grad = repelem(grad, 1, newSplits)
        else:
            # There is no stored gradient.
            grad = 0
        
        # Compute the heuristic.
        hi = _aux_computeHeuristic(heuristic, i, li, ui, dr, sens, grad,
                                   None, prevNrXs, neuronIds, True, 0.7)
        
        # Append heuristic and sort.
        h_combined = np.concatenate([h, hi], axis=0) if h.size > 0 else hi
        sort_idx = np.argsort(h_combined, axis=0)[::-1]  # Sort descending
        # Only keep the constraints for the top neurons.
        numConstr_ = min(numConstr, h_combined.shape[0])
        h = h_combined[sort_idx[:numConstr_, :], np.arange(bSz)]
    
        # Obtain the indices for the relevant constraints.
        cIdx = sub2ind((At0.shape[1], At0.shape[2]),
                       sort_idx[:numConstr_, :].flatten('F'),  # 1-based
                       np.tile(np.arange(1, bSz + 1), numConstr_))  # 1-based
        cIdx = cIdx - 1  # Convert to 0-based
        
        # Select the relevant constraints.
        At0_flat = At0.reshape(q, -1)
        At0 = At0_flat[:, cIdx].reshape(q, numConstr_, bSz)
        bt0_flat = bt0.reshape(-1, bSz)
        bt0 = bt0_flat[cIdx, :].reshape(numConstr_, bSz)
        Atd_flat = Atd.reshape(q, -1)
        Atd = Atd_flat[:, cIdx].reshape(q, numConstr_, bSz)
        btd_flat = btd.reshape(-1, bSz)
        btd = btd_flat[cIdx, :].reshape(numConstr_, bSz)
        Atd2_flat = Atd2.reshape(q, -1)
        Atd2 = Atd2_flat[:, cIdx].reshape(q, numConstr_, bSz)
        btd2_flat = btd2.reshape(-1, bSz)
        btd2 = btd2_flat[cIdx, :].reshape(numConstr_, bSz)
        
        # Update number of constraints.
        p = At0.shape[1]
        
        # Update indices.
        nrCt_new = np.tile(neuronIds.reshape(-1, 1), (1, bSz))  # (nk, bSz)
        nrCt = np.concatenate([nrCt, nrCt_new], axis=0) if nrCt.size > 0 else nrCt_new
        nrCt = nrCt[sort_idx[:numConstr_, :], np.arange(bSz)]  # (numConstr_, bSz)
    
    # Concatenate the constraints.
    At = np.concatenate([At0, Atd, Atd2], axis=1)  # (q, 3*numConstr_, bSz)
    bt = np.concatenate([bt0, btd, btd2], axis=0)  # (3*numConstr_, bSz)
    nrTypesOfConstr = At.shape[1] // nrCt.shape[0] if nrCt.size > 0 else 1
    nrCt = np.tile(nrCt, (nrTypesOfConstr, 1)) if nrCt.size > 0 else nrCt
    
    # Find the minimal number of invalid constraints across the batch.
    minNumInvalidConstraints = np.min(np.sum(np.any(np.isnan(At), axis=0), axis=0))
    # Sort the constraints and remove invalidated constraints.
    sortIds = np.argsort(np.isnan(bt), axis=0)[::-1]  # Sort descending (NaN first)
    sortIdx = sub2ind((At.shape[1], At.shape[2]),
                      sortIds.flatten('F'),  # 1-based
                      np.tile(np.arange(1, bSz + 1), At.shape[1]))  # 1-based
    sortIdx = sortIdx - 1  # Convert to 0-based
    # Reorder the constraints.
    At_flat = At.reshape(q, -1)
    At = At_flat[:, sortIdx].reshape(q, At.shape[1], bSz)
    bt_flat = bt.reshape(-1, bSz)
    bt = bt_flat[sortIdx, :].reshape(bt.shape[0], bSz)
    nrCt_flat = nrCt.reshape(-1, bSz)
    nrCt = nrCt_flat[sortIdx, :].reshape(nrCt.shape[0], bSz) if nrCt.size > 0 else nrCt
    # Remove the invalid constraints.
    At = At[:, minNumInvalidConstraints:, :]
    bt = bt[minNumInvalidConstraints:, :]
    nrCt = nrCt[minNumInvalidConstraints:, :] if nrCt.size > 0 else nrCt
    # Set all remaining invalid constraints to 0.
    At[np.isnan(At)] = 0
    bt[np.isnan(At)] = 0
    # Transpose constraint matrix.
    At = np.transpose(At, (1, 0, 2))  # (num_constraints, q, bSz)
    
    return At, bt, nrCt
