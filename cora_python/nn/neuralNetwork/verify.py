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

# Import helper functions for verification refinement logic
from .verify_helpers import (
    _aux_pop, _aux_split_with_dim, _aux_enumerateLayers, _enumerateLayers,
    _aux_matchBatchSize, _aux_scaleAndOffsetZonotope, _aux_computeHeuristic,
    _aux_dimSplitConstraints, _aux_convertSplitConstraints, _aux_boundsOfBoundedPolytope,
    _aux_boundsOfConZonotope, _aux_constructUnsafeOutputSet, _aux_neuronConstraints,
    _aux_computeBoundsZonotope, _aux_computeBoundsOfInputSet, _aux_refineInputSet,
    _aux_updateGradients, _aux_obtainBoundsFromSplits, _aux_reluTightenConstraints,
    _aux_constructInputZonotope
)

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
    
    # Ensure b has correct shape for broadcasting: (num_constraints, 1) or (num_constraints,)
    # MATLAB: b is typically (p, 1) where p is number of constraints
    # In MATLAB, b can be scalar, (p, 1) column vector, or (1, p) row vector
    # For broadcasting with ld_ys which is (p, batch_size), we need b to be (p, 1)
    # Convert b to numpy array and ensure it's a column vector
    b_original = b  # Store original for reference
    b = np.asarray(b)
    if b.ndim == 0:
        # b is scalar, reshape to (1, 1) for broadcasting
        b = b.reshape(1, 1)
    elif b.ndim == 1:
        # b is 1D array, could be (p,) or (1, p) in MATLAB
        # MATLAB's b from specs.set.d or specs.set.b is typically (p, 1) column vector
        # But if it's passed as (p,), we need to reshape to (p, 1)
        # If it's (1, p) row vector, we need to transpose to (p, 1)
        # For now, assume it's (p,) and reshape to (p, 1) column vector
        b = b.reshape(-1, 1)
    elif b.ndim == 2:
        # b is 2D array, could be (p, 1) or (1, p)
        if b.shape[0] == 1 and b.shape[1] > 1:
            # b is (1, p) row vector, transpose to (p, 1) column vector
            b = b.T
        # else b is already (p, 1) column vector, keep as is
    # b should now be (num_constraints, 1) for consistent broadcasting
    
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
    # MATLAB: xs = x; rs = r; nrXs = zeros([0 size(x,2)]);
    xs = x.copy()  # (n, num_patches) - initially (n, 1) for single patch
    rs = r.copy()  # (n, num_patches) - initially (n, 1) for single patch
    nrXs = np.zeros((0, x.shape[1]), dtype=inputDataClass)  # Neuron split indices

    if verbose:
        print(f"Initial radius: min={np.min(rs):.6f}, max={np.max(rs):.6f}")
    # Obtain number of input dimensions.
    n0 = x.shape[0]
    
    # Extract numInitGens before the main loop (MATLAB: line 79)
    # MATLAB: numInitGens = min(options.nn.train.num_init_gens,n0);
    numInitGens = options.get('nn', {}).get('train', {}).get('num_init_gens', n0)
    numInitGens = min(numInitGens, n0)

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
        # MATLAB: [xi,ri,nrXi,xs,rs,nrXs,qIdx] = aux_pop(xs,rs,nrXs,bSz,options);
        xi, ri, nrXi, xs, rs, nrXs, qIdx = _aux_pop(xs, rs, nrXs, bs, options)
        
        # Move the batch to the GPU.
        # In MATLAB: xi = cast(xi,'like',inputDataClass); ri = cast(ri,'like',inputDataClass); nrXi = cast(nrXi,'like',inputDataClass);
        if useGpu and TORCH_AVAILABLE:
            # Convert to PyTorch tensors and move to GPU
            xi = torch.tensor(xi, dtype=torch.float32, device=device)
            ri = torch.tensor(ri, dtype=torch.float32, device=device)
            nrXi = torch.tensor(nrXi, dtype=torch.float32, device=device) if nrXi.size > 0 else nrXi
        else:
            # Use CPU arrays
            xi = xi.astype(inputDataClass)
            ri = ri.astype(inputDataClass)
            nrXi = nrXi.astype(inputDataClass) if nrXi.size > 0 else nrXi
        
        # Obtain the current batch size.
        cbSz = xi.shape[1]
        
        # Store ld_Gyi for zonotack falsification method (computed later in verification section)
        ld_Gyi = None
        
        # Verification --------------------------------------------------------
        # Compute interval gradient if needed for input generator heuristic
        inputGenHeuristic = options.get('nn', {}).get('input_generator_heuristic', 'most-sensitive-input-radius')
        if inputGenHeuristic == 'zono-norm-gradient':
            # MATLAB: ivalGrad = aux_updateGradients(nn,options,idxLayer,Yival, ...);
            # For now, we'll compute this later if needed
            ivalGrad = None
        else:
            ivalGrad = None
        
        # Construct input zonotope.
        # MATLAB: [cxi,Gxi,inputDimIdx] = aux_constructInputZonotope(options,inputGenHeuristic,xi,ri,batchG,sens,ivalGrad,numInitGens);
        # Compute sensitivity for input generator heuristic if needed
        sens = None
        if inputGenHeuristic in ['most-sensitive-input-radius', 'zono-norm-gradient']:
            S, _ = nn.calcSensitivity(xi, options, store_sensitivity=False)
            if S.ndim == 3:
                S_abs = np.abs(S)
                S_max = np.maximum(S_abs, 1e-6)
                sens_max = np.max(S_max, axis=0)  # (input_dim, batch)
                sens = sens_max.T  # (batch, input_dim)
            else:
                sens = np.ones((cbSz, n0), dtype=xi.dtype)
        
        cxi, Gxi, inputDimIdx = _aux_constructInputZonotope(
            options, inputGenHeuristic, xi, ri, batchG, sens, ivalGrad, numInitGens
        )
        
        # Python layers expect 3D input, so reshape cxi if needed
        # MATLAB: cxi can be 2D (n0, bSz) or 3D (n0, 2, bSz) depending on interval_center
        # But Python layers always expect 3D, so reshape if 2D
        if cxi.ndim == 2:
            # Reshape from (n0, bSz) to (n0, 1, bSz) for Python layers
            if useGpu and TORCH_AVAILABLE:
                cxi = cxi.reshape(cxi.shape[0], 1, cxi.shape[1])
            else:
                cxi = cxi.reshape(cxi.shape[0], 1, cxi.shape[1])
        
        # Handle previous neuron splits (nrXi)
        # MATLAB: if ~isempty(nrXi) ... end
        if nrXi.size > 0:
            # Obtain the number of previous split constraints.
            p = nrXi.shape[0]
            # For now we only support storing ReLU splits at 0.
            # TODO: remember arbitrary splits.
            # Therefore, create dummy offsets.
            dummyd = np.zeros((p, cbSz), dtype=xi.dtype)
            # Store computed bounds in the layers for tighter approximations.
            layers = _enumerateLayers(nn)[0]
            for i, layeri in enumerate(layers):
                # Obtain the i-th layer.
                if hasattr(layeri, '__class__') and 'Activation' in layeri.__class__.__name__:
                    # Obtain the indices of the neurons of the current layer.
                    neuronIds = layeri.neuronIds if hasattr(layeri, 'neuronIds') else np.arange(layeri.getOutputSize(layeri.inputSize))
                    # Create dummy centers.
                    dummyc = np.zeros((len(neuronIds), cbSz), dtype=xi.dtype)
                    # Compute bounds from previous splits.
                    li, ui = _aux_obtainBoundsFromSplits(neuronIds, cbSz, nrXi, dummyd, dummyc)
                    # Store the computed bounds in the layers.
                    if hasattr(layeri, 'backprop') and isinstance(layeri.backprop, dict):
                        if 'store' not in layeri.backprop:
                            layeri.backprop['store'] = {}
                        layeri.backprop['store']['l'] = li
                        layeri.backprop['store']['u'] = ui
        
        # Store inputs for each layer by enabling backpropagation.
        storeInputs = options.get('nn', {}).get('train', {}).get('backprop', False)
        options['nn']['train']['backprop'] = storeInputs
        # Compute output enclosure.
        yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)
        # Disable backpropagation.
        options['nn']['train']['backprop'] = False
        
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
                # Verify sgrad is ±1 (or 0)
                # sgrad should contain only values in {-1, 0, 1}
                sgrad_min, sgrad_max = torch.min(sgrad).item(), torch.max(sgrad).item()
                sgrad_abs = torch.abs(sgrad)
                # Check if all values are either ±1 (abs=1) or 0
                is_valid = torch.all((torch.isclose(sgrad_abs, torch.ones_like(sgrad_abs), atol=1e-6)) | (torch.isclose(sgrad, torch.zeros_like(sgrad), atol=1e-6)))
                if not is_valid:
                    # sgrad should be ±1 (or 0 if grad is exactly 0)
                    raise ValueError(f"sgrad should be ±1 or 0, but got range [{sgrad_min}, {sgrad_max}], unique values: {torch.unique(sgrad).cpu().numpy()}")
                xi_repeated = xi.repeat(1, p)  # (n0, cbSz*p)
                ri_repeated = ri.repeat(1, p)  # (n0, cbSz*p)
                zi = xi_repeated + ri_repeated * sgrad
                # Verify zi is within [xi - ri, xi + ri] by construction
                # Convert to numpy for bounds checking
                xi_np = xi.cpu().numpy() if isinstance(xi, torch.Tensor) else xi
                ri_np = ri.cpu().numpy() if isinstance(ri, torch.Tensor) else ri
                zi_np = zi.cpu().numpy() if isinstance(zi, torch.Tensor) else zi
                # Use tolerance for floating-point comparison (MATLAB uses double precision by default)
                tol = 1e-6
                for batch_idx in range(cbSz):
                    xi_b = xi_np[:, batch_idx:batch_idx+1]  # (n0, 1)
                    ri_b = ri_np[:, batch_idx:batch_idx+1]  # (n0, 1)
                    zi_b_candidates = zi_np[:, batch_idx*p:(batch_idx+1)*p]  # (n0, p)
                    zi_lower = xi_b - ri_b  # (n0, 1)
                    zi_upper = xi_b + ri_b  # (n0, 1)
                    # Check all p candidates for this batch entry with tolerance
                    violations_lower = zi_b_candidates < (zi_lower - tol)  # (n0, p)
                    violations_upper = zi_b_candidates > (zi_upper + tol)  # (n0, p)
                    any_violation_lower = np.any(violations_lower)
                    any_violation_upper = np.any(violations_upper)
                    if any_violation_lower or any_violation_upper:
                        # This should never happen if sgrad = ±1 (within tolerance)
                        dims_violating_lower = np.where(np.any(violations_lower, axis=1))[0]
                        dims_violating_upper = np.where(np.any(violations_upper, axis=1))[0]
                        candidates_violating_lower = np.where(np.any(violations_lower, axis=0))[0]
                        candidates_violating_upper = np.where(np.any(violations_upper, axis=0))[0]
                        sgrad_b = sgrad[:, batch_idx*p:(batch_idx+1)*p].cpu().numpy() if isinstance(sgrad, torch.Tensor) else sgrad[:, batch_idx*p:(batch_idx+1)*p]
                        raise ValueError(
                            f"zi out of bounds for batch {batch_idx}:\n"
                            f"  violations_lower: dims={dims_violating_lower}, candidates={candidates_violating_lower}\n"
                            f"  violations_upper: dims={dims_violating_upper}, candidates={candidates_violating_upper}\n"
                            f"  sgrad range=[{sgrad_min}, {sgrad_max}]\n"
                            f"  For violating candidates, sgrad values: {sgrad_b[:, candidates_violating_lower] if len(candidates_violating_lower) > 0 else 'N/A'}\n"
                            f"  xi_b={xi_b.flatten()}, ri_b={ri_b.flatten()}\n"
                            f"  zi_lower={zi_lower.flatten()}, zi_upper={zi_upper.flatten()}\n"
                            f"  zi_b_candidates (first violating)={zi_b_candidates[:, candidates_violating_lower[0] if len(candidates_violating_lower) > 0 else 0].flatten()}"
                        )
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
                # MATLAB: xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;
                # This should keep zi within [xi - ri, xi + ri] since sgrad = ±1
                # Verify sgrad is ±1 (or 0)
                # sgrad should contain only values in {-1, 0, 1}
                sgrad_min, sgrad_max = np.min(sgrad), np.max(sgrad)
                sgrad_abs = np.abs(sgrad)
                # Check if all values are either ±1 (abs=1) or 0
                is_valid = np.all((np.isclose(sgrad_abs, 1.0, atol=1e-6)) | (np.isclose(sgrad, 0.0, atol=1e-6)))
                if not is_valid:
                    # sgrad should be ±1 (or 0 if grad is exactly 0)
                    # If not, there's a bug in sign computation
                    raise ValueError(f"sgrad should be ±1 or 0, but got range [{sgrad_min}, {sgrad_max}], unique values: {np.unique(sgrad)}")
                xi_repeated = np.repeat(xi, p, axis=1)  # (n0, cbSz*p)
                ri_repeated = np.repeat(ri, p, axis=1)  # (n0, cbSz*p)
                zi = xi_repeated + ri_repeated * sgrad
                # Verify zi is within [xi - ri, xi + ri] by construction
                # For each original batch entry, check all p candidates
                for batch_idx in range(cbSz):
                    xi_b = xi[:, batch_idx:batch_idx+1]  # (n0, 1)
                    ri_b = ri[:, batch_idx:batch_idx+1]  # (n0, 1)
                    zi_b_candidates = zi[:, batch_idx*p:(batch_idx+1)*p]  # (n0, p)
                    zi_lower = xi_b - ri_b  # (n0, 1)
                    zi_upper = xi_b + ri_b  # (n0, 1)
                    # Check all p candidates for this batch entry
                    # Compare each dimension: zi_b_candidates is (n0, p), zi_lower/upper are (n0, 1)
                    # Broadcasting: (n0, p) >= (n0, 1) -> (n0, p)
                    # Use tolerance for floating-point comparison (MATLAB uses double precision by default)
                    tol = 1e-6
                    violations_lower = zi_b_candidates < (zi_lower - tol)  # (n0, p)
                    violations_upper = zi_b_candidates > (zi_upper + tol)  # (n0, p)
                    # Check if any dimension violates bounds for any candidate
                    any_violation_lower = np.any(violations_lower)
                    any_violation_upper = np.any(violations_upper)
                    if any_violation_lower or any_violation_upper:
                        # This should never happen if sgrad = ±1
                        # Debug: show which dimensions violate
                        dims_violating_lower = np.where(np.any(violations_lower, axis=1))[0]
                        dims_violating_upper = np.where(np.any(violations_upper, axis=1))[0]
                        candidates_violating_lower = np.where(np.any(violations_lower, axis=0))[0]
                        candidates_violating_upper = np.where(np.any(violations_upper, axis=0))[0]
                        # Check sgrad for the violating candidates
                        sgrad_b = sgrad[:, batch_idx*p:(batch_idx+1)*p]  # (n0, p)
                        raise ValueError(
                            f"zi out of bounds for batch {batch_idx}:\n"
                            f"  violations_lower: dims={dims_violating_lower}, candidates={candidates_violating_lower}\n"
                            f"  violations_upper: dims={dims_violating_upper}, candidates={candidates_violating_upper}\n"
                            f"  sgrad range=[{sgrad_min}, {sgrad_max}]\n"
                            f"  For violating candidates, sgrad values: {sgrad_b[:, candidates_violating_lower] if len(candidates_violating_lower) > 0 else 'N/A'}\n"
                            f"  xi_b={xi_b.flatten()}, ri_b={ri_b.flatten()}\n"
                            f"  zi_lower={zi_lower.flatten()}, zi_upper={zi_upper.flatten()}\n"
                            f"  zi_b_candidates (first violating)={zi_b_candidates[:, candidates_violating_lower[0] if len(candidates_violating_lower) > 0 else 0].flatten()}"
                        )
        # 2.2. Check the specification for adversarial examples.
        # MATLAB: [~,critVal,falsified,x_,y_] = aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);
        # Match MATLAB aux_checkPoints exactly (lines 985-1004)
        yi = nn.evaluate_(zi, options, idxLayer)
        # MATLAB: ld_ys = A*ys;
        ld_yi = A @ yi  # logit difference: A*yi, shape (num_constraints, batch_size)
        # MATLAB: critValPerConstr = ld_ys - b;
        # b should already be reshaped to (num_constraints, 1) at function start
        critValPerConstr = ld_yi - b  # (num_constraints, batch_size)
        if safeSet:
            # MATLAB: falsified = any(ld_ys > b,1);
            # safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
            # Thus, unsafe if any(A*y > b).
            # MATLAB: any(ld_ys > b,1) checks along dimension 1 (constraints), so for each batch sample
            checkSpecs = np.any(ld_yi > b, axis=0)  # (batch_size,)
            critValPerConstr = -critValPerConstr
            critVal = np.min(critValPerConstr, axis=0)  # (batch_size,)
        else:
            # MATLAB: falsified = all(ld_ys <= b,1);
            # unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
            # Thus, unsafe if all(A*y <= b).
            # Note: For unsafe sets, if all constraints are satisfied (all A*y <= b), 
            # it means the property is NOT satisfied (it's unsafe), so we found a counterexample.
            # MATLAB: all(ld_ys <= b,1) checks along dimension 1 (constraints), so for each batch sample
            # ld_yi shape: (num_constraints, batch_size)
            # b shape: (num_constraints, 1) after reshaping
            # ld_yi <= b broadcasts to (num_constraints, batch_size)
            # np.all(..., axis=0) checks along constraint dimension, giving (batch_size,)
            comparison = ld_yi <= b  # (num_constraints, batch_size)
            checkSpecs = np.all(comparison, axis=0)  # (batch_size,)
            critVal = np.max(critValPerConstr, axis=0)  # (batch_size,)
            
            # Debug output for unsafe sets
            if verbose and np.any(checkSpecs):
                print(f"DEBUG UnsafeSet: ld_yi shape={ld_yi.shape}, b shape={b.shape}")
                print(f"DEBUG UnsafeSet: ld_yi={ld_yi.flatten()}")
                print(f"DEBUG UnsafeSet: b={b.flatten()}")
                print(f"DEBUG UnsafeSet: comparison (ld_yi <= b)={comparison.flatten()}")
                print(f"DEBUG UnsafeSet: checkSpecs={checkSpecs}")
                print(f"DEBUG UnsafeSet: critValPerConstr={critValPerConstr.flatten()}")
        
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
            # Debug first iteration to compare with MATLAB
            if np.sum(checkSpecs) == 0 and cbSz == 1:
                # First iteration, no counterexample found - compare with MATLAB
                print(f"DEBUG Adversarial FIRST ITER: ld_yi={ld_yi.flatten()}, b={b.flatten()}")
                print(f"DEBUG Adversarial FIRST ITER: ld_yi shape={ld_yi.shape}, b shape={b.shape}")
                print(f"DEBUG Adversarial FIRST ITER: A shape={A.shape}, yi shape={yi.shape}")
                print(f"DEBUG Adversarial FIRST ITER: A={A.flatten()}, yi={yi.flatten()}")
                zi_flat = zi.flatten() if isinstance(zi, np.ndarray) else zi.cpu().numpy().flatten()
                print(f"DEBUG Adversarial FIRST ITER: zi={zi_flat}")
                # Compute A @ yi manually to verify
                A_yi_manual = A @ yi
                print(f"DEBUG Adversarial FIRST ITER: A @ yi (manual)={A_yi_manual.flatten()}, matches ld_yi: {np.allclose(A_yi_manual, ld_yi)}")
                if not safeSet:
                    comparison_first = ld_yi <= b
                    print(f"DEBUG Adversarial FIRST ITER: comparison (ld_yi <= b)={comparison_first.flatten()}")
                    print(f"DEBUG Adversarial FIRST ITER: all(comparison)={np.all(comparison_first)}")
                    print(f"DEBUG Adversarial FIRST ITER: critValPerConstr={critValPerConstr.flatten()}")
                    print(f"DEBUG Adversarial FIRST ITER: critVal={critVal.flatten()}")
                    # Check if MATLAB would find this as counterexample
                    # MATLAB: falsified = all(ld_ys <= b, 1)
                    # In MATLAB, this checks along dimension 1 (constraints)
                    # For unsafe set: if all(ld_ys <= b) is True, then falsified = True
                    # So we need all constraints to satisfy ld_ys <= b
                    print(f"DEBUG Adversarial FIRST ITER: MATLAB logic: all(ld_ys <= b, 1) = {np.all(comparison_first, axis=0).flatten()}")
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
            # MATLAB: zi is within [xi - ri, xi + ri] by construction (xi_ = xi + ri * sgrad where sgrad = ±1)
            # So we don't need to validate bounds - if construction is correct, counterexample is valid
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
        # Handle both num_relu_constraints and num_relu_tighten_constraints (MATLAB test uses both)
        nReLU = options.get('nn', {}).get('num_relu_constraints', 0)
        if nReLU == 0:
            # Check for num_relu_tighten_constraints as alias (used in MATLAB tests)
            nReLU = options.get('nn', {}).get('num_relu_tighten_constraints', 0)
        # numInitGens is already defined before the loop (matching MATLAB line 79)
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
                # dimId has shape (batch,), reshape to (1, batch) to match MATLAB row vector
                dimId_row = dimId.reshape(1, -1)  # (1, batch)
                dimId_tiled = np.tile(dimId_row, (1, nSplits))  # (1, batch*nSplits)
                # MATLAB automatically pads dimIds to match number of columns when concatenating
                # Pad dimIds to match the number of columns in dimId_tiled
                if dimIds.shape[1] < dimId_tiled.shape[1]:
                    # Pad with NaN to match MATLAB behavior
                    pad_cols = dimId_tiled.shape[1] - dimIds.shape[1]
                    dimIds_padded = np.full((dimIds.shape[0], pad_cols), np.nan, dtype=dimIds.dtype)
                    dimIds = np.hstack([dimIds, dimIds_padded])
                dimIds = np.vstack([dimId_tiled, dimIds])
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
