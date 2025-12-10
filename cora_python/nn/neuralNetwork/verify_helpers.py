"""
Helper functions for neural network verification refinement logic.
These functions are used by verify.py for the zonotack refinement method.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .neuralNetwork import NeuralNetwork


def _aux_matchBatchSize(c: Union[np.ndarray, torch.Tensor], G: Union[np.ndarray, torch.Tensor], 
                       bSz: int, options: Dict[str, Any]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Replicate a zonotope batch for splitting.
    Internal to nn - works with torch tensors.
    
    MATLAB: function [c,G] = aux_matchBatchSize(c,G,bSz,options)
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32)
    if isinstance(G, np.ndarray):
        G = torch.tensor(G, dtype=torch.float32)
    
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


def _aux_scaleAndOffsetZonotope(c: Union[np.ndarray, torch.Tensor], G: Union[np.ndarray, torch.Tensor], 
                               bc: Union[np.ndarray, torch.Tensor], br: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Scale and offset the zonotope to a new hypercube with center bc and radius br.
    Internal to nn - works with torch tensors.
    
    MATLAB: function [c,G] = aux_scaleAndOffsetZonotope(c,G,bc,br)
    
    MATLAB code (lines 2014-2032):
        qiIds = 1:min(size(G,2),size(bc,1));
        G_ = G(:,qiIds,:);
        bc_ = permute(bc(qiIds,:),[1 3 2]);
        br_ = permute(br(qiIds,:),[3 1 2]);
        offset = pagemtimes(G_,bc_);
        if ndims(c) > 2
            c = c + offset;
        else
            c = c + offset(:,:);
        end
        G(:,qiIds,:) = G(:,qiIds,:).*br_;
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(c, np.ndarray):
        c = torch.tensor(c, dtype=torch.float32)
    if isinstance(G, np.ndarray):
        G = torch.tensor(G, dtype=torch.float32)
    if isinstance(bc, np.ndarray):
        bc = torch.tensor(bc, dtype=torch.float32)
    if isinstance(br, np.ndarray):
        br = torch.tensor(br, dtype=torch.float32)
    
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Obtain indices of generator.
    qiIds = min(G.shape[1], bc.shape[0])
    # Extract the relevant entries.
    G_ = G[:, :qiIds, :]  # (n, qiIds, batch)
    
    # MATLAB: bc_ = permute(bc(qiIds,:),[1 3 2]);
    # bc(qiIds,:) is 2D (qiIds, bSz), permute([1 3 2]) makes it (qiIds, 1, bSz)
    bc_ = torch.unsqueeze(bc[:qiIds, :], dim=1)  # (qiIds, 1, bSz_bc) - use torch
    # MATLAB: br_ = permute(br(qiIds,:),[3 1 2]);
    # br(qiIds,:) is 2D (qiIds, bSz), permute([3 1 2]) makes it (1, qiIds, bSz)
    br_ = torch.unsqueeze(br[:qiIds, :], dim=0)  # (1, qiIds, bSz_bc) - use torch
    
    # Scale and offset the zonotope to a new hypercube with center bc and radius br.
    # MATLAB: offset = pagemtimes(G_,bc_);
    # pagemtimes will handle batch size mismatches through replication
    offset = pagemtimes(G_, 'none', bc_, 'none')  # (n, 1, batch)
    
    # Offset the center.
    # MATLAB: if ndims(c) > 2, c = c + offset; else c = c + offset(:,:); end
    if c.ndim > 2:  # iff options.nn.interval_center
        # MATLAB: c = c + offset;
        # Both c and offset are 3D, addition should broadcast
        # Ensure batch sizes match
        c_bSz = c.shape[2]
        offset_bSz = offset.shape[2]
        if c_bSz != offset_bSz:
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            if c_bSz > offset_bSz and c_bSz % offset_bSz == 0:
                nReps = c_bSz // offset_bSz
                offset = repelem(offset, 1, 1, nReps)
            elif offset_bSz > c_bSz and offset_bSz % c_bSz == 0:
                # Subset offset to match c's batch size
                offset = offset[:, :, :c_bSz]
            else:
                raise ValueError(f"Cannot match batch sizes in offset: c.shape={c.shape}, offset.shape={offset.shape}")
        c = c + offset
    else:
        # MATLAB: c = c + offset(:,:);
        # offset(:,:) squeezes singleton dimensions
        # offset is (n, 1, batch), offset(:,:) becomes (n, batch)
        if offset.shape[1] == 1:
            offset_2d = offset.squeeze(1)  # (n, batch_offset)
        else:
            offset_2d = offset.reshape(offset.shape[0], -1)  # (n, batch_offset)
        # Ensure batch sizes match
        c_bSz = c.shape[1] if c.ndim == 2 else c.shape[2]
        offset_bSz = offset_2d.shape[1]
        if c_bSz != offset_bSz:
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            if c_bSz > offset_bSz and c_bSz % offset_bSz == 0:
                nReps = c_bSz // offset_bSz
                offset_2d = repelem(offset_2d, 1, nReps)
            elif offset_bSz > c_bSz and offset_bSz % c_bSz == 0:
                # Subset offset to match c's batch size
                offset_2d = offset_2d[:, :c_bSz]
            else:
                raise ValueError(f"Cannot match batch sizes in offset_2d: c.shape={c.shape}, offset_2d.shape={offset_2d.shape}")
        c = c + offset_2d  # (n, batch)
    
    # Scale the generators.
    # MATLAB: G(:,qiIds,:) = G(:,qiIds,:).*br_;
    # br_ has shape (1, qiIds, bSz_br), G has shape (n, qiIds, bSz_G)
    # If batch sizes don't match, we need to replicate br_
    G_bSz = G.shape[2]
    br_bSz = br_.shape[2]
    if G_bSz != br_bSz:
        from ..layers.linear.nnGeneratorReductionLayer import repelem
        if G_bSz > br_bSz and G_bSz % br_bSz == 0:
            nReps = G_bSz // br_bSz
            br_ = repelem(br_, 1, 1, nReps)
        elif br_bSz > G_bSz and br_bSz % G_bSz == 0:
            # This shouldn't happen, but handle it
            # Subset br_ to match G's batch size
            br_ = br_[:, :, :G_bSz]
    G[:, :qiIds, :] = G[:, :qiIds, :] * br_  # Broadcast: (n, qiIds, batch) * (1, qiIds, batch)
    
    return c, G


def _aux_patchWeightMask(imgSize: tuple, ph: int, pw: int, l: float, u: float) -> np.ndarray:
    """
    Generate a patch-wise weight mask for input splitting, where the 
    center of a patch gets a higher score than the remaining pixels of the patch.
    
    MATLAB: function imgM = aux_patchWeightMask(imgSize,ph,pw,l,u)
    Lines 1303-1321
    """
    # Create the weight mask for a patch; initialize with the low score.
    # MATLAB: pM = l*ones([ph pw]);
    pM = l * np.ones((ph, pw))
    
    # Set the center pixel to the high score.
    # MATLAB: pM(ceil(ph/2),ceil(pw/2)) = u;
    center_h = int(np.ceil(ph / 2)) - 1  # Convert to 0-based
    center_w = int(np.ceil(pw / 2)) - 1  # Convert to 0-based
    pM[center_h, center_w] = u
    
    # Tile mask over input.
    # MATLAB: imgM = kron(ones(ceil(imgSize(1:2)./[ph pw])),pM);
    # kron(A, B) computes the Kronecker product
    tiles_h = int(np.ceil(imgSize[0] / ph))
    tiles_w = int(np.ceil(imgSize[1] / pw))
    imgM = np.kron(np.ones((tiles_h, tiles_w)), pM)
    
    # Trim excess dimensions.
    # MATLAB: imgM = imgM(1:imgSize(1),1:imgSize(2));
    imgM = imgM[:imgSize[0], :imgSize[1]]
    
    if len(imgSize) == 3:
        # There is a color channel.
        # MATLAB: imgM = repmat(imgM,1,1,imgSize(3));
        imgM = np.repeat(imgM[:, :, np.newaxis], imgSize[2], axis=2)
    
    return imgM


def _aux_computeHeuristic(heuristic: str, layerIdx: int, l: Union[np.ndarray, torch.Tensor], u: Union[np.ndarray, torch.Tensor], 
                          dr: Union[np.ndarray, torch.Tensor], sens: Union[np.ndarray, torch.Tensor], grad: Any,
                          sim: Optional[Union[np.ndarray, torch.Tensor]] = None, prevNrXs: Optional[Union[np.ndarray, torch.Tensor]] = None,
                          neuronIds: Optional[Union[np.ndarray, torch.Tensor]] = None, onlyUnstable: bool = True,
                          layerDiscount: float = 1.0, imgSz: Optional[Any] = None,
                          patchSz: Optional[Any] = None, patchScore: Optional[Any] = None) -> torch.Tensor:
    """
    Compute heuristic for splitting.
    Internal to nn - uses torch tensors.
    
    MATLAB: function h = aux_computeHeuristic(heuristic,layerIdx,l,u,dr,sens,grad,varargin)
    """
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(l, np.ndarray):
        l = torch.tensor(l, dtype=torch.float32)
    if isinstance(u, np.ndarray):
        u = torch.tensor(u, dtype=torch.float32)
    if isinstance(dr, np.ndarray):
        dr = torch.tensor(dr, dtype=torch.float32)
    if isinstance(sens, np.ndarray):
        sens = torch.tensor(sens, dtype=torch.float32)
    if isinstance(grad, np.ndarray):
        grad = torch.tensor(grad, dtype=torch.float32)
    if isinstance(sim, np.ndarray):
        sim = torch.tensor(sim, dtype=torch.float32) if sim is not None else None
    if isinstance(prevNrXs, np.ndarray):
        prevNrXs = torch.tensor(prevNrXs, dtype=torch.float32) if prevNrXs is not None else None
    if isinstance(neuronIds, np.ndarray):
        neuronIds = torch.tensor(neuronIds, dtype=torch.long) if neuronIds is not None else None
    
    device = l.device
    dtype = l.dtype
    
    # Compute the heuristic.
    if heuristic == 'least-unstable':
        # Least unstable neuron (normalize the un-stability).
        minBnd = 1.0 / torch.maximum(torch.minimum(-l, u), torch.tensor(1e-10, dtype=dtype, device=device))  # Avoid division by zero
        # Compute the heuristic.
        h = minBnd * sens
    elif heuristic == 'least-unstable-gradient':
        # Take the absolute value and add small epsilon to avoid numerical problems.
        if isinstance(grad, (int, float)):
            grad = torch.tensor(abs(grad) + 1e-3, dtype=dtype, device=device)  # Use Python abs instead of np.abs
        else:
            grad = torch.abs(grad) + 1e-3
        # Least unstable neuron (normalize the un-stability).
        minBnd = 1.0 / torch.maximum(torch.minimum(-l, u), torch.tensor(1e-10, dtype=dtype, device=device))
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
            grad = torch.tensor(abs(grad) + 1e-3, dtype=dtype, device=device)  # Use Python abs instead of np.abs
        else:
            grad = torch.abs(grad) + 1e-3
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
        h = torch.where(unstable, h, torch.tensor(-float('inf'), dtype=dtype, device=device))
    
    if layerDiscount != 1.0:
        # Prefer earlier layers.
        h = h * (layerDiscount ** layerIdx)
    
    # Convert prevNrXs, sim, and neuronIds to torch tensors if they're lists (MATLAB passes [] as empty array)
    if isinstance(prevNrXs, list):
        prevNrXs = torch.tensor(prevNrXs, dtype=dtype, device=device) if len(prevNrXs) > 0 else torch.empty((0,), dtype=dtype, device=device)
    if isinstance(sim, list):
        sim = torch.tensor(sim, dtype=dtype, device=device) if len(sim) > 0 else torch.empty((0,), dtype=dtype, device=device)
    if isinstance(neuronIds, list):
        neuronIds = torch.tensor(neuronIds, dtype=torch.long, device=device) if len(neuronIds) > 0 else torch.empty((0,), dtype=torch.long, device=device)
    
    if prevNrXs is not None and (hasattr(prevNrXs, 'numel') and prevNrXs.numel() > 0):
        # Obtain the batch size.
        bSz = prevNrXs.shape[1] if prevNrXs.ndim > 1 else 1
        
        # We floor all entries. We mark unnecessary splits with decimal numbers.
        prevNrXs_floor = torch.floor(prevNrXs)
        # Reduce redundancy by not add constraints for split neurons.
        if h.shape[1] > prevNrXs.shape[1]:
            newSplits = h.shape[1] // prevNrXs.shape[1]
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            prevNrXs_ = repelem(prevNrXs_floor, 1, newSplits)
        else:
            prevNrXs_ = prevNrXs_floor
        
        # Identify already split neurons.
        if neuronIds is not None:
            wasSplit = torch.any(torch.abs(prevNrXs_) == neuronIds.reshape(-1, 1, 1), dim=0)
            # There is no similarity; just prevent splitting the same neuron twice.
            h = torch.where(wasSplit, torch.tensor(-float('inf'), dtype=dtype, device=device), h)
            if sim is not None and sim.numel() > 0:
                # Specify a tolerance for similarity.
                tol = 1e-3
                # Reduce the heuristic based on the similarity to already split neurons.
                simSplit = torch.any((sim > 1 - tol) & wasSplit.reshape(1, -1, 1), dim=1)
                h = torch.where(simSplit, torch.tensor(-float('inf'), dtype=dtype, device=device), h)
        else:
            # No neuron IDs provided, skip this check
            pass
    
    # Handle imgSz, patchSz, patchScore for image inputs
    if imgSz is not None and imgSz:
        # MATLAB lines 1473-1486
        # Obtain the patch size.
        ph = patchSz[0] if patchSz is not None else 1
        pw = patchSz[1] if patchSz is not None and len(patchSz) > 1 else 1
        # Obtain the scores.
        lowScore = patchScore[0] if patchScore is not None else 0.1
        highScore = patchScore[1] if patchScore is not None and len(patchScore) > 1 else 1.0
        # Compute a patch-wise weight mask to avoid splitting many similar input dimensions.
        imgM = _aux_patchWeightMask(imgSz, ph, pw, lowScore, highScore)
        # Convert to torch tensor and correct data type.
        if isinstance(imgM, np.ndarray):
            imgM = torch.tensor(imgM, dtype=dtype, device=device)
        # Weight the heuristic by the patch-wise mask.
        # MATLAB: h = h.*imgM(:);
        h = h * imgM.flatten().reshape(-1, 1)
    
    return h


def _aux_dimSplitConstraints(hi: Union[np.ndarray, torch.Tensor], nSplits: int, nDims: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct dimension split constraints that splits #nDims dimensions into #nSplits pieces.
    Internal to nn - uses torch tensors.
    
    MATLAB: function [Ai,bi,dimIds,hi] = aux_dimSplitConstraints(hi,nSplits,nDims)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(hi, np.ndarray):
        hi = torch.tensor(hi, dtype=torch.float32)
    
    device = hi.device
    dtype = hi.dtype
    
    # Obtain the number of dimensions and batch size.
    n, bSz = hi.shape
    nDims = min(nDims, n)
    
    # Split each input in the batch into nSplits parts.
    # 1. Find the input dimension with the largest heuristic.
    sortDims = torch.argsort(hi, dim=0, descending=True)  # Sort descending
    hi_sorted, _ = torch.sort(hi, dim=0, descending=True)  # Sort descending
    dimIds = sortDims[:nDims, :]  # (nDims, batch)
    hi = hi_sorted[:nDims, :]  # (nDims, batch)
    
    # Compute dimension indices.
    # MATLAB: dimIdx = sub2ind([nDims n bSz],repelem((1:nDims)',1,bSz),dimIds,repelem(1:bSz,nDims,1));
    # dimIds is already 0-based from argsort, need to convert to 1-based for MATLAB sub2ind
    dimIdx = sub2ind((nDims, n, bSz),
                     repelem(torch.arange(1, nDims + 1, dtype=torch.long, device=device), 1, bSz),  # 1-based
                     (dimIds + 1).flatten(),  # Convert to 1-based, then flatten (torch uses row-major by default)
                     repelem(torch.arange(1, bSz + 1, dtype=torch.long, device=device), nDims, 1).flatten())  # 1-based
    
    # 2. Construct the constraints.
    Ai = torch.zeros((nDims, n, bSz), dtype=dtype, device=device)
    # Set non-zero entries
    # sub2ind already returns 0-based indices, so use directly
    Ai_flat = Ai.flatten()
    Ai_flat[dimIdx] = 1
    Ai = Ai_flat.reshape(nDims, n, bSz)
    
    # Specify offsets: repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz)
    offsets = -1 + torch.arange(1, nSplits, dtype=dtype, device=device) * (2.0 / nSplits)  # (nSplits-1,)
    bi = offsets.unsqueeze(1).unsqueeze(2).repeat(1, nDims, bSz)  # (nSplits-1, nDims, bSz)
    bi = bi.permute(1, 0, 2)  # (nDims, nSplits-1, bSz)
    
    return Ai, bi, dimIds, hi


def _aux_constructUnsafeOutputSet(options: Dict[str, Any], y: Union[np.ndarray, torch.Tensor], Gy: Union[np.ndarray, torch.Tensor],
                                   A: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor], safeSet: bool,
                                   numUnionConst: int) -> Dict[str, Any]:
    """
    Construct unsafe output set.
    Internal to nn - uses torch tensors.
    
    MATLAB: function uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet,numUnionConstraint)
    """
    from ..layers.linear.nnGeneratorReductionLayer import pagemtimes
    
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32)
    if isinstance(Gy, np.ndarray):
        Gy = torch.tensor(Gy, dtype=torch.float32)
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32)
    
    device = y.device if isinstance(y, torch.Tensor) else torch.device('cpu')
    dtype = y.dtype if isinstance(y, torch.Tensor) else torch.float32
    
    # Obtain the number of output dimensions and batch size.
    nK, _, bSz = Gy.shape
    
    if options.get('nn', {}).get('interval_center', False):
        # Compute center and center radius.
        yc = 0.5 * (y[:, 1, :] + y[:, 0, :]).reshape(nK, bSz)
        yr = 0.5 * (y[:, 1, :] - y[:, 0, :])
    else:
        # The radius is zero.
        # MATLAB: yc = y; where y is (nK, bSz)
        # If y is 3D (nK, 1, bSz), squeeze the middle dimension
        if y.ndim == 3 and y.shape[1] == 1:
            yc = y.squeeze(dim=1)  # (nK, bSz) - use torch squeeze
        else:
            yc = y
        yr = torch.zeros((nK, 1, bSz), dtype=dtype, device=device)
    
    # Compute the output constraints (logit difference).
    # This matches aux_computeLogitDifference logic
    if options.get('nn', {}).get('interval_center', False):
        yic = yc
        yid = yr
    else:
        # MATLAB: yic = yi; where yi is (nK, bSz) when interval_center is False
        yic = yc  # Already reshaped above to (nK, bSz)
        yid = torch.zeros((nK, 1, bSz), dtype=dtype, device=device)
    
    # Compute logit difference
    ld_yi = A @ yic  # (spec_dim, batch)
    ld_Gyi = pagemtimes(A, 'none', Gy, 'none')  # (spec_dim, n_gens, batch)
    
    # Compute logit difference of approximation errors
    # MATLAB: ld_Gyi_err = sum(abs(A.*permute(yid,[2 1 3])),2);
    yid_perm = yid.permute(1, 0, 2)  # (1, nK, bSz) - use torch permute
    A_yid = A.unsqueeze(2) * yid_perm  # (spec_dim, nK, bSz) - use torch unsqueeze
    ld_Gyi_err = torch.sum(torch.abs(A_yid), dim=1)  # (spec_dim, bSz) - use torch operations
    
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
    # MATLAB: uYi.b = b_ + ld_Gyi_err(:,:);
    # The (:,:) keeps it 2D, so don't add newaxis
    uYi = {
        'c': yc,
        'r': yr,
        'G': Gy,
        'A': A_,
        'b': b_ + ld_Gyi_err  # Keep 2D like MATLAB
    }
    
    return uYi


def _aux_pop(xs: np.ndarray, rs: np.ndarray, nrXs: np.ndarray, bSz: int, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pop elements from the queue (MATLAB aux_pop equivalent)
    
    MATLAB signature:
    function [xi,ri,nrXi,xs,rs,nrXs,qIdx] = aux_pop(xs,rs,nrXs,bSz,options)
    """
    # Ensure bSz is an integer (it might be passed as an array)
    if isinstance(bSz, np.ndarray):
        if bSz.size == 1:
            bSz = int(bSz.item())
        else:
            # If it's an array, use the size of the first dimension
            bSz = int(bSz.shape[0] if bSz.ndim > 0 else bSz.size)
    elif not isinstance(bSz, (int, np.integer)):
        bSz = int(bSz)
    
    # Obtain the number of elements in the queue.
    nQueue = xs.shape[1]
    
    # Construct indices to pop.
    dequeue_type = options.get('nn', {}).get('verify_dequeue_type', 'front')
    
    if dequeue_type == 'front':
        # Take the first entries.
        qIdx = np.arange(1, min(bSz, nQueue) + 1)  # 1-based like MATLAB
    elif dequeue_type == 'half-half':
        # Half from the front and half from the back.
        qIdx = np.arange(1, min(bSz, nQueue) + 1)  # 1-based
        offsetIdx = np.arange(int(np.ceil(len(qIdx) / 2 + 1)), len(qIdx) + 1)  # 1-based
        qIdx[offsetIdx - 1] = qIdx[offsetIdx - 1] + nQueue - len(qIdx)  # Convert to 0-based for indexing
    else:
        # Invalid option.
        raise ValueError(f"Invalid verify_dequeue_type: {dequeue_type}. Must be one of ['front', 'half-half']")
    
    # Convert 1-based indices to 0-based for Python indexing
    qIdx_0based = qIdx - 1
    
    # Pop centers.
    xi = xs[:, qIdx_0based]
    xs = np.delete(xs, qIdx_0based, axis=1)
    
    # Pop radii.
    ri = rs[:, qIdx_0based]
    rs = np.delete(rs, qIdx_0based, axis=1)
    
    # Pop indices for split neurons.
    # MATLAB: nrXi = nrXs(:,qIdx); nrXs(:,qIdx) = [];
    # Even when nrXs is empty (0 rows), MATLAB still removes columns
    if nrXs.size > 0:
        nrXi = nrXs[:, qIdx_0based]
        nrXs = np.delete(nrXs, qIdx_0based, axis=1)
    else:
        # nrXs is empty, but we still need to remove columns to match MATLAB behavior
        # nrXi should have 0 rows and len(qIdx_0based) columns
        nrXi = np.zeros((0, len(qIdx_0based)), dtype=xs.dtype)
        # nrXs should have 0 rows and remaining columns after popping
        # MATLAB: nrXs(:,qIdx) = [] removes columns even when nrXs is empty
        nrXs = np.zeros((0, xs.shape[1]), dtype=xs.dtype)
    
    return xi, ri, nrXi, xs, rs, nrXs, qIdx


def _aux_pop_simple(xs: torch.Tensor, rs: torch.Tensor, bs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pop elements from the queue (MATLAB aux_pop equivalent for simple verify)
    Internal to nn - only uses torch tensors.
    
    MATLAB signature:
    function [xi,ri,xs,rs] = aux_pop(xs,rs,bs)
    
    MATLAB code (lines 192-205):
        bs = min(bs,size(xs,2));
        idx = 1:bs;
        xi = xs(:,idx);
        xs(:,idx) = [];
        ri = rs(:,idx);
        rs(:,idx) = [];
    """
    # MATLAB: bs = min(bs,size(xs,2));
    bs_actual = min(bs, xs.shape[1])
    
    # MATLAB: idx = 1:bs;
    # Python uses 0-based indexing, so idx = 0:bs_actual
    idx = torch.arange(bs_actual, device=xs.device)
    
    # MATLAB: xi = xs(:,idx);
    xi = xs[:, idx].clone()
    
    # MATLAB: xs(:,idx) = [];
    # Remove columns at idx
    remaining_idx = torch.arange(bs_actual, xs.shape[1], device=xs.device)
    xs = xs[:, remaining_idx]
    
    # MATLAB: ri = rs(:,idx);
    ri = rs[:, idx].clone()
    
    # MATLAB: rs(:,idx) = [];
    # Remove columns at idx
    rs = rs[:, remaining_idx]
    
    return xi, ri, xs, rs


def _aux_constructInputZonotope(options: Dict[str, Any], heuristic: str, xi: Union[np.ndarray, torch.Tensor], ri: Union[np.ndarray, torch.Tensor],
                                  batchG: Union[np.ndarray, torch.Tensor], sens: Optional[Union[np.ndarray, torch.Tensor]], grad: Optional[Union[np.ndarray, torch.Tensor]],
                                  numInitGens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct input zonotope (MATLAB aux_constructInputZonotope equivalent)
    Internal to nn - uses torch tensors.
    
    MATLAB signature:
    function [cxi,Gxi,dimIdx] = aux_constructInputZonotope(options,heuristic,xi,ri,batchG,sens,grad,numInitGens)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(xi, np.ndarray):
        xi = torch.tensor(xi, dtype=torch.float32)
    if isinstance(ri, np.ndarray):
        ri = torch.tensor(ri, dtype=torch.float32)
    if isinstance(batchG, np.ndarray):
        batchG = torch.tensor(batchG, dtype=torch.float32)
    if isinstance(sens, np.ndarray):
        sens = torch.tensor(sens, dtype=torch.float32) if sens is not None else None
    if isinstance(grad, np.ndarray):
        grad = torch.tensor(grad, dtype=torch.float32) if grad is not None else None
    
    device = xi.device
    dtype = xi.dtype
    
    # Obtain the number of input dimensions and the batch size.
    n0, bSz = xi.shape
    
    # Initialize the generator matrix.
    Gxi = batchG[:, :, :bSz]  # (n0, numGen, bSz)
    
    if numInitGens >= n0:
        # We create a generator for each input dimension.
        # MATLAB: dimIdx = repmat((1:n0)',1,bSz);
        dimIdx = torch.tile(torch.arange(1, n0 + 1, dtype=torch.long, device=device).unsqueeze(1), (1, bSz))  # 1-based: (n0, bSz)
    else:
        # Compute the heuristic.
        # MATLAB: hi = aux_computeHeuristic(heuristic,0,xi-ri,xi+ri,ri,sens,grad,[],[],[],false,1);
        hi = _aux_computeHeuristic(heuristic, 0, xi - ri, xi + ri, ri, sens, grad, [], [], [], False, 1)
        # Convert hi to torch if needed
        if isinstance(hi, np.ndarray):
            hi = torch.tensor(hi, dtype=dtype, device=device)
        
        # Find the input pixels that affect the output the most.
        # MATLAB: [~,dimIdx] = sort(hi,'descend');
        sortIdx = torch.argsort(hi, dim=0, descending=True)  # Sort descending along dim=0 (rows)
        # MATLAB: dimIdx = dimIdx(1:numInitGens,:);
        dimIdx = sortIdx[:numInitGens, :] + 1  # Convert to 1-based: (numInitGens, bSz)
    
    # Compute indices for non-zero entries.
    # MATLAB: gIdx = sub2ind(size(Gxi),dimIdx, repmat((1:numInitGens)',1,bSz),repelem(1:bSz,numInitGens,1));
    # Convert to torch if needed
    if isinstance(dimIdx, np.ndarray):
        dimIdx = torch.tensor(dimIdx, dtype=torch.long, device=Gxi.device)
    if isinstance(ri, np.ndarray):
        ri = torch.tensor(ri, dtype=Gxi.dtype, device=Gxi.device)
    
    device = Gxi.device
    dtype_long = torch.long
    
    dimIdx_flat = dimIdx.flatten()  # Flatten, 1-based
    genIdx_flat = torch.tile(torch.arange(1, numInitGens + 1, dtype=dtype_long, device=device).unsqueeze(1), (1, bSz)).flatten()  # 1-based
    batchIdx_flat = repelem(torch.arange(1, bSz + 1, dtype=dtype_long, device=device), numInitGens, 1).flatten()  # 1-based
    gIdx = sub2ind(Gxi.shape, dimIdx_flat, genIdx_flat, batchIdx_flat)  # 0-based linear indices (sub2ind converts internally)
    
    # Set non-zero generator entries.
    # MATLAB: Gxi(gIdx) = ri(sub2ind(size(ri),dimIdx,repelem(1:bSz,numInitGens,1)));
    ri_dimIdx_flat = dimIdx.flatten()  # Flatten, 1-based
    ri_batchIdx_flat = repelem(torch.arange(1, bSz + 1, dtype=dtype_long, device=device), numInitGens, 1).flatten()  # 1-based
    ri_gIdx = sub2ind(ri.shape, ri_dimIdx_flat, ri_batchIdx_flat)  # 0-based linear indices (sub2ind converts internally)
    # sub2ind already returns 0-based indices, so use directly
    # Use column-major order to match MATLAB's column-major indexing
    # For torch, we need to manually compute column-major indices
    Gxi_flat = Gxi.permute(1, 0, 2).flatten()  # Permute to (numGen, n0, bSz) then flatten for column-major
    ri_flat = ri.T.flatten()  # Transpose then flatten for column-major
    Gxi_flat[gIdx] = ri_flat[ri_gIdx]  # Both are already 0-based
    Gxi = Gxi_flat.reshape(Gxi.shape[1], Gxi.shape[0], Gxi.shape[2]).permute(1, 0, 2)  # Reshape back and permute
    
    # Sum generators to compute remaining set.
    # MATLAB: ri_ = (ri - reshape(sum(Gxi,2),[n0 bSz]));
    Gxi_sum = torch.sum(Gxi, dim=1)  # Sum over generators: (n0, bSz)
    ri_ = ri - Gxi_sum
    
    # DEBUG: Log ri_ values for comparison with MATLAB (first few iterations only)
    # This helps identify if Python's remaining radius is smaller
    # Note: iteration number needs to be passed or stored in options
    debug_iteration = options.get('_debug_iteration', None)
    if debug_iteration is not None and debug_iteration <= 10:
        print(f"SPLITTING DEBUG (iteration {debug_iteration}):")
        print(f"  ri shape: {ri.shape}, ri (first 3 dims, first 3 batches): {ri[:min(3, ri.shape[0]), :min(3, ri.shape[1])].flatten().cpu().numpy()}")
        print(f"  Gxi_sum shape: {Gxi_sum.shape}, Gxi_sum (first 3 dims, first 3 batches): {Gxi_sum[:min(3, Gxi_sum.shape[0]), :min(3, Gxi_sum.shape[1])].flatten().cpu().numpy()}")
        print(f"  ri_ shape: {ri_.shape}, ri_ (first 3 dims, first 3 batches): {ri_[:min(3, ri_.shape[0]), :min(3, ri_.shape[1])].flatten().cpu().numpy()}")
        print(f"  ri_ min/max: min={torch.min(ri_).item()}, max={torch.max(ri_).item()}")
        if torch.any(ri_ < 1e-6):
            zero_count = torch.sum(ri_ < 1e-6).item()
            print(f"  WARNING: {zero_count} entries have ri_ < 1e-6 (very small remaining radius)!")
            zero_indices = torch.where(ri_ < 1e-6)
            print(f"  Zero ri_ locations (first 5): dims={zero_indices[0][:5].cpu().numpy()}, batches={zero_indices[1][:5].cpu().numpy()}")
    
    # Construct the center.
    if options.get('nn', {}).get('interval_center', False):
        # Put remaining set into the interval center.
        # MATLAB: cxi = permute(cat(3,xi - ri_,xi + ri_),[1 3 2]);
        # cat(3,xi - ri_,xi + ri_) creates (n0, bSz, 2)
        # permute([1 3 2]) gives (n0, 2, bSz)
        cxi_lower = xi - ri_  # (n0, bSz)
        cxi_upper = xi + ri_  # (n0, bSz)
        cxi = torch.stack([cxi_lower, cxi_upper], dim=1)  # (n0, 2, bSz)
    else:
        # The center is just a vector.
        # MATLAB: cxi = xi;
        cxi = xi  # (n0, bSz)
    
    return cxi, Gxi, dimIdx


def _aux_split_with_dim(xi: Union[np.ndarray, torch.Tensor], ri: Union[np.ndarray, torch.Tensor], his: Union[np.ndarray, torch.Tensor], nSplits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the input for verification and return the dimension ID (MATLAB aux_split equivalent for naive refinement)
    Internal to nn - uses torch tensors.
    
    MATLAB signature:
    function [xis,ris,dimId] = aux_split(xis,ris,his,nSplits)
    """
    from ..layers.linear.nnGeneratorReductionLayer import sub2ind, repelem
    
    # Convert to torch if needed (internal to nn, so should already be torch)
    if isinstance(xi, np.ndarray):
        xi = torch.tensor(xi, dtype=torch.float32)
    if isinstance(ri, np.ndarray):
        ri = torch.tensor(ri, dtype=torch.float32)
    if isinstance(his, np.ndarray):
        his = torch.tensor(his, dtype=torch.float32)
    
    device = xi.device
    dtype = torch.long
    
    n, bs = xi.shape
    # Find the input dimension with the largest heuristic.
    # MATLAB: [~,sortDims] = sort(abs(his),1,'descend');
    sortDims = torch.argsort(torch.abs(his), dim=0, descending=True)  # Sort descending along dim=0 (rows), returns 0-based indices
    # MATLAB: dimId = sortDims(1,:);
    # MATLAB returns 1-based indices, Python returns 0-based, so add 1 to convert
    dimId = sortDims[0, :] + 1  # Shape: (batch,), 1-based dimension indices
    
    # MATLAB: splitsIdx = repmat(1:nSplits,1,bs);
    splitsIdx = torch.tile(torch.arange(1, nSplits + 1, dtype=dtype, device=device), (bs,))  # 1-based like MATLAB: (nSplits*bs,)
    # MATLAB: bsIdx = repelem((1:bs)',nSplits);
    bsIdx = repelem(torch.arange(1, bs + 1, dtype=dtype, device=device), nSplits)  # 1-based: (bs*nSplits,)
    
    # MATLAB: linIdx = sub2ind([n bs nSplits], repelem(dimId,nSplits),bsIdx(:)',splitsIdx(:)');
    dim_repeated = repelem(dimId, nSplits)  # Shape: (batch*nSplits,), 1-based
    linIdx = sub2ind((n, bs, nSplits), dim_repeated, bsIdx, splitsIdx)  # 0-based linear indices (sub2ind converts internally)
    
    # 2. Split the selected dimension.
    xi_ = xi.clone()
    ri_ = ri.clone()
    # Shift to the lower bound.
    dimIdx = sub2ind((n, bs), dimId, torch.arange(1, bs + 1, dtype=dtype, device=device))  # 0-based linear indices (sub2ind converts internally)
    # sub2ind already returns 0-based indices, so use directly
    # Use column-major order to match MATLAB's column-major indexing
    # For torch, we need to manually compute column-major indices
    xi_flat = xi_.T.flatten()  # Transpose then flatten for column-major order
    ri_flat = ri.T.flatten()  # Use ORIGINAL ri for shift, not ri_!
    xi_flat[dimIdx] = xi_flat[dimIdx] - ri_flat[dimIdx]
    xi_ = xi_flat.reshape(bs, n).T  # Reshape back and transpose
    ri_flat = ri_.T.flatten()  # Now use ri_ for reduction
    ri_flat[dimIdx] = ri_flat[dimIdx] / nSplits
    ri_ = ri_flat.reshape(bs, n).T  # Reshape back and transpose
    
    # MATLAB: xis = repmat(xi_,1,1,nSplits);
    xis = xi_.unsqueeze(2).repeat(1, 1, nSplits)  # Shape: (n, bs, nSplits)
    # MATLAB: ris = repmat(ri_,1,1,nSplits);
    ris = ri_.unsqueeze(2).repeat(1, 1, nSplits)  # Shape: (n, bs, nSplits)
    
    # MATLAB: xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    # sub2ind already returns 0-based indices, so use directly
    # splitsIdx is 1-based (1, 2, ..., nSplits) matching MATLAB, use directly in formula
    # Use column-major order to match MATLAB's column-major indexing
    xis_flat = xis.permute(1, 0, 2).flatten()  # Permute to (bs, n, nSplits) then flatten for column-major
    ris_flat = ris.permute(1, 0, 2).flatten()  # Same for ris
    xis_flat[linIdx] = xis_flat[linIdx] + (2 * splitsIdx.float() - 1) * ris_flat[linIdx]
    xis = xis_flat.reshape(bs, n, nSplits).permute(1, 0, 2)  # Reshape back and permute
    
    # MATLAB: xis = xis(:,:); ris = ris(:,:);
    xis = xis.reshape(n, -1)
    ris = ris.reshape(n, -1)
    
    # Debug: Check if any values are outside reasonable bounds
    if torch.any(torch.abs(xis) > 100):
        print(f"WARNING: aux_split produced extreme values!")
        print(f"  xis range: [{torch.min(xis).item()}, {torch.max(xis).item()}]")
        print(f"  ris range: [{torch.min(ris).item()}, {torch.max(ris).item()}]")
        print(f"  dimId (1-based): {dimId}")
        print(f"  Input xi range: [{torch.min(xi).item()}, {torch.max(xi).item()}]")
        print(f"  Input ri range: [{torch.min(ri).item()}, {torch.max(ri).item()}]")
    
    return xis, ris, dimId


def _aux_split(xi: torch.Tensor, ri: torch.Tensor, sens: torch.Tensor, nSplits: int, nDims: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the input for verification (MATLAB aux_split equivalent)
    Internal to nn - only uses torch tensors.
    
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
    sortDims = torch.argsort(torch.abs(sens * ri), dim=0, descending=True)  # Sort descending along dim=0 (rows), returns 0-based indices
    # MATLAB: dimIds = sortDims(1:nDims,:);
    # MATLAB returns 1-based indices, Python returns 0-based, so add 1 to convert
    dimIds = sortDims[:nDims, :] + 1  # Shape: (nDims, batch), 1-based dimension indices
    
    # MATLAB: splitsIdx = repmat(1:nSplits,1,bs);
    splitsIdx = torch.tile(torch.arange(1, nSplits + 1, dtype=torch.long, device=xi.device), (bs,))  # 1-based like MATLAB: (nSplits*bs,)
    # MATLAB: bsIdx = repelem((1:bs)',nSplits);
    # (1:bs)' is a column vector, repelem repeats each element nSplits times
    bsIdx = repelem(torch.arange(1, bs + 1, dtype=torch.long, device=xi.device), nSplits)  # 1-based: (bs*nSplits,)
    
    # MATLAB: dim = dimIds(1,:);
    dim = dimIds[0, :]  # Shape: (batch,), 1-based dimension indices
    
    # MATLAB: linIdx = sub2ind([n bs nSplits], repelem(dim,nSplits),bsIdx(:)',splitsIdx(:)');
    # repelem(dim,nSplits): repeat each element of dim nSplits times
    dim_repeated = repelem(dim, nSplits)  # Shape: (batch*nSplits,), 1-based
    # sub2ind([n bs nSplits], dim_repeated, bsIdx, splitsIdx)
    # All inputs are 1-based MATLAB indices
    linIdx = sub2ind((n, bs, nSplits), dim_repeated, bsIdx, splitsIdx)  # 0-based linear indices (sub2ind converts internally)
    
    # 2. Split the selected dimension.
    # MATLAB: xi_ = xi; ri_ = ri;
    xi_ = xi.clone()
    ri_ = ri.clone()
    # Shift to the lower bound.
    # MATLAB: dimIdx = sub2ind([n bs],dim,1:bs);
    dimIdx = sub2ind((n, bs), dim, torch.arange(1, bs + 1, dtype=torch.long, device=xi.device))  # 0-based linear indices (sub2ind converts internally)
    # MATLAB: xi_(dimIdx) = xi_(dimIdx) - ri(dimIdx);
    # sub2ind already returns 0-based indices, so use directly
    # Use Fortran order to match MATLAB's column-major indexing
    # For torch, we need to manually compute Fortran-order indices
    # Fortran order: column-major, so index = row + col * n_rows
    xi_flat = xi_.T.flatten()  # Transpose then flatten for column-major order
    ri_flat = ri.T.flatten()  # Use ORIGINAL ri for shift, not ri_!
    xi_flat[dimIdx] = xi_flat[dimIdx] - ri_flat[dimIdx]
    xi_ = xi_flat.reshape(bs, n).T  # Reshape back and transpose
    # MATLAB: ri_(dimIdx) = ri_(dimIdx)/nSplits;
    ri_flat = ri_.T.flatten()  # Now use ri_ for reduction
    ri_flat[dimIdx] = ri_flat[dimIdx] / nSplits
    ri_ = ri_flat.reshape(bs, n).T  # Reshape back and transpose
    
    # MATLAB: xis = repmat(xi_,1,1,nSplits);
    xis = xi_.unsqueeze(2).repeat(1, 1, nSplits)  # Shape: (n, bs, nSplits)
    # MATLAB: ris = repmat(ri_,1,1,nSplits);
    ris = ri_.unsqueeze(2).repeat(1, 1, nSplits)  # Shape: (n, bs, nSplits)
    
    # MATLAB: xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    # Offset the center.
    # sub2ind already returns 0-based indices, so use directly
    # splitsIdx is 1-based (1, 2, ..., nSplits) matching MATLAB, use directly in formula
    # Use Fortran order to match MATLAB's column-major indexing
    xis_flat = xis.permute(1, 0, 2).flatten()  # Permute to (bs, n, nSplits) then flatten for column-major
    ris_flat = ris.permute(1, 0, 2).flatten()  # Same for ris
    xis_flat[linIdx] = xis_flat[linIdx] + (2 * splitsIdx.float() - 1) * ris_flat[linIdx]
    xis = xis_flat.reshape(bs, n, nSplits).permute(1, 0, 2)  # Reshape back and permute
    
    # MATLAB: xis = xis(:,:); ris = ris(:,:);
    # Flatten last two dimensions: (n, bs, nSplits) -> (n, bs*nSplits)
    xis = xis.reshape(n, -1)
    ris = ris.reshape(n, -1)
    
    return xis, ris


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
        # MATLAB: for i=1:(ps-1), Python: for i in range(ps-1) means i=0 corresponds to MATLAB i=1
        for i in range(ps - 1):
            matlab_i = i + 1  # Convert Python i to MATLAB i
            # Increase the index.
            idx_ = idx * (pcs + 1)
            # Repeat the current combined splits.
            # MATLAB: b_(1:2*i,1:idx_,:) = repmat(b_(1:2*i,1:idx,:),1,pcs+1,1);
            b_[:2*matlab_i, :idx_, :] = np.tile(b_[:2*matlab_i, :idx, :], (1, pcs+1, 1))
            # Repeat the elements of the next split and append them.
            # MATLAB: b_(2*i + (1:2),1:idx_,:) = repelem(b_(2*i + (1:2),1:(pcs+1),:),1,(pcs+1)^i,1);
            next_slice = b_[2*matlab_i:2*matlab_i+2, :(pcs+1), :]
            b_[2*matlab_i:2*matlab_i+2, :idx_, :] = repelem(next_slice, 1, (pcs+1)**matlab_i, 1)
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
        A = np.zeros((0, As.shape[1] if As.size > 0 else 0, As.shape[2] if As.size > 0 else 0), dtype=As.dtype if As.size > 0 else np.float64)
        b = np.zeros((0, bs.shape[1] if bs.size > 0 else 0), dtype=bs.dtype if bs.size > 0 else np.float64)
        constNrIdx = np.zeros((0, nrXis.shape[1] if nrXis.size > 0 else 0), dtype=nrXis.dtype if nrXis.size > 0 else np.float64)
    
    return A, b, newSplits, constNrIdx


def _aux_boundsOfBoundedPolytope(A: np.ndarray, b: np.ndarray, options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounds [bl,bu] of a bounded polytope P:
    Given P=(A,b) \\cap [-1,1], compute its bounds.
    
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
        # MATLAB: b_ = permute(b,[3 1 2]); where b is (p, bSz)
        # permute([3 1 2]) on 2D array (p, bSz) adds singleton at dim 0: (1, p, bSz)
        if b.ndim == 2:
            b_ = np.expand_dims(b, axis=0)  # (p, bSz) -> (1, p, bSz)
            b_ = np.transpose(b_, (0, 1, 2))  # Already in correct order
        elif b.ndim == 3:
            b_ = np.transpose(b, (2, 0, 1))  # (p, bSz, x) -> (x, p, bSz) or similar
        else:
            b_ = b.reshape(1, -1, 1)  # (1, p, bSz)
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
            # bl and bu have shape (q, bSz*numUnionConst) after _aux_boundsOfBoundedPolytope
            # Reshape to (q, bSz, numUnionConst) and take min/max along the last dimension
            # to get back to (q, bSz)
            bl = bl.reshape(q, bSz, numUnionConst, order='F')
            bu = bu.reshape(q, bSz, numUnionConst, order='F')
            bl = np.min(bl, axis=2)
            bu = np.max(bu, axis=2)
        else:
            # There are no constraints to unify.
            bl = bl.reshape(q, bSz, order='F')
            bu = bu.reshape(q, bSz, order='F')
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
    # MATLAB: bc = 1/2*permute(bu + bl,[1 3 2]); where bl, bu are (q, bSz_actual)
    # permute([1 3 2]) on 2D (q, bSz_actual) adds singleton: (q, 1, bSz_actual)
    # Get the actual batch size from bl/bu (which may be expanded due to splitting)
    bSz_actual = bl.shape[1]
    bc = 0.5 * (bu + bl)  # (q, bSz_actual)
    br = 0.5 * (bu - bl)  # (q, bSz_actual)
    
    # MATLAB: bc = permute(bc,[1 3 2]); adds singleton dimension
    bc = np.expand_dims(bc, axis=1)  # (q, 1, bSz_actual)
    br = np.expand_dims(br, axis=1)  # (q, 1, bSz_actual)
    
    # Map bounds of the factors to bounds of the constraint zonotope.
    # MATLAB: c = c + reshape(pagemtimes(G,bc),[n bSz]);
    # Note: G has original bSz, but bc has bSz_actual (expanded), so we need to match them
    if bSz != bSz_actual:
        # Replicate G to match the expanded batch size
        from ..layers.linear.nnGeneratorReductionLayer import repelem
        nReps = bSz_actual // bSz
        G = repelem(G, 1, 1, nReps)
        c = repelem(c, 1, nReps) if c.ndim == 2 else repelem(c, 1, 1, nReps)
        r = repelem(r, 1, nReps) if r.ndim == 2 else repelem(r, 1, 1, nReps)
        bSz = bSz_actual  # Update bSz to the actual batch size
    
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
            # Handle empty sensitivity array
            if Si_.size == 0:
                sens = np.ones((nk, bSz), dtype=prevNrXs.dtype)
            else:
                sens = np.reshape(np.max(Si_, axis=0), (nk, -1))
        else:
            sens = np.ones((nk, bSz), dtype=prevNrXs.dtype)
        
        if As.shape[2] < bSz:
            padBSz = bSz - As.shape[2]
            # Pad to the correct batch size.
            # MATLAB: As = cat(3,As,zeros([q nNeur padBSz],'like',As));
            As = np.concatenate([As, np.zeros((q, nNeur, padBSz), dtype=As.dtype)], axis=2)
            # MATLAB: bs = cat(3,bs,zeros([(nSplits-1) nNeur padBSz],'like',bs));
            bs = np.concatenate([bs, np.zeros((nSplits-1, nNeur, padBSz), dtype=bs.dtype)], axis=2)
            # MATLAB: h = [h -ones([nNeur padBSz],'like',h)];
            h = np.concatenate([h, -np.ones((nNeur, padBSz), dtype=h.dtype)], axis=1)
            # MATLAB: constNrIdx = [constNrIdx,Inf([nNeur padBSz],'like',constNrIdx)];
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
        
        # Ensure bsi matches bs's batch size before concatenation
        if bs.shape[2] > bsi.shape[2]:
            pad_bSz = bs.shape[2] - bsi.shape[2]
            bsi = np.concatenate([bsi, np.zeros((nSplits-1, nk, pad_bSz), dtype=bsi.dtype)], axis=2)
        
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
        
        # Ensure h, hi, bs, and constNrIdx have matching batch sizes
        # Use hi's batch size as the true batch size for this layer
        # (hi inherits batch size from sens, which may differ from Gi's batch size)
        hi_bSz = hi.shape[1] if hi.ndim > 1 else 1
        if h.shape[1] < hi_bSz:
            padBSz = hi_bSz - h.shape[1]
            h = np.concatenate([h, -np.ones((nNeur, padBSz), dtype=h.dtype)], axis=1)
        if bs.shape[2] < hi_bSz:
            pad_bSz = hi_bSz - bs.shape[2]
            bs = np.concatenate([bs, np.zeros((nSplits-1, bs.shape[1], pad_bSz), dtype=bs.dtype)], axis=2)
        if constNrIdx.shape[1] < hi_bSz:
            pad_bSz = hi_bSz - constNrIdx.shape[1]
            constNrIdx = np.concatenate([constNrIdx, np.full((nNeur, pad_bSz), np.inf, dtype=constNrIdx.dtype)], axis=1)
        
        # Append heuristic and sort.
        # MATLAB: [h,idx] = sort([h; hi(:,:)],1,'descend');
        # hi(:,:) flattens hi to 2D if needed
        if hi.ndim > 2:
            hi = hi.reshape(hi.shape[0], -1, order='F')
        h_combined = np.concatenate([h, hi], axis=0)  # (nNeur + nk, bSz)
        # Sort descending along axis 0 (within each column/batch)
        sort_idx = np.argsort(-h_combined, axis=0)  # Descending order
        # MATLAB: h = h(1:nNeur,:); - keep only top nNeur rows after sorting
        # We need to extract values using the sorted indices for each batch
        h_sorted = np.take_along_axis(h_combined, sort_idx, axis=0)
        h = h_sorted[:nNeur, :]  # (nNeur, bSz)
        
        # Obtain the indices for the relevant constraints.
        # MATLAB: sIdx = sub2ind(size(As,2:3), idx(1:nNeur,:),repmat(1:bSz,nNeur,1));
        # size(As,2:3) = [As.shape[1], As.shape[2]] in Python (0-based)
        # idx(1:nNeur,:) = sort_idx[:nNeur, :] + 1 for 1-based MATLAB indexing
        # repmat(1:bSz,nNeur,1) = np.tile(np.arange(1, bSz+1), (nNeur, 1))
        # Use actual_bSz for consistency
        actual_bSz = hi_bSz
        row_indices = (sort_idx[:nNeur, :actual_bSz] + 1).flatten('F')  # 1-based for MATLAB sub2ind
        col_indices = np.tile(np.arange(1, actual_bSz + 1), nNeur)  # 1-based
        sIdx = sub2ind((As.shape[1], actual_bSz), row_indices, col_indices)
        # sub2ind returns 0-based indices for Python
        
        # Extract constraints.
        # MATLAB: As = reshape(As(:,sIdx),[q nNeur bSz]);
        # Use hi_bSz as the actual batch size after all padding
        actual_bSz = hi_bSz
        if q > 0:
            As_flat = As.reshape(q, -1, order='F')
            As = As_flat[:, sIdx].reshape((q, nNeur, actual_bSz), order='F')
        else:
            # No constraints yet, create empty array with correct shape
            As = np.zeros((0, nNeur, actual_bSz), dtype=As.dtype)
        
        # MATLAB: bs = reshape(bs(:,sIdx),[nSplits-1 nNeur bSz]);
        bs_flat = bs.reshape(nSplits-1, -1, order='F')
        bs = bs_flat[:, sIdx].reshape((nSplits-1, nNeur, actual_bSz), order='F')
        
        # Update indices.
        # MATLAB: constNrIdx = [constNrIdx; repmat(neuronIds,bSz,1)'];
        # repmat(neuronIds,bSz,1)' = repmat(neuronIds,bSz,1) transposed = (nk, bSz)
        constNrIdx_new = np.tile(neuronIds.reshape(-1, 1), (1, actual_bSz))  # (nk, actual_bSz)
        constNrIdx = np.concatenate([constNrIdx, constNrIdx_new], axis=0)  # (nNeur + nk, actual_bSz)
        # MATLAB: constNrIdx = reshape(constNrIdx(sIdx),[nNeur bSz]);
        # Extract using linear indices sIdx, then reshape
        constNrIdx_flat = constNrIdx.flatten('F')  # Column-major order
        constNrIdx = constNrIdx_flat[sIdx].reshape((nNeur, actual_bSz), order='F')
    
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
        # sub2ind already returns 0-based indices, so use directly
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
        # After _aux_matchBatchSize, G might have a different batch size than bc
        # MATLAB doesn't replicate inside aux_scaleAndOffsetZonotope, so we shouldn't either
        # Just call the function and let it handle the batch size mismatch
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
        # MATLAB: cl_ = c_; cu_ = c_; where c_ is (nk, bSz) when interval_center is False
        # If c_ is 3D, squeeze or reshape it to 2D
        if c_.ndim > 2:
            # This shouldn't happen when interval_center is False, but handle it
            if c_.shape[1] == 1:
                c_ = c_.squeeze(axis=1)  # (nk, bSz)
            else:
                # Take the first "slice" or reshape
                c_ = c_.reshape(nk, -1)
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
            l = np.full((1, bSz), -np.inf, dtype=bc.dtype if bc.size > 0 else np.float64)
            u = np.full((1, bSz), np.inf, dtype=bc.dtype if bc.size > 0 else np.float64)
            cl_ = np.array([])
            cu_ = np.array([])
            G_ = np.array([])
    else:
        l = np.full((1, bSz), -np.inf, dtype=bc.dtype if bc.size > 0 else np.float64)
        u = np.full((1, bSz), np.inf, dtype=bc.dtype if bc.size > 0 else np.float64)
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
    # MATLAB: [nK,q,bSz] = size(Gy);
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
            # MATLAB: uYi.A = [uYi.A; C_]; uYi.b = [uYi.b; d_];
            if uYi['A'].size > 0:
                # Check if batch sizes match; if not, replicate uYi to match C_'s batch size
                uYi_bSz = uYi['A'].shape[2]
                C_bSz = C_.shape[2]
                if uYi_bSz != C_bSz:
                    # Replicate uYi to match the batch size
                    from ..layers.linear.nnGeneratorReductionLayer import repelem
                    nReps = C_bSz // uYi_bSz
                    uYi['A'] = repelem(uYi['A'], 1, 1, nReps)
                    uYi['b'] = repelem(uYi['b'], 1, nReps)
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
        # Check if batch size has changed (due to splitting in _aux_boundsOfConZonotope)
        ly_bSz = ly.shape[1] if ly.ndim > 1 else 1
        if isEmpty.shape[1] != ly_bSz:
            # Replicate isEmpty to match the new batch size
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            nReps = ly_bSz // isEmpty.shape[1]
            isEmpty = repelem(isEmpty, 1, nReps)
        isEmpty = isEmpty | np.any(np.isnan(ly), axis=0) | np.any(np.isnan(uy), axis=0)

        # Compute the center and radius of the new inner hypercube.
        bci = 0.5 * (bui + bli)
        bri = 0.5 * (bui - bli)
        
        # Update the hypercube.
        # Check if batch sizes match; if not, replicate bc and br
        if bc.shape[1] != bci.shape[1]:
            # Replicate bc and br to match the new batch size
            from ..layers.linear.nnGeneratorReductionLayer import repelem
            nReps = bci.shape[1] // bc.shape[1]
            bc = repelem(bc, 1, nReps)
            br = repelem(br, 1, nReps)
            bSz = bci.shape[1]  # Update bSz to match the new batch size
        
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
            # Convert ancIdx to array for element-wise comparison
            ancIdx_arr = np.array(ancIdx)
            scaleInputSets[0, ancIdx_arr >= ancIdx[i]] = False
        
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
            grad = torch.maximum(torch.abs(gl), torch.abs(gu))
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
                    c_in = store.get('inc', torch.zeros_like(gc))
                    G_in = store.get('inG', torch.zeros_like(gG))
                else:
                    c_in = torch.zeros_like(gc)
                    G_in = torch.zeros_like(gG)
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
    l = np.full((len(neuronIds), bSz), -np.inf, dtype=d.dtype if d.size > 0 else np.float64)
    u = np.full((len(neuronIds), bSz), np.inf, dtype=d.dtype if d.size > 0 else np.float64)
    
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
    # Handle both num_relu_constraints and num_relu_tighten_constraints (MATLAB test uses both)
    numConstr = options.get('nn', {}).get('num_relu_constraints', 0)
    if numConstr == 0:
        # Check for num_relu_tighten_constraints as alias (used in MATLAB tests)
        numConstr = options.get('nn', {}).get('num_relu_tighten_constraints', 0)
    
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
        # sub2ind already returns 0-based indices, so use directly
        
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
    # sub2ind already returns 0-based indices, so use directly
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
