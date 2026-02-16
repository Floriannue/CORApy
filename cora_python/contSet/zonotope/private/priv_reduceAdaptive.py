"""
priv_reduceAdaptive - reduces the zonotope order until a maximum amount
   of over-approximation defined by the Hausdorff distance between the
   original zonotope and the reduced zonotope; based on [Thm 3.2,1]

Syntax:
    Z = priv_reduceAdaptive(Z,diagpercent)
    [Z,dHerror,gredIdx] = priv_reduceAdaptive(Z,diagpercent,type)

Inputs:
    Z - zonotope object
    diagpercent - percentage of diagonal of box over-approximation of
               zonotope (used to compute dHmax) [0,1]
    type - optional, 'girard' (default) or 'penven'

Outputs:
    Z - reduced zonotope
    dHerror - Hausdorff distance between Z and reduced Z
    gredIdx - index of reduced generators

References:
    [1] Wetzlinger et al. "Adaptive Parameter Tuning for Reachability 
        Analysis of Nonlinear Systems", HSCC 2021

Authors:       Mark Wetzlinger (MATLAB)
               Python translation by AI Assistant
Written:       01-October-2020 (MATLAB)
Last update:   16-June-2021 (MATLAB)
               2025 (Python translation)
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.zonotope import Zonotope


def priv_reduceAdaptive(Z: 'Zonotope', diagpercent: float,
                        type: Optional[str] = None, 
                        track_details: bool = False) -> Tuple['Zonotope', float, np.ndarray]:
    """
    Adaptive reduction method for zonotopes.
    
    Args:
        Z: Zonotope object
        diagpercent: Percentage of diagonal (used to compute dHmax) [0,1]
        type: Optional, 'girard' (default) or 'penven'
    
    Returns:
        Tuple of (reduced zonotope, dHerror, gredIdx)
    """
    from cora_python.contSet.zonotope import Zonotope
    
    # Handle type parameter - can be string or dict (for backward compatibility)
    track_details_from_type = False
    if isinstance(type, dict):
        # Backward compatibility: type was passed as dict with track_details
        track_details_from_type = type.get('track_details', False)
        type_str = type.get('type', 'girard')
        type = type_str if type_str in ['penven', 'girard'] else 'girard'
        track_details = track_details or track_details_from_type
    elif type is None:
        type = 'girard'
    elif type not in ['penven', 'girard']:
        type = 'girard'
    
    dHerror = 0.0
    G = Z.generators()
    
    # Convert sparse matrices to dense arrays for efficient computation
    # This reduction algorithm performs many element-wise operations, slicing, and reductions
    # that are more efficient with dense arrays. The conversion is intentional and necessary
    # for performance, not a workaround.
    import scipy.sparse
    if scipy.sparse.issparse(G):
        G = np.asarray(G.toarray(), dtype=np.float64)
    else:
        G = np.asarray(G, dtype=np.float64)
    
    if G is None or G.size == 0:
        dHerror = 0.0
        gredIdx = np.array([], dtype=int)
        return Z, dHerror, gredIdx
    
    # Ensure Gabs is dense array
    Gabs = np.asarray(np.abs(G), dtype=np.float64)
    
    # Initialize tracking
    reduction_details = None
    if track_details:
        reduction_details = {
            'diagpercent': diagpercent
        }
    
    # Compute maximum admissible dH
    # MATLAB: Gbox = sum(Gabs,2);
    Gbox = np.sum(Gabs, axis=1, keepdims=True)  # Column vector (n, 1)
    # MATLAB: dHmax = (diagpercent * 2) * sqrt(sum(Gbox.^2));
    dHmax = (diagpercent * 2) * np.sqrt(np.sum(Gbox ** 2))
    
    if track_details and reduction_details is not None:
        reduction_details['dHmax'] = float(dHmax)
        reduction_details['Gbox_sum'] = float(np.sum(Gbox))
    
    n, nrG = G.shape
    
    if type == 'penven':
        # Compute spherical (naive) bound for each generator
        # MATLAB: naive = sqrt(sum(Gabs.^2,1));
        naive = np.sqrt(np.sum(Gabs ** 2, axis=0))  # Row vector (1, nrG)
        
        # Compute Le Penven bound for each generator
        # MATLAB: penven = naive .* sqrt(2*(abs(1-  sum(Gabs.^4,1)./sum(Gabs.^2,1).^2 )  ));
        Gabs_sq = Gabs ** 2
        Gabs_4 = Gabs ** 4
        sum_sq = np.sum(Gabs_sq, axis=0)
        sum_4 = np.sum(Gabs_4, axis=0)
        penven = naive * np.sqrt(2 * np.abs(1 - sum_4 / (sum_sq ** 2)))
        
        # Compare them
        # MATLAB: resulting = min([naive;penven]);
        resulting = np.minimum(naive, penven)
        
        # Sort them - MATLAB uses mink(resulting,nrG) which returns sorted ascending
        # Python: use argpartition to get indices of smallest values
        idx = np.argpartition(resulting, nrG - 1)[:nrG]  # Get all indices, sorted
        idx = idx[np.argsort(resulting[idx])]  # Sort by value
        h = resulting[idx]
        
        if not np.any(h):
            # no generators or all are h=0
            newG = np.diag(Gbox.flatten())
            # Remove zero columns
            newG = newG[:, np.any(newG, axis=0)]
            Z_reduced = Zonotope(Z.center(), newG)
            gredIdx = idx
            return Z_reduced, dHerror, gredIdx
        
        # box generators with h = 0
        hzeroIdx = idx[h == 0]
        Gzeros = np.sum(Gabs[:, hzeroIdx], axis=1, keepdims=True)
        last0Idx = len(hzeroIdx)
        gensred = Gabs[:, idx[last0Idx:]]
        
        # Take cumsum of the bounds for each generator, take the first few
        # MATLAB: s = cumsum(h);
        s = np.cumsum(h)
        # MATLAB: redIdx = find(s(last0Idx+1:end) <= dHmax, 1, 'last');
        # This finds the last index in s(last0Idx+1:end) where s <= dHmax
        # MATLAB indexing: s(last0Idx+1:end) means from index last0Idx+1 to end (1-based)
        # Python: s[last0Idx:] means from index last0Idx to end (0-based)
        s_subset = s[last0Idx:]
        redIdx_arr = np.where(s_subset <= dHmax)[0]
        if len(redIdx_arr) == 0:
            redIdx = 0
            dHerror = 0.0
            gredIdx = hzeroIdx
        else:
            # redIdx is the index in s_subset, so we need to add last0Idx to get index in h
            # But MATLAB's find returns the position in s_subset, which is 1-based
            # So redIdx is the count of how many to reduce from s_subset
            redIdx = redIdx_arr[-1] + 1  # +1 because MATLAB is 1-based
            # dHerror is the value at that position in h (not s)
            # MATLAB: dHerror = h(redIdx); but redIdx is in s_subset indexing
            # Actually, looking at MATLAB: dHerror = h(redIdx) where redIdx is from s_subset
            # But h is the original h array, so we need h[last0Idx + redIdx - 1] (convert to 0-based)
            dHerror = h[last0Idx + redIdx - 1] if redIdx > 0 else 0.0
            # gredIdx includes hzeroIdx plus redIdx more from idx
            gredIdx = idx[:last0Idx + redIdx]
    
    else:  # 'girard'
        # Select generators using 'girard'
        # MATLAB: norminf = max(Gabs,[],1);
        norminf = np.max(Gabs, axis=0)  # Row vector (1, nrG)
        # MATLAB: normsum = sum(Gabs,1);
        normsum = np.sum(Gabs, axis=0)  # Row vector (1, nrG)
        # MATLAB: [h,idx] = mink(normsum - norminf,nrG);
        # mink returns sorted ascending, so we want smallest values first
        diff = normsum - norminf
        idx = np.argpartition(diff, nrG - 1)[:nrG]  # Get all indices
        idx = idx[np.argsort(diff[idx])]  # Sort by value
        h_initial = diff[idx]  # Store initial h for tracking
        h = h_initial
        
        if not np.any(h):
            # no generators or all are h=0
            newG = np.diag(Gbox.flatten())
            # Remove zero columns
            newG = newG[:, np.any(newG, axis=0)]
            # Ensure newG is dense array
            newG = np.asarray(newG, dtype=np.float64)
            # Ensure center is dense array
            c = Z.center()
            import scipy.sparse
            if scipy.sparse.issparse(c):
                c = np.asarray(c.toarray().flatten(), dtype=np.float64)
            else:
                c = np.asarray(c, dtype=np.float64)
            Z_reduced = Zonotope(c, newG)
            dHerror = 0.0
            gredIdx = idx
            return Z_reduced, dHerror, gredIdx
        
        # box generators with h = 0
        hzeroIdx = idx[h == 0]
        Gzeros = np.sum(Gabs[:, hzeroIdx], axis=1, keepdims=True)
        last0Idx = len(hzeroIdx)
        gensred = Gabs[:, idx[last0Idx:]]
        # Ensure gensred is dense array
        gensred = np.asarray(gensred)
        
        # MATLAB: [maxval,maxidx] = max(gensred,[],1);
        maxidx = np.argmax(gensred, axis=0)  # Indices of max along axis 0 (dimension)
        maxval = np.max(gensred, axis=0)  # Max values
        
        # use linear indexing
        # MATLAB: mugensred = zeros(n,nrG-last0Idx);
        nrG_red = nrG - last0Idx
        mugensred = np.zeros((n, nrG_red), dtype=G.dtype)
        # MATLAB: cols = n*(0:nrG-last0Idx-1);
        cols = n * np.arange(nrG_red)
        # MATLAB: mugensred(cols+maxidx) = maxval;
        # Ensure indices are valid and use proper indexing
        valid_indices = cols + maxidx
        mugensred.flat[valid_indices] = maxval
        
        # compute new over-approximation of dH
        # MATLAB: gensdiag = cumsum(gensred-mugensred,2);
        # MATLAB cumsum along dimension 2 (columns) = Python axis=1
        gensdiag = np.cumsum(gensred - mugensred, axis=1)
        # MATLAB: h = 2 * vecnorm(gensdiag,2);
        # MATLAB vecnorm(gensdiag,2) without dimension argument defaults to dimension 1 (columns)
        # This computes the 2-norm of each column vector: sqrt(sum(gensdiag.^2,1))
        # In Python, axis=0 means compute norm along rows (for each column)
        # Verify: vecnorm(gensdiag,2) = sqrt(sum(gensdiag.^2,1)) for each column
        h = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)  # norm of each column
        
        # index until which gens are reduced
        # MATLAB: redIdx = find(h <= dHmax,1,'last');
        # This finds the last index where h <= dHmax
        # MATLAB's find returns 1-based index into h array
        # h has length (nrG - last0Idx), so redIdx is 1-based index into h
        redIdx_arr = np.where(h <= dHmax)[0]
        
        if len(redIdx_arr) == 0:
            redIdx = 0
            dHerror = 0.0
            gredIdx = hzeroIdx
        else:
            # MATLAB: redIdx = find(h <= dHmax,1,'last');
            # MATLAB returns 1-based index into h
            # Python: redIdx_arr[-1] is 0-based index into h
            # Convert to 1-based to match MATLAB
            redIdx_0based = redIdx_arr[-1]  # Last valid 0-based index into h
            redIdx = redIdx_0based + 1  # Convert to 1-based (MATLAB style)
            # dHerror is the value at that position in h
            dHerror = h[redIdx_0based]
            # gredIdx includes hzeroIdx plus redIdx more from idx
            # MATLAB: gredIdx = idx(1:length(hzeroIdx)+redIdx);
            # This means first (last0Idx + redIdx) elements of idx
            # redIdx is the count of how many from gensred to reduce
            # So we take idx[0:last0Idx+redIdx] (first last0Idx+redIdx elements)
            gredIdx = idx[:last0Idx + redIdx]
            
            # Debug tracking disabled - data already captured in initReach_tracking
            
            # Track details if requested
            if track_details and reduction_details is not None:
                reduction_details.update({
                    'nrG': int(nrG),
                    'h_initial': h_initial.tolist() if 'h_initial' in locals() and hasattr(h_initial, 'tolist') else None,
                    'hzeroIdx': hzeroIdx.tolist() if hasattr(hzeroIdx, 'tolist') else list(hzeroIdx),
                    'last0Idx': int(last0Idx),
                    'gensred_shape': gensred.shape,
                    'h_computed': h.tolist() if hasattr(h, 'tolist') else list(h),
                    'h_computed_max': float(np.max(h)),
                    'h_computed_min': float(np.min(h)),
                    'redIdx_arr': redIdx_arr.tolist() if hasattr(redIdx_arr, 'tolist') else list(redIdx_arr),
                    'redIdx_0based': int(redIdx_0based),
                    'redIdx': int(redIdx),
                    'dHerror': float(dHerror),
                    'gredIdx': gredIdx.tolist() if hasattr(gredIdx, 'tolist') else list(gredIdx),
                    'gredIdx_len': len(gredIdx)
                })
    
    # MATLAB: Gred = sum(gensred(:,1:redIdx),2);
    # MATLAB: gensred(:,1:redIdx) means columns 1 to redIdx (1-based, inclusive)
    # This is columns 0 to redIdx-1 in 0-based (redIdx columns total)
    # So we use gensred[:, :redIdx] which gives columns 0 to redIdx-1
    if redIdx > 0:
        Gred = np.sum(gensred[:, :redIdx], axis=1, keepdims=True)
    else:
        Gred = np.zeros((n, 1))
    
    # MATLAB: Gunred = G(:,sort(idx(last0Idx+redIdx+1:end)));
    # MATLAB: idx(last0Idx+redIdx+1:end) means from index last0Idx+redIdx+1 to end (1-based)
    # Python: idx[last0Idx+redIdx:] means from index last0Idx+redIdx to end (0-based)
    # sort to keep correspondances!
    if last0Idx + redIdx < len(idx):
        gunred_idx = idx[last0Idx + redIdx:]
        gunred_idx_sorted = np.sort(gunred_idx)
        Gunred = G[:, gunred_idx_sorted]
    else:
        Gunred = np.array([]).reshape(n, 0)
    
    # MATLAB: Z.G = [Gunred,diag(Gred+Gzeros)];
    # MATLAB's diag(v) creates a diagonal matrix where v is the diagonal
    # In MATLAB: diag([a;b;c]) creates [a 0 0; 0 b 0; 0 0 c]
    # This is a (n, n) matrix where each column is a generator
    # So we want np.diag(Gred_total) which creates the same structure
    Gred_total = (Gred + Gzeros).flatten()
    G_diag = np.diag(Gred_total)  # Creates (n, n) diagonal matrix
    # Remove zero columns (generators that are all zero)
    G_diag = G_diag[:, np.any(G_diag, axis=0)]
    G_new = np.hstack([Gunred, G_diag]) if Gunred.size > 0 else G_diag
    # Ensure G_new is dense array
    G_new = np.asarray(G_new, dtype=np.float64)
    
    # Ensure center is dense array
    c = Z.center()
    import scipy.sparse
    if scipy.sparse.issparse(c):
        c = np.asarray(c.toarray().flatten(), dtype=np.float64)
    else:
        c = np.asarray(c, dtype=np.float64)
    
    Z_reduced = Zonotope(c, G_new)
    
    # Store reduction details in Z if tracking enabled
    if track_details and reduction_details is not None:
        reduction_details['final_generators'] = G_new.shape[1]
        Z_reduced._reduction_details = reduction_details
        
        # Also write to file for reliable access (similar to MATLAB approach)
        try:
            import pickle
            import os
            debug_file = 'reduceAdaptive_debug_python.pkl'
            # Read existing data if file exists
            if os.path.exists(debug_file):
                with open(debug_file, 'rb') as f:
                    try:
                        debug_data = pickle.load(f)
                    except (EOFError, ValueError):
                        debug_data = []
            else:
                debug_data = []
            # Append new entry
            debug_data.append(reduction_details)
            # Write back
            with open(debug_file, 'wb') as f:
                pickle.dump(debug_data, f)
        except Exception:
            # Don't fail if file I/O fails
            pass
    
    return Z_reduced, dHerror, gredIdx
