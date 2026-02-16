"""
quadMap - computes the quadratic map of a zonotope

Description:
    This function computes the quadratic map of a zonotope according to the 
    methods described in [1].

Syntax:
    Zquad = quadMap(Z1, Q)
    Zquad = quadMap(Z1, Z2, Q)

Inputs:
    Z1 - zonotope object
    Z2 - zonotope object (optional)
    Q - quadratic coefficients as a list of matrices

Outputs:
    Zquad - zonotope object

Example: 
    Z = Zonotope(np.array([[0, 1, 1], [0, 1, 0]]))
    Q = [np.array([[0.5, 0.5], [0, -0.5]]), np.array([[-1, 0], [1, 1]])]
    res = quadMap(Z, Q)

References:
    [1] M. Althoff et al. "Avoiding Geometric Intersection Operations in 
        Reachability Analysis of Hybrid Systems", HSCC 2011.

Authors: Matthias Althoff, Niklas Kochdumper (MATLAB)
         Python translation by AI Assistant
Written: 07-December-2011 (MATLAB)
Last update: 22-November-2019 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from typing import List
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.helper.sets.contSet.zonotope import nonzeroFilter

from .zonotope import Zonotope

def quadMap(Z1: Zonotope, *args) -> Zonotope:
    """
    Compute the quadratic map of a zonotope
    
    Args:
        Z1: First zonotope object
        *args: Either (Q,) for single zonotope case or (Z2, Q) for mixed case
        
    Returns:
        Zonotope object representing the quadratic map
    """
    
    if len(args) == 1:
        Q = args[0]
        # Check if Q contains matZonotope objects (matching MATLAB logic)
        # MATLAB: isa(varargin{1}{1},'matZonotope')
        # Python: check if first element of Q is matZonotope type
        try:
            from cora_python.matrixSet import matZonotope
            if len(Q) > 0 and isinstance(Q[0], matZonotope):
                return _aux_quadMapSingleMatZono(Z1, Q)
            else:
                return _aux_quadMapSingle(Z1, Q)
        except ImportError:
            # matZonotope class not available, use regular matrices
            return _aux_quadMapSingle(Z1, Q)
    elif len(args) == 2:
        Z2, Q = args
        return _aux_quadMapMixed(Z1, Z2, Q)
    else:
        raise ValueError("Invalid number of input arguments")


def _aux_quadMapSingle(Z: Zonotope, Q: List[np.ndarray]) -> Zonotope:
    """
    Compute an over-approximation of the quadratic map 
    {x_i = x^T Q{i} x | x ∈ Z} 
    of a zonotope according to Lemma 1 in [1]
    """
    
    # Get matrix of zonotope [c, G]
    if Z.G.size == 0:
        Zmat = Z.c
    else:
        Zmat = np.hstack([Z.c, Z.G])
    
    dimQ = len(Q)
    gens = Z.G.shape[1] if Z.G.size > 0 else 0

    # Initialize solution
    c = np.zeros((dimQ, 1))
    G = np.zeros((dimQ, int(0.5 * (gens**2 + gens) + gens)))
    
    # Count non-empty matrices
    Qnonempty = np.zeros(dimQ, dtype=bool)
    
    # For each dimension, compute generator elements
    for i in range(dimQ):
        # MATLAB: Qnonempty(i) = any(Q{i}(:));
        # Q[i] can be an Interval object (from Hessian tensor)
        from cora_python.contSet.interval import Interval
        if isinstance(Q[i], Interval):
            # For Interval, check if any element is non-zero
            # Flatten the interval and check if any element is non-zero
            # MATLAB: any(Q{i}(:)) checks if any element in the flattened matrix is non-zero
            import scipy.sparse as sp
            Q_inf = Q[i].inf.toarray() if sp.issparse(Q[i].inf) else Q[i].inf
            Q_sup = Q[i].sup.toarray() if sp.issparse(Q[i].sup) else Q[i].sup
            Q_i_inf_flat = np.asarray(Q_inf).flatten()
            Q_i_sup_flat = np.asarray(Q_sup).flatten()
            # Check if any element is non-zero (either inf or sup is non-zero)
            Qnonempty[i] = bool(np.any(Q_i_inf_flat != 0) or np.any(Q_i_sup_flat != 0))
            Q_i = Q[i]  # Keep as Interval for matrix multiplication (uses @ operator)
        else:
            # Handle different types: numeric array, sparse matrix, etc.
            try:
                import scipy.sparse
                if scipy.sparse.issparse(Q[i]):
                    # For sparse matrices, check if any element is non-zero
                    Qnonempty[i] = Q[i].nnz > 0
                    Q_i = Q[i]  # Keep as sparse matrix
                else:
                    # For regular arrays
                    Q_i = np.asarray(Q[i])
                    # Flatten and check if any element is non-zero
                    if Q_i.size > 0:
                        Q_i_flat = Q_i.flatten()
                        # Use .item() to extract scalar from np.any result
                        Qnonempty[i] = bool(np.any(Q_i_flat).item() if hasattr(np.any(Q_i_flat), 'item') else np.any(Q_i_flat))
                    else:
                        Qnonempty[i] = False
            except ImportError:
                # No scipy, assume regular array
                Q_i = np.asarray(Q[i])
                if Q_i.size > 0:
                    Q_i_flat = Q_i.flatten()
                    Qnonempty[i] = bool(np.any(Q_i_flat).item() if hasattr(np.any(Q_i_flat), 'item') else np.any(Q_i_flat))
                else:
                    Qnonempty[i] = False
        
        if Qnonempty[i]:
            # Pure quadratic evaluation
            quadMat = Zmat.T @ Q_i @ Zmat
            
            # Track quadMat for comparison (if tracking enabled)
            # This helps debug why errorSec differs
            if hasattr(Z, '_track_quadmat') and Z._track_quadmat:
                from cora_python.contSet.interval import Interval
                import scipy.sparse
                quadmat_info = {
                    'type': type(quadMat).__name__,
                    'is_interval': isinstance(quadMat, Interval),
                    'is_sparse': scipy.sparse.issparse(quadMat),
                }
                if isinstance(quadMat, Interval):
                    quadmat_info['inf_max'] = np.max(np.abs(quadMat.inf)) if hasattr(quadMat.inf, '__abs__') else None
                    quadmat_info['sup_max'] = np.max(np.abs(quadMat.sup)) if hasattr(quadMat.sup, '__abs__') else None
                    quadmat_info['center_max'] = np.max(np.abs(quadMat.center())) if hasattr(quadMat, 'center') else None
                elif scipy.sparse.issparse(quadMat):
                    quadmat_info['sparse_max'] = np.max(np.abs(quadMat.data)) if hasattr(quadMat, 'data') else None
                    quadmat_dense = quadMat.toarray()
                    quadmat_info['dense_max'] = np.max(np.abs(quadmat_dense))
                    if gens > 0:
                        quadmat_info['dense_diag'] = np.diag(quadmat_dense[1:gens+1, 1:gens+1]).copy()
                elif isinstance(quadMat, np.ndarray):
                    quadmat_info['dense_max'] = np.max(np.abs(quadMat))
                    if gens > 0:
                        quadmat_info['dense_diag'] = np.diag(quadMat[1:gens+1, 1:gens+1]).copy()
                        # Also store full matrix for detailed comparison
                        quadmat_info['dense_full'] = quadMat.copy()
                
                if not hasattr(Z, '_quadmat_values'):
                    Z._quadmat_values = []
                Z._quadmat_values.append((i, quadmat_info))
            
            # Extract numeric values from quadMat if it's an Interval
            # MATLAB uses max(abs(infimum), abs(supremum)) for conservative over-approximation
            # This matches the tensorOrder==2 conversion: max(infimum(abs(H_)), supremum(abs(H_)))
            from cora_python.contSet.interval import Interval
            import scipy.sparse
            
            # Debug: Check if quadMat is actually Interval (should be numeric for 'standard' Hessian)
            # If Q_i is sparse/numeric, quadMat should be numeric too
            if isinstance(quadMat, Interval):
                # For Interval, use max(abs(inf), abs(sup)) for conservative over-approximation
                # This matches MATLAB's behavior in tensorOrder==2 case
                quadMat_inf = quadMat.inf.toarray() if scipy.sparse.issparse(quadMat.inf) else quadMat.inf
                quadMat_sup = quadMat.sup.toarray() if scipy.sparse.issparse(quadMat.sup) else quadMat.sup
                # Use maximum of absolute values (conservative over-approximation)
                quadMat = np.maximum(np.abs(quadMat_inf), np.abs(quadMat_sup))
            elif hasattr(quadMat, 'inf') and hasattr(quadMat, 'sup'):
                # Handle other interval-like objects
                quadMat_inf = quadMat.inf.toarray() if scipy.sparse.issparse(quadMat.inf) else quadMat.inf
                quadMat_sup = quadMat.sup.toarray() if scipy.sparse.issparse(quadMat.sup) else quadMat.sup
                # Use maximum of absolute values (conservative over-approximation)
                quadMat = np.maximum(np.abs(quadMat_inf), np.abs(quadMat_sup))
            elif scipy.sparse.issparse(quadMat):
                # Convert sparse matrix to dense
                quadMat = quadMat.toarray()
            
            # Ensure quadMat is a numpy array (dense)
            if not isinstance(quadMat, np.ndarray):
                quadMat = np.asarray(quadMat)
            elif scipy.sparse.issparse(quadMat):
                quadMat = quadMat.toarray()
            
            # Track quadMat after conversion
            if hasattr(Z, '_track_quadmat') and Z._track_quadmat:
                if hasattr(Z, '_quadmat_values') and len(Z._quadmat_values) > 0:
                    idx, info = Z._quadmat_values[-1]
                    info['after_convert_max'] = np.max(np.abs(quadMat))
                    if gens > 0:
                        info['after_convert_diag'] = np.diag(quadMat[1:gens+1, 1:gens+1]).copy()
            
            if gens > 0:
                # Diagonal elements
                # MATLAB: G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));
                # Python: quadMat[1:gens+1, 1:gens+1] (0-based indexing)
                quadMat_sub = quadMat[1:gens+1, 1:gens+1]
                G[i, :gens] = 0.5 * np.diag(quadMat_sub)
                
                # Center
                # MATLAB: c(i,1) = quadMat(1,1) + sum(G(i,1:gens));
                c[i, 0] = quadMat[0, 0] + np.sum(G[i, :gens])
                
                # Off-diagonal elements
                # MATLAB: quadMatoffdiag = quadMat + quadMat';
                quadMatoffdiag = quadMat + quadMat.T
                # MATLAB: quadMatoffdiag = quadMatoffdiag(:);
                # MATLAB uses column-major (Fortran) order for flattening
                quadMatoffdiag_flat = quadMatoffdiag.flatten(order='F')
                
                # Create lower triangular mask (excluding diagonal)
                # MATLAB: kInd = tril(true(gens+1,gens+1),-1);
                kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
                # MATLAB: G(i, gens+1:end) = quadMatoffdiag(kInd(:));
                # MATLAB uses column-major (Fortran) order for mask flattening too
                G[i, gens:] = quadMatoffdiag_flat[kInd.flatten(order='F')]
            else:
                # No generators case
                c[i, 0] = quadMat[0, 0] if quadMat.size > 0 else 0
    
    # Generate new zonotope
    tmp_sum = np.sum(Qnonempty)
    if tmp_sum < 1 or withinTol(tmp_sum, 1):
        # Single or no non-empty Q matrices
        G_sum = np.sum(np.abs(G), axis=1, keepdims=True)
        return Zonotope(c, G_sum)
    else:
        # Multiple non-empty Q matrices
        G_filtered = nonzeroFilter(G)
        return Zonotope(c, G_filtered)


def _aux_quadMapMixed(Z1: Zonotope, Z2: Zonotope, Q: List[np.ndarray]) -> Zonotope:
    """
    Compute an over-approximation of the quadratic map 
    {x_i = x1^T Q{i} x2 | x1 ∈ Z1, x2 ∈ Z2} 
    of two zonotope objects.
    """
    
    # Get matrices of zonotopes [c, G]
    if Z1.G.size == 0:
        Zmat1 = Z1.c
    else:
        Zmat1 = np.hstack([Z1.c, Z1.G])
        
    if Z2.G.size == 0:
        Zmat2 = Z2.c
    else:
        Zmat2 = np.hstack([Z2.c, Z2.G])
    
    dimQ = len(Q)
    
    # Initialize solution (center + generator matrix)
    Z = np.zeros((dimQ, Zmat1.shape[1] * Zmat2.shape[1]))
    
    # Count non-empty matrices
    Qnonempty = np.zeros(dimQ, dtype=bool)

    # For each dimension, compute center + generator elements
    for i in range(dimQ):
        # MATLAB: Qnonempty(i) = any(Q{i}(:));
        # Q[i] can be an Interval object (from Hessian tensor) or sparse matrix
        from cora_python.contSet.interval import Interval
        if isinstance(Q[i], Interval):
            # For Interval, check if any element is non-zero
            Q_i_inf_flat = Q[i].inf.flatten()
            Q_i_sup_flat = Q[i].sup.flatten()
            Qnonempty[i] = bool(np.any(Q_i_inf_flat != 0) or np.any(Q_i_sup_flat != 0))
            Q_i = Q[i]  # Keep as Interval for matrix multiplication (uses @ operator)
        else:
            # Handle different types: numeric array, sparse matrix, etc.
            try:
                import scipy.sparse
                if scipy.sparse.issparse(Q[i]):
                    # For sparse matrices, check if any element is non-zero
                    Qnonempty[i] = Q[i].nnz > 0
                    Q_i = Q[i]  # Keep as sparse matrix
                else:
                    # For regular arrays
                    Q_i = np.asarray(Q[i])
                    if Q_i.size > 0:
                        Q_i_flat = Q_i.flatten()
                        # Use .item() to extract scalar from np.any result
                        Qnonempty[i] = bool(np.any(Q_i_flat).item() if hasattr(np.any(Q_i_flat), 'item') else np.any(Q_i_flat))
                    else:
                        Qnonempty[i] = False
            except ImportError:
                # No scipy, assume regular array
                Q_i = np.asarray(Q[i])
                if Q_i.size > 0:
                    Q_i_flat = Q_i.flatten()
                    Qnonempty[i] = bool(np.any(Q_i_flat).item() if hasattr(np.any(Q_i_flat), 'item') else np.any(Q_i_flat))
                else:
                    Qnonempty[i] = False
        
        if Qnonempty[i]:
            # Pure quadratic evaluation
            quadMat = Zmat1.T @ Q_i @ Zmat2
            # Use column-major order (F) to match MATLAB's flattening
            Z[i, :] = quadMat.flatten(order='F')

    # Generate new zonotope
    tmp_sum = np.sum(Qnonempty)
    if tmp_sum < 1 or withinTol(tmp_sum, 1):
        # Single or no non-empty Q matrices
        c = Z[:, 0:1]  # First column as center
        G_sum = np.sum(np.abs(Z[:, 1:]), axis=1, keepdims=True)
        return Zonotope(c, G_sum)
    else:
        # Multiple non-empty Q matrices
        c = Z[:, 0:1]  # First column as center
        G_filtered = nonzeroFilter(Z[:, 1:])
        return Zonotope(c, G_filtered)


def _aux_quadMapSingleMatZono(Z: Zonotope, Q) -> Zonotope:
    """
    Compute an over-approximation of the quadratic map
    {x_i = x^T Q{i} x | x ∈ Z} 
    of a zonotope according to Theorem 1 in [1], where Q is a matrix zonotope
    """
    # zonotope Z_D for the center of the matrix zonotope
    Q_ = [None] * len(Q)
    
    for i in range(len(Q)):
        # Q_{i} = Q{i}.C  # Access the center of matZonotope
        Q_[i] = Q[i].C
    
    Z_D = quadMap(Z, Q_)
    
    # zonotopes Z_Kj for the generator of the matrix zonotope
    Z_K = [None] * Q[0].numgens()
    
    for j in range(len(Z_K)):
        for i in range(len(Q)):
            # Q_{i} = Q{i}.G(:,:,j)  # Access the j-th generator of matZonotope
            Q_[i] = Q[i].G[:, :, j]
        
        temp = quadMap(Z, Q_)
        Z_K[j] = np.hstack([temp.c, temp.G])
    
    # overall zonotope
    if len(Z_K) > 0:
        G_combined = np.hstack(Z_K)
        Zquad = Zonotope(Z_D.c, np.hstack([Z_D.G, G_combined]))
    else:
        Zquad = Z_D
    
    # Compact the result
    from .compact_ import compact_
    Zquad = compact_(Zquad, 'zeros', np.finfo(float).eps)
    
    return Zquad 