"""
mtimes - Overloaded '*' operator for intervals

Syntax:
    res = factor1 * factor2
    res = mtimes(factor1, factor2)

Inputs:
    factor1 - interval object, numeric
    factor2 - interval object, numeric, contSet object

Outputs:
    res - interval

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Last update: 04-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
from typing import Union
from .interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

# Optional scipy import for sparse matrix support
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def mtimes(factor1: Union[Interval, np.ndarray, float, int], 
          factor2: Union[Interval, np.ndarray, float, int]) -> Interval:
    """
    Overloaded '*' operator for intervals
    
    Args:
        factor1: Interval object or numeric
        factor2: Interval object, numeric, or contSet object
        
    Returns:
        res: Interval result
    """
    # Handle contSet cases - interval * zonotope/polyZonotope/conZonotope/zonoBundle
    if hasattr(factor2, '__class__'):
        cls_name = factor2.__class__.__name__
        # Convert factor1 to interval if needed
        if not isinstance(factor1, Interval) and cls_name in [
            'Zonotope', 'PolyZonotope', 'ConZonotope', 'ZonoBundle']:
            factor1 = _numeric_to_Interval(factor1)
        if cls_name == 'Zonotope':
            return _aux_mtimes_zonotope(factor1, factor2)
        if cls_name == 'PolyZonotope':
            return _aux_mtimes_polyZonotope(factor1, factor2)
        if cls_name == 'ConZonotope':
            return _aux_mtimes_conZonotope(factor1, factor2)
        if cls_name == 'ZonoBundle':
            return _aux_mtimes_zonoBundle(factor1, factor2)
    
    # Other contSet cases not supported
    if (hasattr(factor2, 'precedence') and 
        not isinstance(factor2, Interval) and 
        hasattr(factor2, '__class__')):
        raise CORAerror('CORA:noops', f'Operation not supported between {type(factor1)} and {type(factor2)}')
    
    # Convert factor1 to interval if needed (factor1 must be interval in this function)
    if not isinstance(factor1, Interval):
        factor1 = _numeric_to_Interval(factor1)
    
    # Keep factor2 as-is if numeric (for optimization in _mtimes_nonsparse)
    # Only convert to interval if needed for scalar cases
    factor2_is_numeric = not isinstance(factor2, Interval)
    
    # Scalar case - both must be intervals for scalar functions
    if _is_scalar(factor1):
        if factor2_is_numeric:
            # Convert numeric to interval for processing
            factor2 = _numeric_to_Interval(factor2)
            factor2_is_numeric = False
            if _is_scalar(factor2):
                return _mtimes_scalar(factor1, factor2)
            else:
                # factor1 is scalar, factor2 is matrix
                return _mtimes_scalar_matrix(factor1, factor2)
        else:
            if _is_scalar(factor1) and _is_scalar(factor2):
                return _mtimes_scalar(factor1, factor2)
            if _is_scalar(factor1):  # factor2 is a matrix
                return _mtimes_scalar_matrix(factor1, factor2)
    
    if not factor2_is_numeric and _is_scalar(factor2):  # factor1 is a matrix
        return _mtimes_matrix_scalar(factor1, factor2)
    
    # Handle case where factor2 is numeric scalar and factor1 is matrix
    if factor2_is_numeric:
        factor2_scalar = factor2
        if np.isscalar(factor2_scalar):
            factor2 = _numeric_to_Interval(factor2_scalar)
            factor2_is_numeric = False
            return _mtimes_matrix_scalar(factor1, factor2)
    
    # Matrix case
    # Pass numeric factor2 directly to _mtimes_nonsparse for optimization (matches MATLAB)
    if not _is_sparse(factor1) and not (not factor2_is_numeric and _is_sparse(factor2)):
        return _mtimes_nonsparse(factor1, factor2)
    else:
        # Convert factor2 to interval for sparse case
        if factor2_is_numeric:
            factor2 = _numeric_to_Interval(factor2)
        return _mtimes_sparse(factor1, factor2)


def _numeric_to_Interval(value):
    """Convert numeric value to interval"""
    if isinstance(value, (int, float)):
        return Interval(np.array([value]))
    elif isinstance(value, np.ndarray):
        return Interval(value, value)
    else:
        raise CORAerror('CORA:wrongInput', f'Cannot convert {type(value)} to interval')


def _is_scalar(obj):
    """Check if object represents a scalar"""
    if isinstance(obj, Interval):
        return obj.inf.size == 1
    return np.isscalar(obj)


def _is_sparse(obj):
    """Check if object is sparse"""
    if not HAS_SCIPY:
        return False
    
    if isinstance(obj, Interval):
        return sp.issparse(obj.inf) or sp.issparse(obj.sup)
    return sp.issparse(obj)


def _mtimes_scalar(factor1: Interval, factor2: Interval) -> Interval:
    """
    Multiply two scalar intervals
    Matches MATLAB's aux_mtimes_scalar implementation
    """
    # MATLAB: obtain possible values
    # Check if factor1 or factor2 is numeric (converted to interval)
    factor1_is_numeric = np.allclose(factor1.inf, factor1.sup)
    factor2_is_numeric = np.allclose(factor2.inf, factor2.sup)
    
    if factor1_is_numeric:
        # MATLAB: if isnumeric(factor1)
        # res = factor2;
        res = factor2
        factor1_val = factor1.inf.item()
        if factor1_val == 0:
            # MATLAB: as 0*[-inf,inf] = {0*x|-inf<x<inf}={0}
            possible_values = [0]
        else:
            # MATLAB: possibleValues = [factor1*factor2.inf, factor1*factor2.sup];
            possible_values = [factor1_val * factor2.inf.item(), factor1_val * factor2.sup.item()]
    elif factor2_is_numeric:
        # MATLAB: elseif isnumeric(factor2)
        # res = factor1;
        res = factor1
        factor2_val = factor2.inf.item()
        if factor2_val == 0:
            # MATLAB: as 0*[-inf,inf] = {0*x|-inf<x<inf}={0}
            possible_values = [0]
        else:
            # MATLAB: possibleValues = [factor1.inf*factor2, factor1.sup*factor2];
            possible_values = [factor1.inf.item() * factor2_val, factor1.sup.item() * factor2_val]
    else:
        # MATLAB: else (both are intervals)
        # res = factor1;
        res = factor1
        # MATLAB: possibleValues = [factor1.inf*factor2.inf, factor1.inf*factor2.sup, ...
        #     factor1.sup*factor2.inf, factor1.sup*factor2.sup];
        possible_values = [
            factor1.inf.item() * factor2.inf.item(),
            factor1.inf.item() * factor2.sup.item(),
            factor1.sup.item() * factor2.inf.item(),
            factor1.sup.item() * factor2.sup.item()
        ]
    
    # MATLAB: res.inf = min(possibleValues);
    # MATLAB: res.sup = max(possibleValues);
    res.inf = np.array([np.min(possible_values)])
    res.sup = np.array([np.max(possible_values)])
    
    return res


def _mtimes_scalar_matrix(factor1: Interval, factor2: Interval) -> Interval:
    """
    Multiply scalar interval with matrix interval
    Matches MATLAB's aux_mtimes_scalar_matrix implementation
    """
    # MATLAB: obtain possible values
    # Check if factor1 is numeric (converted to interval)
    factor1_is_numeric = np.allclose(factor1.inf, factor1.sup)
    
    if factor1_is_numeric:
        # MATLAB: if isnumeric(factor1)
        factor1_val = factor1.inf.item()
        if factor1_val < 0:
            # MATLAB: infimum and supremum
            # res = interval(factor1*factor2.sup, factor1*factor2.inf);
            return Interval(factor1_val * factor2.sup, factor1_val * factor2.inf)
        elif factor1_val > 0:
            # MATLAB: infimum and supremum
            # res = interval(factor1*factor2.inf, factor1*factor2.sup);
            return Interval(factor1_val * factor2.inf, factor1_val * factor2.sup)
        else:  # factor1_val == 0
            # MATLAB: as 0*[-inf,inf] = {0*x|-inf<x<inf}={0}
            # res = interval(zeros(size(factor2.inf)));
            return Interval(np.zeros_like(factor2.inf))
    else:
        # MATLAB: else
        # res = factor1.*factor2;
        # Use element-wise multiplication
        return factor1.times(factor2)


def _mtimes_matrix_scalar(factor1: Interval, factor2: Interval) -> Interval:
    """
    Multiply matrix interval with scalar interval
    Matches MATLAB's aux_mtimes_matrix_scalar implementation
    """
    # MATLAB: obtain possible values
    # Check if factor2 is numeric (converted to interval)
    # Handle sparse matrices
    import scipy.sparse
    if scipy.sparse.issparse(factor2.inf) or scipy.sparse.issparse(factor2.sup):
        # For sparse matrices, check if they're equal by converting to dense
        factor2_is_numeric = np.allclose(factor2.inf.toarray() if scipy.sparse.issparse(factor2.inf) else factor2.inf,
                                         factor2.sup.toarray() if scipy.sparse.issparse(factor2.sup) else factor2.sup)
    else:
        factor2_is_numeric = np.allclose(factor2.inf, factor2.sup)
    
    if factor2_is_numeric:
        # MATLAB: if isnumeric(factor2)
        # Handle sparse matrices
        import scipy.sparse
        if scipy.sparse.issparse(factor2.inf):
            factor2_val = factor2.inf.toarray().item()
        else:
            factor2_val = factor2.inf.item()
        if factor2_val < 0:
            # MATLAB: infimum and supremum
            # res = interval(factor2*factor1.sup, factor2*factor1.inf);
            return Interval(factor2_val * factor1.sup, factor2_val * factor1.inf)
        elif factor2_val > 0:
            # MATLAB: infimum and supremum
            # res = interval(factor2*factor1.inf, factor2*factor1.sup);
            return Interval(factor2_val * factor1.inf, factor2_val * factor1.sup)
        else:  # factor2_val == 0
            # MATLAB: as 0*[-inf,inf] = {0*x|-inf<x<inf}={0}
            # res = interval(zeros(size(factor1.inf)));
            return Interval(np.zeros_like(factor1.inf))
    else:
        # MATLAB: else
        # res = factor1.*factor2;
        # Use element-wise multiplication
        return factor1.times(factor2)


def _mtimes_nonsparse(factor1: Interval, factor2: Union[Interval, np.ndarray]) -> Interval:
    """
    Matrix multiplication for non-sparse intervals
    Matches MATLAB's aux_mtimes_nonsparse implementation
    """
    # MATLAB: compute fast algorithm
    # [m, k] * [k, n] = [m, n]
    # -> [m, k, 1] .* [1, k, n] = [m, k, n]
    
    # MATLAB: [m, ~] = size(factor1);
    # Get first dimension m and ensure 2D
    if factor1.inf.ndim == 1:
        m = 1
        f1_inf = factor1.inf.reshape(1, -1)
        f1_sup = factor1.sup.reshape(1, -1)
        k1 = f1_inf.shape[1]
    else:
        m = factor1.inf.shape[0]
        f1_inf = factor1.inf
        f1_sup = factor1.sup
        k1 = f1_inf.shape[1] if f1_inf.ndim >= 2 else 1
    
    # MATLAB handles numeric factor2 differently
    if isinstance(factor2, np.ndarray) or (not isinstance(factor2, Interval) and not hasattr(factor2, 'inf')):
        # MATLAB: if isnumeric(factor2)
        # Convert numeric to interval for processing, but use it directly
        factor2_numeric = factor2
        # Convert scalar to array for consistent handling
        if np.isscalar(factor2_numeric):
            factor2_numeric = np.array(factor2_numeric)
        elif not isinstance(factor2_numeric, np.ndarray):
            factor2_numeric = np.asarray(factor2_numeric)
        
        if factor2_numeric.ndim == 0:
            # Scalar case - should have been handled earlier, but handle it here too
            factor2_numeric = factor2_numeric.reshape(1)
        
        if factor2_numeric.ndim == 1:
            extSize = (1, factor2_numeric.shape[0])
            k2 = factor2_numeric.shape[0]
        else:
            # MATLAB: extSize = [1, size(factor2)];
            # For 2D array of shape (k, n), extSize = [1, k, n]
            # This creates a leading dimension of 1 for broadcasting
            extSize = (1,) + factor2_numeric.shape
            # For matrix multiplication [m, k] * [k, n], k2 is the number of rows in factor2
            # which is the first dimension (index 0) of factor2
            k2 = factor2_numeric.shape[0]
        
        # Check dimension compatibility
        if k1 != k2:
            raise CORAerror('CORA:wrongInput', 
                           f'Matrix dimensions incompatible: {factor1.inf.shape} * {factor2_numeric.shape}')
        
        # Reshape factor2
        f2 = factor2_numeric.reshape(extSize)
        
        # Create factor1 with trailing dimension
        if f1_inf.ndim == 1:
            f1_inf_bc = f1_inf.reshape(1, -1, 1)
            f1_sup_bc = f1_sup.reshape(1, -1, 1)
        else:
            f1_inf_bc = f1_inf[:, :, np.newaxis]
            f1_sup_bc = f1_sup[:, :, np.newaxis]
        
        # MATLAB: res = factor1 .* factor2;
        # Element-wise multiplication
        res_inf = f1_inf_bc * f2
        res_sup = f1_sup_bc * f2
        
        # Handle NaN cases
        res_inf = np.where(np.isnan(res_inf), 0, res_inf)
        res_sup = np.where(np.isnan(res_sup), 0, res_sup)
        
        # MATLAB: res.inf = sum(res.inf, 2); res.sup = sum(res.sup, 2);
        # Sum along dimension 1 (which is axis 1 in 0-indexed): [m, k, n] -> [m, n]
        # MATLAB's sum(..., 2) means sum along the 2nd dimension (1-indexed), which is axis 1 (0-indexed)
        inf_result = np.sum(res_inf, axis=1)
        sup_result = np.sum(res_sup, axis=1)
        
        # MATLAB: res = reshape(res, m, [])
        # MATLAB directly uses the sums - no min/max needed here
        # When factor1 was originally numeric (converted to interval), factor1.inf == factor1.sup
        # So res_inf == res_sup, and inf_result == sup_result
        # When factor1 is a true interval, we still use the sums directly
        # The interval bounds come from the sum of inf and sum of sup separately
        
        # MATLAB: res = reshape(res, m, [])
        if inf_result.ndim == 1:
            inf_result = inf_result.reshape(1, -1)
            sup_result = sup_result.reshape(1, -1)
        else:
            inf_result = inf_result.reshape(m, -1)
            sup_result = sup_result.reshape(m, -1)
        
        return Interval(inf_result, sup_result)
    
    else:
        # MATLAB: else (factor2 is interval)
        # MATLAB: extSize = [1, size(factor2)];
        if factor2.inf.ndim == 1:
            extSize = (1, factor2.inf.shape[0])
            k2 = factor2.inf.shape[0]
            n = 1
        else:
            extSize = (1,) + factor2.inf.shape
            k2 = factor2.inf.shape[0]
            n = factor2.inf.shape[1] if factor2.inf.ndim >= 2 else 1
        
        # Check dimension compatibility
        if k1 != k2:
            raise CORAerror('CORA:wrongInput', 
                           f'Matrix dimensions incompatible: {factor1.inf.shape} * {factor2.inf.shape}')
        
        # MATLAB: factor2.inf = reshape(factor2.inf, extSize);
        f2_inf = factor2.inf.reshape(extSize)
        f2_sup = factor2.sup.reshape(extSize)
        
        # Create factor1 with trailing dimension
        if f1_inf.ndim == 1:
            f1_inf_bc = f1_inf.reshape(1, -1, 1)
            f1_sup_bc = f1_sup.reshape(1, -1, 1)
        else:
            f1_inf_bc = f1_inf[:, :, np.newaxis]
            f1_sup_bc = f1_sup[:, :, np.newaxis]
        
        # Reshape f2_inf and f2_sup for 3D broadcasting: [1, k, n]
        # f2_inf is (1, k) or (1, k, n) after reshape, need (1, k, n) for broadcasting
        # MATLAB: [m, k, 1] .* [1, k, n] = [m, k, n]
        if f2_inf.ndim == 2:
            # f2_inf is (1, k) after reshape, need (1, k, n) where n is the last dimension of original factor2
            # For 2D factor2 with shape (k, n), after reshape to extSize (1, k, n), we already have the right shape
            # But we need to ensure it has the right number of dimensions for broadcasting
            if len(extSize) == 2:
                # extSize = (1, k) - this is a vector case, n=1
                f2_inf_bc = f2_inf[:, :, np.newaxis]  # (1, k, 1)
                f2_sup_bc = f2_sup[:, :, np.newaxis]  # (1, k, 1)
            else:
                # extSize = (1, k, n) - already has the right shape
                f2_inf_bc = f2_inf
                f2_sup_bc = f2_sup
        else:
            # Already higher dimensional, ensure it has trailing dimension
            # extSize already includes all dimensions, just need to ensure compatibility
            if len(extSize) == f2_inf.ndim:
                # Shapes match, use as is
                f2_inf_bc = f2_inf
                f2_sup_bc = f2_sup
            else:
                # Need to add trailing dimension
                f2_inf_bc = f2_inf.reshape(extSize + (1,))
                f2_sup_bc = f2_sup.reshape(extSize + (1,))
        
        # MATLAB: res = factor1 .* factor2;
        # Element-wise multiplication: [m, k, 1] .* [1, k, n] = [m, k, n]
        products = [
            f1_inf_bc * f2_inf_bc,
            f1_inf_bc * f2_sup_bc,
            f1_sup_bc * f2_inf_bc,
            f1_sup_bc * f2_sup_bc
        ]
        
        # Stack and find min/max element-wise
        all_products = np.stack(products, axis=-1)
        
        # Handle NaN cases
        nan_mask = np.isnan(all_products)
        all_products[nan_mask] = 0
        
        inf_result = np.min(all_products, axis=-1)
        sup_result = np.max(all_products, axis=-1)
        
        # MATLAB: res.inf = sum(res.inf, 2); res.sup = sum(res.sup, 2);
        inf_result = np.sum(inf_result, axis=1)
        sup_result = np.sum(sup_result, axis=1)
        
        # MATLAB: res = reshape(res, m, [])
        if inf_result.ndim == 1:
            inf_result = inf_result.reshape(1, -1)
            sup_result = sup_result.reshape(1, -1)
        else:
            inf_result = inf_result.reshape(m, -1)
            sup_result = sup_result.reshape(m, -1)
        
        return Interval(inf_result, sup_result)


def _mtimes_sparse(factor1: Interval, factor2: Interval) -> Interval:
    """Matrix multiplication for sparse intervals"""
    # For sparse matrices, use the slower but more memory-efficient algorithm
    f1_shape = factor1.inf.shape
    f2_shape = factor2.inf.shape
    
    if len(f1_shape) == 1:
        m, k = 1, f1_shape[0]
    else:
        m, k = f1_shape
        
    if len(f2_shape) == 1:
        k2, n = f2_shape[0], 1
    else:
        k2, n = f2_shape
    
    if k != k2:
        raise CORAerror('CORA:wrongInput', 
                       f'Matrix dimensions incompatible: {factor1.inf.shape} * {factor2.inf.shape}')
    
    # Preallocate output bounds
    res_inf = np.zeros((m, n))
    res_sup = np.zeros((m, n))
    
    # Create temporary interval for row operations
    temp_interval = Interval(np.zeros(k), np.zeros(k))
    
    for i in range(m):
        # Get i-th row
        temp_interval.inf = factor1.inf[i, :]
        temp_interval.sup = factor1.sup[i, :]
        
        # Multiply row with matrix
        row_result = _mtimes_vector_matrix(temp_interval, factor2)
        res_inf[i, :] = row_result.inf
        res_sup[i, :] = row_result.sup
    
    return Interval(res_inf, res_sup)


def _mtimes_vector_matrix(vector_interval: Interval, matrix_interval: Interval) -> Interval:
    """Multiply vector interval with matrix interval"""
    k, n = matrix_interval.inf.shape
    
    # [k] .* [k, n] = [k, n]
    v_inf = vector_interval.inf[:, np.newaxis]
    v_sup = vector_interval.sup[:, np.newaxis]
    
    products = [
        v_inf * matrix_interval.inf,
        v_inf * matrix_interval.sup,
        v_sup * matrix_interval.inf,
        v_sup * matrix_interval.sup
    ]
    
    # Stack and handle NaN
    all_products = np.stack(products, axis=-1)
    nan_mask = np.isnan(all_products)
    all_products[nan_mask] = 0
    
    # Sum over k dimension
    sums = np.sum(all_products, axis=0)  # [n, 4]
    
    inf_result = np.min(sums, axis=-1)
    sup_result = np.max(sums, axis=-1)
    
    return Interval(inf_result, sup_result)


def _aux_mtimes_zonotope(I: Interval, Z):
    """
    Auxiliary function for interval matrix * zonotope
    See Theorem 3.3 in [1]
    
    Args:
        I: Interval matrix
        Z: Zonotope object
        
    Returns:
        Zonotope: Result of multiplication
    """
    from cora_python.contSet.zonotope import Zonotope
    from .center import center
    from .rad import rad
    
    # Get center and radius of interval matrix
    T = center(I)  # Center matrix
    S = rad(I)     # Radius matrix
    
    # Compute sum of absolute values: sum(abs([Z.c, Z.G]), axis=1)
    # MATLAB: Zabssum = sum(abs([Z.c,Z.G]),2)
    Z_c_G = np.hstack([Z.c, Z.G]) if Z.G.size > 0 else Z.c
    Zabssum = np.sum(np.abs(Z_c_G), axis=1, keepdims=True)
    
    # Compute new zonotope
    # MATLAB: Z.c = T*Z.c
    Z_c_new = T @ Z.c
    
    # MATLAB: Z.G = [T*Z.G, diag(S*Zabssum)]
    if Z.G.size > 0:
        Z_G_new = T @ Z.G
    else:
        Z_G_new = np.zeros((Z.c.shape[0], 0))
    
    # diag(S*Zabssum) - create diagonal matrix from vector
    S_Zabssum = S @ Zabssum  # Matrix-vector multiplication
    diag_S_Zabssum = np.diag(S_Zabssum.flatten())
    
    # Concatenate generators
    if Z_G_new.size > 0:
        Z_G_final = np.hstack([Z_G_new, diag_S_Zabssum])
    else:
        Z_G_final = diag_S_Zabssum
    
    return Zonotope(Z_c_new, Z_G_final)


def _aux_mtimes_polyZonotope(I: Interval, pZ):
    """
    Auxiliary function for interval matrix * polyZonotope
    Matches MATLAB's aux_mtimes_polyZonotope implementation.
    """
    from cora_python.contSet.polyZonotope import PolyZonotope
    from .center import center
    from .rad import rad

    # Work on a copy to avoid mutating input
    pZ_out = pZ.copy() if hasattr(pZ, 'copy') else PolyZonotope(pZ)

    # center and radius of interval matrix
    m = center(I)
    r = rad(I)

    # interval over-approximation of polyZonotope
    I_pZ = pZ_out.interval() if hasattr(pZ_out, 'interval') else Interval(pZ_out)
    s = np.abs(center(I_pZ)) + rad(I_pZ)

    # compute new polyZonotope
    pZ_out.c = m @ pZ_out.c
    if pZ_out.G.size > 0:
        pZ_out.G = m @ pZ_out.G
    if pZ_out.GI.size > 0:
        GI_new = m @ pZ_out.GI
        diag_term = np.diag((r @ s).flatten())
        pZ_out.GI = np.hstack([GI_new, diag_term]) if GI_new.size > 0 else diag_term
    else:
        pZ_out.GI = np.diag((r @ s).flatten())

    return pZ_out


def _aux_mtimes_conZonotope(I: Interval, cZ):
    """
    Auxiliary function for interval matrix * conZonotope
    Matches MATLAB's aux_mtimes_conZonotope implementation.
    """
    from cora_python.contSet.conZonotope import ConZonotope
    from .center import center
    from .rad import rad

    # Work on a copy to avoid mutating input
    cZ_out = ConZonotope(cZ)

    # center and radius of interval matrix
    m = center(I)
    r = rad(I)

    # absolute value of zonotope center and generators
    Z_c_G = np.hstack([cZ_out.c, cZ_out.G]) if cZ_out.G.size > 0 else cZ_out.c
    Zabssum = np.sum(np.abs(Z_c_G), axis=1, keepdims=True)

    # construct resulting conZonotope
    cZ_out.c = m @ cZ_out.c
    if cZ_out.G.size > 0:
        G_new = m @ cZ_out.G
    else:
        G_new = np.zeros((cZ_out.c.shape[0], 0))
    diag_term = np.diag((r @ Zabssum).flatten())
    cZ_out.G = np.hstack([G_new, diag_term]) if G_new.size > 0 else diag_term

    # extend constraint matrix
    if hasattr(cZ_out, 'A') and cZ_out.A.size > 0:
        cZ_out.A = np.hstack([cZ_out.A, np.zeros((cZ_out.A.shape[0], Zabssum.shape[0]))])

    return cZ_out


def _aux_mtimes_zonoBundle(I: Interval, zB):
    """
    Auxiliary function for interval matrix * zonoBundle
    Matches MATLAB's aux_mtimes_zonoBundle implementation.
    """
    from cora_python.contSet.zonoBundle import ZonoBundle

    zB_out = ZonoBundle(zB)
    for i in range(zB_out.parallelSets):
        zB_out.Z[i] = _aux_mtimes_zonotope(I, zB_out.Z[i])
    return zB_out


def _is_zero_Interval(obj: Interval) -> bool:
    """Check if interval represents zero"""
    return np.allclose(obj.inf, 0) and np.allclose(obj.sup, 0) 
