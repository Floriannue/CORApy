# Code Logic Comparison: MATLAB vs Python

## quadMap Function: Complete Chain Analysis

### 1. Input: H (Hessian)
- **MATLAB**: `H = nlnsys.hessian(...)` returns numeric (sparse) matrices for `setHessian('standard')`
- **Python**: `H = nlnsys.hessian(...)` returns numeric (sparse) matrices for `setHessian('standard')`
- **Status**: ✅ MATCHES

### 2. Input: Z (Zonotope)
- **MATLAB**: `Zmat = [Z.c, Z.G]` - horizontal concatenation
- **Python**: `Zmat = np.hstack([Z.c, Z.G])` - horizontal concatenation
- **Status**: ✅ MATCHES

### 3. Matrix Multiplication: quadMat = Zmat' * H * Zmat
- **MATLAB**: `quadMat = Zmat'*Q{i}*Zmat;`
- **Python**: `quadMat = Zmat.T @ Q_i @ Zmat`
- **Status**: ✅ MATCHES (transpose and matrix multiplication)

### 4. Diagonal Extraction
- **MATLAB**: `G(i,1:gens) = 0.5*diag(quadMat(2:gens+1,2:gens+1));`
  - MATLAB uses 1-based indexing: `quadMat(2:gens+1,2:gens+1)` = rows 2 to gens+1, cols 2 to gens+1
- **Python**: `quadMat_sub = quadMat[1:gens+1, 1:gens+1]` then `G[i, :gens] = 0.5 * np.diag(quadMat_sub)`
  - Python uses 0-based indexing: `quadMat[1:gens+1, 1:gens+1]` = rows 1 to gens, cols 1 to gens
- **Status**: ✅ MATCHES (correct 1-based to 0-based conversion)

### 5. Center Calculation
- **MATLAB**: `c(i,1) = quadMat(1,1) + sum(G(i,1:gens));`
- **Python**: `c[i, 0] = quadMat[0, 0] + np.sum(G[i, :gens])`
- **Status**: ✅ MATCHES

### 6. Off-Diagonal Elements (CRITICAL)
- **MATLAB**:
  ```matlab
  quadMatoffdiag = quadMat + quadMat';
  quadMatoffdiag = quadMatoffdiag(:);  % Column-major flattening
  kInd = tril(true(gens+1,gens+1),-1);
  G(i, gens+1:end) = quadMatoffdiag(kInd(:));  % Column-major flattening
  ```
- **Python (BEFORE FIX)**:
  ```python
  quadMatoffdiag = quadMat + quadMat.T
  quadMatoffdiag_flat = quadMatoffdiag.flatten()  # Row-major (WRONG!)
  kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
  G[i, gens:] = quadMatoffdiag_flat[kInd.flatten()]  # Row-major (WRONG!)
  ```
- **Python (AFTER FIX)**:
  ```python
  quadMatoffdiag = quadMat + quadMat.T
  quadMatoffdiag_flat = quadMatoffdiag.flatten(order='F')  # Column-major (CORRECT!)
  kInd = np.tril(np.ones((gens+1, gens+1), dtype=bool), -1)
  G[i, gens:] = quadMatoffdiag_flat[kInd.flatten(order='F')]  # Column-major (CORRECT!)
  ```
- **Status**: ✅ FIXED - Now matches MATLAB's column-major flattening

### 7. Final Zonotope Construction
- **MATLAB**: `Zquad = zonotope([c,sum(abs(G),2)])` or `zonotope([c,nonzeroFilter(G)])`
- **Python**: `Zonotope(np.hstack([c, np.sum(np.abs(G), axis=1, keepdims=True)]))` or `Zonotope(np.hstack([c, nonzeroFilter(G)]))`
- **Status**: ✅ MATCHES

## Root Cause Identified

The **off-diagonal element extraction** was using row-major flattening instead of MATLAB's column-major flattening. This would cause different elements to be selected, leading to incorrect generator values in the final zonotope.

## Fix Applied

Changed both flattening operations to use `order='F'` (Fortran/column-major order) to match MATLAB's behavior.
