# quadMap Fix Verification

## Fix Applied: Column-Major Flattening

### Issue
Python was using row-major (C order) flattening, while MATLAB uses column-major (Fortran order) flattening for:
1. `quadMatoffdiag = quadMat + quadMat'` → `quadMatoffdiag(:)`
2. `kInd = tril(...)` → `kInd(:)`

### Fix
Changed both to use `order='F'` (Fortran/column-major order):
```python
# BEFORE:
quadMatoffdiag_flat = quadMatoffdiag.flatten()  # Row-major
G[i, gens:] = quadMatoffdiag_flat[kInd.flatten()]  # Row-major

# AFTER:
quadMatoffdiag_flat = quadMatoffdiag.flatten(order='F')  # Column-major
G[i, gens:] = quadMatoffdiag_flat[kInd.flatten(order='F')]  # Column-major
```

## Current Status

### Test Results
- **errorSec difference**: Still 20.36% (Step 3)
- **VerrorDyn difference**: 27-31% (Steps 12-20)
- **rerr1 difference**: 45-62% (Steps 12-20)

### Possible Causes for Remaining Differences

1. **Fix not fully effective**: The off-diagonal elements might not be the main contributor
2. **Other indexing issues**: There might be other places where row/column-major matters
3. **Numerical precision**: BLAS library differences (MATLAB MKL vs Python OpenBLAS)
4. **Sparse matrix handling**: Different sparse matrix operations
5. **Other logic differences**: There might be other subtle differences in the code

## Next Steps

1. ✅ Verify fix is in code (DONE)
2. 🔄 Run test to verify fix works correctly
3. 🔄 Compare actual quadMat values element-by-element
4. 🔄 Check if there are other places where flatten order matters
5. 🔄 Investigate numerical precision differences
