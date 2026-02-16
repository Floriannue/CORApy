# Final Investigation Summary

## Root Cause: Increasing Relative Differences in errorSec

The divergence in `errorSec` (20.36% → 24.31% → 24.88% → 28.93%) is caused by **how Python handles Interval matrices in `quadMap`**.

## Key Findings

### 1. MATLAB's quadMap Cannot Accept Interval
- Test showed: `quadMap` fails with "Invalid data type" when given Interval
- But MATLAB code calls `quadMap(Z,H)` where H should be Interval
- **Conclusion**: H must be converted to numeric before `quadMap` call

### 2. Hessian Type Difference
- **tensorOrder == 2**: Uses `setHessian('int')` → Returns Interval Hessian
- **tensorOrder == 3**: Uses `setHessian('standard')` → Should return NUMERIC Hessian

### 3. Python's quadMap Fix
- Changed from `center()` (midpoint) to `max(abs(inf), abs(sup))` (conservative)
- This is correct in principle but differences persist

### 4. Hypothesis
MATLAB's `setHessian('standard')` returns **numeric** Hessian, so `quadMap` receives numeric matrices.
Python might be:
- Returning Interval Hessian even with 'standard'
- Or converting numeric to Interval somewhere
- Or the matrix multiplication produces Interval even with numeric inputs

## Next Steps

1. **Verify H type**: Check if Python's H is numeric or Interval when `setHessian('standard')` is used
2. **Compare H values**: If H types match, compare actual H values between Python and MATLAB
3. **Check matrix multiplication**: Verify if `Zmat.T @ Q_i @ Zmat` produces Interval when Q_i is numeric
4. **Fix if needed**: Ensure Python's 'standard' mode returns numeric Hessian like MATLAB

## Current Status

- ✅ Fix implemented: `max(abs(inf), abs(sup))` in `quadMap`
- ⚠️ Differences persist: 20-29% (need to verify H type)
- 🔍 Investigation ongoing: Checking if H type matches between Python and MATLAB
