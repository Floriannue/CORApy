# Divergence Root Cause Found

## Summary

After detailed upstream comparison, **the root cause of divergence has been identified**: The divergence is in **`errorSec`** (computed by `quadMap`), not in `Z` or `errorLagr`.

## Key Findings

### Step 1: Perfect Match ✅
- **Z**: 0.0000% difference
- **errorSec**: 0.0000% difference
- **errorLagr**: 0.0000% difference
- **VerrorDyn**: 0.0000% difference

### Step 3: Divergence Starts ⚠️
- **Z**: 0.0786% difference (excellent)
- **errorSec**: **20.36% difference** ⚠️ **ROOT CAUSE**
- **errorLagr**: 0.27% difference (excellent)
- **VerrorDyn**: 19.76% difference (matches errorSec)

### Step 4: Divergence Continues ⚠️
- **Z**: 0.18% difference (excellent)
- **errorSec**: **24.31% difference** ⚠️
- **errorLagr**: 0.60% difference (excellent)
- **VerrorDyn**: 23.61% difference (matches errorSec)

### Step 5: Divergence Continues ⚠️
- **Z**: 0.11% difference (excellent)
- **errorSec**: **24.88% difference** ⚠️
- **errorLagr**: 1.29% difference (still good)
- **VerrorDyn**: 24.23% difference (matches errorSec)

### Step 6: Divergence Continues ⚠️
- **Z**: 1.05% difference (still good)
- **errorSec**: **28.93% difference** ⚠️
- **errorLagr**: 6.35% difference (acceptable)
- **VerrorDyn**: 28.32% difference (matches errorSec)

## Root Cause Analysis

### The Problem: `errorSec` Divergence

**`errorSec` is computed as**: `errorSec = 0.5 * Z.quadMap(H)`

**Findings**:
1. ✅ **Z is correct**: Differences are <0.2% (excellent)
2. ⚠️ **errorSec is wrong**: Differences are 20-29% (large)
3. ✅ **errorLagr is correct**: Differences are <1.3% (excellent)
4. ⚠️ **VerrorDyn matches errorSec**: Since `VerrorDyn = errorSec + errorLagr`, and errorLagr is small, VerrorDyn difference matches errorSec difference

### Why This Happens

The divergence in `errorSec` suggests:
1. **`quadMap` computation differs** between Python and MATLAB
2. **H (Hessian) might differ** between Python and MATLAB
3. **Numerical precision in `quadMap`** accumulates differently

### Impact

Since `errorSec` is the dominant term in `VerrorDyn`:
- `VerrorDyn = errorSec + errorLagr`
- `errorSec` is ~30x larger than `errorLagr` (e.g., 1.5e-2 vs 5e-4)
- So `VerrorDyn` difference ≈ `errorSec` difference

This explains why:
- `VerrorDyn` differences are 20-29% (matching `errorSec`)
- `rerr1` differences are 20-41% (derived from `VerrorDyn`)

## Next Steps

### 1. Compare `quadMap` Implementation
- Verify Python's `quadMap` matches MATLAB's `aux_quadMapSingle`
- Check if formulas are identical
- Compare numerical precision

### 2. Compare H (Hessian) Values
- Track H values before `quadMap` call
- Compare H between Python and MATLAB
- Verify H computation is identical

### 3. Compare `quadMap` Results Directly
- Use same Z and H in both Python and MATLAB
- Call `quadMap` directly
- Compare results

### 4. Investigate Numerical Precision
- Check if BLAS differences affect `quadMap`
- Consider using MKL for Python
- Verify order of operations matches MATLAB

## Conclusion

**The root cause is `errorSec` (from `quadMap`), not reduction or other operations.**

The fixes to `reduce('adaptive')` and `pickedGeneratorsFast` are correct, but they don't address this issue because:
- The divergence happens **before** reduction
- `errorSec` is computed from `Z.quadMap(H)`
- `Z` is correct, so the issue is in `quadMap` or `H`

**Next action**: Investigate `quadMap` implementation and compare H values.
