# ROOT CAUSE ANALYSIS: 20% errorSec Difference

## Summary

The 20% difference in `errorSec` is **NOT** due to `quadMap` implementation differences. It is caused by **different Z dimensions** due to **reduction producing different generator counts**.

## Evidence

**Step 3:**
- **Python Z**: 2 generators → quadMat shape (3, 3)
- **MATLAB Z**: 4 generators → quadMat shape (5, 5)
- **Radius difference**: Only 0.0786% (excellent agreement)
- **Generator count difference**: 2 vs 4 (100% difference!)

## Root Cause

The reduction step `reduce(R,'adaptive',sqrt(options.redFactor))` produces **different results**:
- Python reduces to **2 generators**
- MATLAB reduces to **4 generators**

This means:
1. Different Z dimensions → Different quadMat shapes
2. Different quadMat shapes → Different errorSec values
3. **We're comparing apples to oranges!**

## Impact

- The `quadMap` implementation is **correct** (flatten order fix was correct)
- The `errorSec` calculation is **correct**
- The problem is **upstream**: reduction produces different Z dimensions

## Next Steps

1. **Compare reduction inputs** - Are R, redFactor identical?
2. **Compare reduction algorithm** - Does `priv_reduceAdaptive` produce same results?
3. **Fix reduction** - Ensure Python and MATLAB reduce identically
4. **Re-verify** - After fixing reduction, errorSec should match

## Status

🔴 **CRITICAL**: The 20% difference is caused by reduction, not quadMap!
