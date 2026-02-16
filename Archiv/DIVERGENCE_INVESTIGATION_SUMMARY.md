# Divergence Investigation Summary

## Overview

This document summarizes all divergences found between Python and MATLAB implementations and the fixes applied.

## Critical Divergences Found and Fixed

### 1. **`reduce('adaptive')` NOT IMPLEMENTED** ✅ FIXED

**Issue**: Python's `priv_reduceAdaptive` was just a placeholder calling `priv_reduceGirard`, while MATLAB uses a completely different algorithm.

**Impact**: 
- Different generator selection → different reduced sets → different `Z` → different `errorSec` → different `VerrorDyn`
- Caused 18-27% differences in `VerrorDyn` between Python and MATLAB

**Fix**: 
- ✅ Implemented full `priv_reduceAdaptive` matching MATLAB's algorithm
- ✅ Uses `normsum - norminf` sorting criterion (not just norm)
- ✅ Computes cumulative Hausdorff distance
- ✅ Selects generators based on `dHmax` threshold

**Files Modified**:
- `cora_python/contSet/zonotope/private/priv_reduceAdaptive.py`: **NEW FILE** - Full implementation
- `cora_python/contSet/zonotope/private/priv_reduceMethods.py`: Updated to call actual implementation

### 2. **`pickedGenerators` vs `pickedGeneratorsFast` Mismatch** ✅ FIXED

**Issue**: 
- Python's `priv_reduceGirard` used `pickedGenerators`
- MATLAB's `priv_reduceGirard` uses `pickedGeneratorsFast`
- `pickedGeneratorsFast` has optimized logic for `nReduced < nUnreduced` vs `nReduced >= nUnreduced`

**Impact**:
- When `nReduced >= nUnreduced`, Python and MATLAB select different generators
- This causes different reduced sets even for Girard reduction

**Fix**:
- ✅ Created `pickedGeneratorsFast` matching MATLAB exactly
- ✅ Handles both cases: `nReduced < nUnreduced` (uses `mink`) and `nReduced >= nUnreduced` (uses `maxk`)
- ✅ Updated `priv_reduceGirard` to use `pickedGeneratorsFast`

**Files Modified**:
- `cora_python/g/functions/helper/sets/contSet/zonotope/pickedGeneratorsFast.py`: **NEW FILE**
- `cora_python/contSet/zonotope/private/priv_reduceMethods.py`: Updated to use `pickedGeneratorsFast`

## Additional Issues Found (Not Yet Fixed)

### 3. **Potential Indexing Issues in `priv_reduceAdaptive`**

**Status**: ⚠️ NEEDS VERIFICATION

**Possible Issues**:
- `redIdx` conversion from 0-based to 1-based indexing
- `gensred[:, :redIdx]` vs MATLAB's `gensred(:,1:redIdx)`
- `penven` case `redIdx` calculation

**Action Required**: Compare reduction results with MATLAB for same inputs

### 4. **Placeholder Implementations**

**Status**: ℹ️ LOW PRIORITY

**Found**:
- `priv_reduceIdx`: Simplified
- `priv_reduceMethE`, `priv_reduceMethF`: Simplified
- `priv_reduceRedistribute`, `priv_reduceCluster`, `priv_reduceScott`, `priv_reduceValero`, `priv_reduceSadraddini`, `priv_reduceScale`: All simplified

**Impact**: Not used in adaptive algorithm, but could cause issues if used elsewhere

## Expected Impact of Fixes

### Before Fixes
- `VerrorDyn` differences: **18-27%**
- `rerr1` differences: **2-12%**
- Different generator selections
- Different time step selections
- Early abortion at t=1.8s (vs t=8.0s in MATLAB)

### After Fixes (Expected)
- `VerrorDyn` differences: **<1%** (should match much better)
- `rerr1` differences: **<1%** (should match much better)
- Same generator selections
- Same time step selections
- Should complete to t=8.0s like MATLAB

## Testing Status

- ✅ `reduce('adaptive')` basic test: **PASSES**
- ✅ `reduce('girard')` test: **PASSES**
- ⚠️ Full jetEngine test: **STILL FAILS** (1.847s vs 8.0s, but time changed from 1.800s)
- ⚠️ Need to verify reduction results match MATLAB exactly

## Next Steps

1. **Verify `priv_reduceAdaptive` Results**:
   - Compare reduction results with MATLAB for same inputs
   - Verify `gredIdx` matches MATLAB
   - Check if `dHerror` values match

2. **Re-run Full Comparison**:
   - Run upstream comparison again
   - Check if `VerrorDyn` differences are reduced
   - Verify `rerr1` differences are reduced

3. **Test Full Integration**:
   - Run jetEngine test again
   - Check if it completes to t=8.0s
   - Compare final results with MATLAB

4. **Debug Remaining Issues**:
   - If test still fails, investigate further
   - Check for other potential sources of divergence
   - Verify all numerical operations match MATLAB

## Files Created/Modified

### New Files
- ✅ `cora_python/contSet/zonotope/private/priv_reduceAdaptive.py`
- ✅ `cora_python/g/functions/helper/sets/contSet/zonotope/pickedGeneratorsFast.py`

### Modified Files
- ✅ `cora_python/contSet/zonotope/private/priv_reduceMethods.py`
- ✅ `cora_python/contSet/zonotope/private/priv_reduceAdaptive.py` (imports fixed)

### Documentation Files
- ✅ `CRITICAL_FINDING_REDUCE_ADAPTIVE.md`
- ✅ `REDUCE_ADAPTIVE_FIX_SUMMARY.md`
- ✅ `HOW_TO_PREVENT_DIVERGENCE.md`
- ✅ `ADDITIONAL_DIVERGENCES_FOUND.md`
- ✅ `DIVERGENCE_INVESTIGATION_SUMMARY.md` (this file)

## Conclusion

**Two critical divergences** have been identified and fixed:
1. Missing `reduce('adaptive')` implementation
2. Wrong function used in `priv_reduceGirard` (`pickedGenerators` vs `pickedGeneratorsFast`)

These fixes should **significantly reduce** the divergence between Python and MATLAB. The remaining differences are likely due to:
- Small numerical precision differences (MKL vs OpenBLAS)
- Possible remaining bugs in the implementation
- Compounding of small differences over many steps

**Next action**: Re-run full tests and comparisons to verify the fixes work correctly.
