# Continued Investigation Summary

## Status

I've continued investigating the 2 vs 4 generators difference in `Rlintp` (Step 2 Run 2). Here's what I've verified and what remains:

## Verified Correct

1. **`vecnorm` computation**: ✅ Correct
   - `np.linalg.norm(gensdiag, axis=0, ord=2)` correctly computes the 2-norm of each column
   - Matches MATLAB's `vecnorm(gensdiag,2)`

2. **`gensred` indexing**: ✅ Correct
   - `Gabs[:, idx[last0Idx:]]` matches MATLAB's `Gabs(:,idx(last0Idx+1:end))`

3. **`Gred` indexing**: ✅ Correct
   - `gensred[:, :redIdx]` matches MATLAB's `gensred(:,1:redIdx)` when `redIdx` is 1-based

4. **`Gunred` indexing**: ✅ Correct
   - `idx[last0Idx + redIdx:]` matches MATLAB's `idx(last0Idx+redIdx+1:end)`

5. **`redIdx` conversion**: ✅ Correct
   - Converting from 0-based to 1-based appears correct

## Root Cause Hypothesis

The divergence must come from **different computed values**, not indexing bugs:

1. **Different `h` values**: The computed Hausdorff distance estimates differ
2. **Different `dHmax` values**: The maximum admissible distance differs
3. **Floating-point precision**: The `h <= dHmax` comparison might be affected by numerical differences

## Current Blocker

**MATLAB's `initReach_tracking` is empty for Step 2 Run 2**, preventing direct comparison of:
- `dHmax` values
- `h_computed` arrays
- `redIdx` values

## Tools Created

1. `compare_reduction_params.py` - Compares reduction parameters
2. `compare_generator_counts.py` - Compares generator counts
3. `find_tracking_entries.py` - Finds entries with tracking data
4. `test_vecnorm.py` - Verifies vecnorm computation
5. `test_reduction_logic.py` - Tests reduction logic
6. `test_gunred_indexing.py` - Verifies Gunred indexing
7. `extract_python_reduction_params.py` - Extracts Python reduction parameters

## Next Steps

1. **Re-run MATLAB with tracking enabled** to capture `initReach_tracking` for Step 2 Run 2
2. **Compare Step 3 Run 2** (which has tracking in both) to see if the pattern is consistent
3. **Add detailed logging** to capture `h`, `dHmax`, and `redIdx` values during reduction
4. **Check for floating-point precision issues** in the `h <= dHmax` comparison

## Key Finding

The indexing logic is **correct**. The issue is in the **computed values** (`h`, `dHmax`, or both), which requires comparing actual runtime data between Python and MATLAB.
