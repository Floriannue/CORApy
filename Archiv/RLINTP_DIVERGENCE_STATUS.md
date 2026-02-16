# Rlintp Divergence Status

## Current Problem

**Rlintp generator count divergence:**
- Python: 2 generators
- MATLAB: 4 generators

This occurs in Step 2 Run 2.

## Root Cause Analysis

### Chain of Divergence

1. **Rhom_tp** (before reduction): Both have 5 generators ✅
2. **Reduction in priv_reduceAdaptive**: 
   - Python: Reduces 5→2 generators (reduces 3)
   - MATLAB: Reduces 5→4 generators (reduces 1)
3. **Rend_tp** (after reduction):
   - Python: 2 generators
   - MATLAB: 4 generators
4. **Rlintp** (computed from Rend.tp):
   - Python: 2 generators
   - MATLAB: 4 generators

### Why Reduction Differs

The reduction algorithm `priv_reduceAdaptive` produces different `redIdx` values:
- Python: `redIdx = 3` (reduces 3 generators)
- MATLAB: `redIdx = 1` (reduces 1 generator)

This depends on:
1. **dHmax** value (computed from `diagpercent` and `Gbox`)
2. **h array** values (computed from `gensdiag`)
3. The comparison `h <= dHmax`

### Current Status

- ✅ **Reduction algorithm logic verified**: Indexing and array operations match MATLAB
- ⏳ **Reduction parameters not captured**: `initReach_tracking` doesn't have reduction details
- ⏳ **Need to compare dHmax and h values**: To find where divergence occurs

## Next Steps

1. **Fix reduction details capture**: Ensure `_reduction_details` is properly stored and read
2. **Compare dHmax values**: Check if Python and MATLAB compute the same `dHmax`
3. **Compare h arrays**: Check if `h_computed` values match between Python and MATLAB
4. **Compare redIdx computation**: Verify `redIdx = find(h <= dHmax, 1, 'last')` produces same result

## TODO

- Generate a lot of MATLAB-Python value pairs from tracking data
- Update Python tests along the whole chain with additional test cases using these values
