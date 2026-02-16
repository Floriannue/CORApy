# Final Summary: Reduction Algorithm Divergence

## Root Cause Identified

**Python and MATLAB are reducing DIFFERENT INPUT ZONOTOPES:**

- **Python**: 8 generators before reduction → 2 generators after
- **MATLAB**: 13 generators before reduction → 4 generators after

## Key Findings

### 1. Input Mismatch
- Python has **8 generators** in `R` before reduction
- MATLAB has **13 generators** in `R` before reduction
- **Difference: 5 generators**

### 2. Reduction Results
- **Python**: `redIdx=2`, `final_generators=2`, `dHerror=8.3e-06`
- **MATLAB**: `redIdx=5`, `final_generators=4`, `dHerror=0.00447`

### 3. dHmax Comparison
- Python: `0.006948944389885456`
- MATLAB: `0.006943114379917405`
- **Difference: 0.08%** (acceptable, likely due to different input sizes)

### 4. Centers
- Python: `[0.00011355, -0.0001106]`
- MATLAB: `[0.0001136, -0.00011065]`
- Very close but not identical (likely due to different generator counts)

## Where R Comes From

In `linReach_adaptive.m`:
```matlab
R = reduce(Rdelta, 'adaptive', sqrt(options.redFactor));
[~,~,L0_3,options] = priv_abstractionError_adaptive(nlnsys,R,Rdelta,...);
```

So `R` is the result of reducing `Rdelta`. The `R` passed to `priv_abstractionError_adaptive` is already reduced.

## Next Steps

1. ✅ **Fixed MATLAB tracking** to capture generators correctly (centers now work)
2. ⏳ **Fix generator storage** in MATLAB (still showing shape `()`)
3. ⏳ **Track Rdelta** in `linReach_adaptive` to see where the 8 vs 13 difference originates
4. ⏳ **Compare Rdelta** between Python and MATLAB
5. ⏳ **Trace upstream** to find where Rdelta diverges (likely in previous reduction steps or cartProd)

## Impact

This input mismatch explains:
- Different reduction results (2 vs 4 generators)
- Different `Z` dimensions after `cartProd(Rred, U)`
- Different `quadMap` inputs
- 20% difference in `errorSec`
- Early abortion in Python simulation

## Action Items

1. Fix MATLAB generator storage (ensure `generators(R)` is stored as a matrix)
2. Add Rdelta tracking in `linReach_adaptive` before the reduction
3. Compare Rdelta between Python and MATLAB
4. Trace further upstream to find where the generator count diverges
