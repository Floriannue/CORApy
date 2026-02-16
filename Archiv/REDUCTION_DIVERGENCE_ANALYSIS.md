# Reduction Algorithm Divergence Analysis

## Root Cause Identified

**Python reduces to 2 generators, MATLAB reduces to 4 generators** - This is the root cause of the 20% difference in `errorSec`.

## Key Differences (Step 3)

| Metric | Python | MATLAB | Difference |
|--------|--------|--------|------------|
| `final_generators` | **2** | **4** | **2 generators** |
| `redIdx` | 2 | 5 | 3 indices |
| `dHerror` | 8.3e-06 | 0.00447 | 99.81% relative |
| `h_computed_max` | 8.3e-06 | 0.0251 | 99.97% relative |

## Analysis

### Python Behavior
- `redIdx = 2`: Only 2 generators can be reduced while staying under `dHmax`
- `dHerror = 8.3e-06`: Very small error, well below `dHmax`
- `final_generators = 2`: Results in a zonotope with 2 generators

### MATLAB Behavior  
- `redIdx = 5`: 5 generators can be reduced while staying under `dHmax`
- `dHerror = 0.00447`: Larger error, but still under `dHmax`
- `final_generators = 4`: Results in a zonotope with 4 generators

## Critical Question

**Which implementation is correct?**

The reduction algorithm should reduce generators until the Hausdorff distance error (`dHerror`) exceeds `dHmax`. Both implementations claim to do this, but they produce different results.

## Possible Causes

1. **Different `dHmax` calculation**: Need to verify both compute the same `dHmax`
2. **Different `h` computation**: The `h` array (Hausdorff distance estimates) might be computed differently
3. **Different `find` logic**: The `find(h <= dHmax, 1, 'last')` might behave differently
4. **Different indexing**: 0-based vs 1-based indexing issues

## Next Steps

1. **Compare `dHmax` values** between Python and MATLAB
2. **Compare `h_computed` arrays** element-by-element
3. **Trace the exact reduction logic** step-by-step for the same input
4. **Verify `find` behavior** - ensure both find the same `redIdx`

## Impact

This 2-generator difference causes:
- Different `Z` dimensions after reduction (2 vs 4 generators)
- Different `quadMap` input dimensions
- Different `errorSec` values (20% difference)
- Different `VerrorDyn` and `rerr1` values
- Early abortion in Python simulation
