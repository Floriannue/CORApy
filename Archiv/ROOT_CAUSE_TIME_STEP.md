# ROOT CAUSE IDENTIFIED: Time Step Divergence

## Summary

The generator count divergence (Python: 2 vs MATLAB: 4) is **NOT** caused by the reduction algorithm itself, but by **different time step values** used in `initReach_adaptive`.

## Key Finding

**Python Run 1 vs Run 2 Comparison (Step 4):**
- **Run 1**: `timeStep = 0.0165` → Reduces 5→4 generators (matches MATLAB)
- **Run 2**: `timeStep = 0.0098` → Reduces 5→2 generators (differs from MATLAB)

## Impact Chain

Different `timeStep` → Different `eAt` → Different `Rtrans`/`inputCorr` → Different `Rhom_tp` generator values → Different reduction results

### Evidence

1. **Time Step Difference:**
   - Run 1: `0.016529655902895162`
   - Run 2: `0.009760596514100566`
   - Difference: `0.006769` (41% relative difference)

2. **eAt Difference:**
   - Run 1: `[[ 0.936796   -0.01586976], [ 0.04760928  0.98320953]]`
   - Run 2: `[[ 0.9616908  -0.00952578], [ 0.02857735  0.99014674]]`
   - Max difference: `0.0249` (2.5%)

3. **Rhom_tp Generator Values:**
   - Run 1: `[[ 8.817e-02 -1.650e-03 ...], [ 4.481e-03  1.022e-01 ...]]`
   - Run 2: `[[ 9.052e-02 -9.905e-04 ...], [ 2.690e-03  1.030e-01 ...]]`
   - Max difference: `0.00234` (2.6% relative)

4. **Reduction Results:**
   - Run 1: 5 generators → 4 generators (conservative reduction)
   - Run 2: 5 generators → 2 generators (aggressive reduction)

## Why This Happens

The adaptive time step selection in `initReach_adaptive` chooses different time steps based on:
- Abstraction error bounds
- Taylor term convergence
- Previous iteration results

**Python Run 2** is selecting a smaller time step (`0.0098`) than **MATLAB Run 2** (or Python Run 1), which leads to:
- Different exponential matrix `eAt`
- Different input correction terms
- Different `Rhom_tp` generator values
- Different reduction behavior (smaller generators get reduced more aggressively)

## Time Step Selection

The time step is selected by `_aux_optimaldeltat()` in `linReach_adaptive.py` (line 578). This function optimizes the time step based on:

- `Rstart` - Starting zonotope (center and generators)
- `Rerror_h` - Error zonotope from abstraction error
- `finitehorizon` - Finite horizon value
- `varphi` - Parameter computed from error bounds
- `zetaP` - Parameter from options
- `options` - Including `decrFactor`, `redFactor`, etc.

**Key Finding**: Since `Rstart` differs between Run 1 and Run 2 (different centers), this leads to different time step selections:
- Run 1: `Rstart_center = [0.0267, -0.0144]` → `timeStep = 0.0165`
- Run 2: `Rstart_center = [0.0158, -0.0085]` → `timeStep = 0.0098`

## Next Steps

1. **Compare `_aux_optimaldeltat` inputs** between Python Run 2 and MATLAB Run 2:
   - `Rstart` (center and generators)
   - `Rerror_h` (error zonotope)
   - `finitehorizon`
   - `varphi`
   - `zetaP`
   
2. **Check if `Rstart` differs** between Python Run 2 and MATLAB Run 2 - this would explain the time step difference

3. **Verify `_aux_optimaldeltat` implementation** - ensure Python and MATLAB compute the same optimal time step for the same inputs

4. **Compare abstraction error computation** - if Python computes different `Rerror_h`, it will select different time steps

## Files to Check

- `cora_python/contDynamics/linearSys/initReach_adaptive.py` - Time step selection
- `cora_matlab/contDynamics/@linearSys/initReach_adaptive.m` - Time step selection
- `cora_python/contDynamics/linearSys/errorSolution_adaptive.py` - Abstraction error
- `cora_matlab/contDynamics/@linearSys/errorSolution_adaptive.m` - Abstraction error

## Status

🔴 **ROOT CAUSE**: Time step divergence, not reduction algorithm bug
