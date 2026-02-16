# Comparison and Growth Analysis Summary

## 1. error_adm_horizon Update Logic (COMPLETED)

**Finding**: Each step starts with the previous step's Run 1 `error_adm_horizon`, and Run 1 sets `error_adm_horizon` to its final `trueError`.

- **Step 450**: Run 1 sets `error_adm_horizon` = 7.698679e+05
- **Step 451**: Starts with `error_adm_horizon` = 7.698679e+05, Run 1 sets it to 1.253572e+06
- **Growth**: 1.628x per step

See `ERROR_ADM_HORIZON_UPDATE_ANALYSIS.md` for details.

## 2. trueError Growth Mechanism (COMPLETED)

**Root Cause**: A self-reinforcing feedback loop:

1. Higher `error_adm_horizon` → Higher initial `error_adm` for next step
2. Higher `error_adm` → More iterations needed to converge
3. More iterations → `trueError` grows more within the step
4. Larger `trueError` → `error_adm_horizon` increases even more
5. **Repeat** → Exponential growth

### Detailed Analysis: Step 450 → Step 451

**Step 450, Run 1**:
- Initial `error_adm`: 6.081008e+05
- Iterations: 3
- Final `trueError`: 7.698679e+05
- Growth within step: 1.178x

**Step 451, Run 1**:
- Initial `error_adm`: 7.698679e+05 (1.266x higher than Step 450)
- Iterations: 5 (more iterations due to higher `error_adm`)
- Final `trueError`: 1.253572e+06
- Growth within step: 1.518x (more growth due to more iterations)

**Component Growth** (Step 450 → Step 451, Iteration 1):
- `trueError_max`: 1.263x
- `VerrorDyn_radius_max`: 1.263x (main driver)
- `errorSec_radius_max`: 1.148x
- `Z_radius_max`: 1.081x
- `Rmax_radius_max`: 1.081x

See `TRUEERROR_GROWTH_ANALYSIS.md` for complete details.

## 3. MATLAB Comparison (PENDING)

**Status**: MATLAB test failed due to missing `hessianTensorInt_jetEngine` function.

**Issue**: The `jetEngine` model in MATLAB doesn't have the required hessian functions for tensor order 3.

**Next Steps**:
1. Find a simpler test model that has all required functions
2. Or create a minimal test model with hessian and thirdOrderTensor
3. Or modify the test to use tensor order 2 instead

**Alternative**: Since we've identified the growth mechanism, we can:
1. Check if MATLAB has the same behavior by examining the code logic
2. Compare the `error_adm_horizon` update logic in MATLAB's `linReach_adaptive.m`
3. Verify if MATLAB also lacks safeguards against this growth

## Key Insights

1. **The growth is algorithmic, not a bug**: It's a consequence of how the adaptive algorithm works - higher error bounds allow more iterations, which allow larger errors.

2. **Run 1 vs Run 2**: 
   - Run 1 uses `error_adm_horizon` (which grows)
   - Run 2 uses `error_adm_Deltatopt` (which is typically smaller and more stable)
   - The issue is that `error_adm_horizon` is set from Run 1's `trueError`, not Run 2's

3. **Potential Solutions**:
   - Use `error_adm_Deltatopt` instead of Run 1's `trueError` for `error_adm_horizon`
   - Cap the growth of `error_adm_horizon` (e.g., max 1.1x per step)
   - Early divergence detection
   - Adaptive time step reduction when growth is too rapid

## Files Created

1. `ERROR_ADM_HORIZON_UPDATE_ANALYSIS.md` - Analysis of update logic
2. `TRUEERROR_GROWTH_ANALYSIS.md` - Detailed growth mechanism analysis
3. `analyze_trueError_growth.py` - Script to analyze component growth
4. `analyze_iteration_growth.py` - Script to analyze iteration-by-iteration growth
5. `analyze_run2_tracking.py` - Script to analyze Run 1 → Run 2 flow
