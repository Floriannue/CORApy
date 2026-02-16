# trueError Growth Analysis

## Summary

The rapid growth of Run 1's `trueError` (1.626x per step) is caused by a **feedback loop** between `error_adm_horizon` and the inner loop convergence process.

## The Feedback Loop

1. **Step N completes** → `error_adm_horizon` is set to Run 1's final `trueError`
2. **Step N+1 starts** → Initial `error_adm` is set from `error_adm_horizon` (or `error_adm_Deltatopt` for Run 1)
3. **Higher `error_adm`** → More iterations needed to converge (because `perfIndCurr = max(trueError ./ error_adm)` must be ≤ 1)
4. **More iterations** → `trueError` grows more within the step (each iteration increases `error_adm`, which allows larger `trueError`)
5. **Larger final `trueError`** → `error_adm_horizon` increases even more
6. **Repeat** → The cycle continues, causing exponential growth

## Detailed Analysis: Step 450 → Step 451

### Step 450, Run 1
- **Initial `error_adm`**: 6.081008e+05 (from Step 449's `error_adm_horizon`)
- **Iteration 1**: `trueError` = 6.537455e+05
- **Iteration 2**: `trueError` = 7.216425e+05
- **Iteration 3 (final)**: `trueError` = 7.698679e+05
- **Growth within step**: 1.178x
- **Final `error_adm_horizon`**: 7.698679e+05

### Step 451, Run 1
- **Initial `error_adm`**: 7.698679e+05 (from Step 450's `error_adm_horizon`) - **1.266x higher**
- **Iteration 1**: `trueError` = 8.255770e+05
- **Iteration 2**: `trueError` = 9.352174e+05
- **Iteration 3**: `trueError` = 1.038434e+06
- **Iteration 4**: `trueError` = 1.142295e+06
- **Iteration 5 (final)**: `trueError` = 1.253572e+06
- **Growth within step**: 1.518x (more iterations = more growth)
- **Final `error_adm_horizon`**: 1.253572e+06

### Step 450 → Step 451 Growth
- **Final `trueError` growth**: 1.628x
- **Initial `error_adm` growth**: 1.266x
- **Within-step growth (Step 451)**: 1.518x

## Why More Iterations = More Growth

Within each iteration:
1. `trueError` is computed from `VerrorDyn` (which depends on `Rmax`, `Z`, `errorSec`)
2. If `perfIndCurr = max(trueError ./ error_adm) > 1`, the loop continues
3. `error_adm` is updated: `error_adm = error_adm * (1 + perfIndCurr) / 2` (approximately)
4. With a larger `error_adm`, the next iteration can tolerate a larger `trueError`
5. This allows `trueError` to grow further before convergence

## Component Growth (Step 450 → Step 451, Iteration 1)

- **`trueError_max`**: 1.263x
- **`VerrorDyn_radius_max`**: 1.263x (matches `trueError` - this is the main driver)
- **`errorSec_radius_max`**: 1.148x (quadratic error from `Z.quadMap(H)`)
- **`Z_radius_max`**: 1.081x (depends on `Rmax`)
- **`Rmax_radius_max`**: 1.081x (reachable set size)
- **`error_adm_max`**: 1.266x (initial error bound)

## Key Insight

The growth is **self-reinforcing**:
- Higher `error_adm_horizon` → Higher initial `error_adm` → More iterations → Larger `trueError` → Higher `error_adm_horizon`

This explains why the growth accelerates over time (Step 451 needs 5 iterations vs Step 450's 3 iterations).

## Comparison with MATLAB

**TODO**: Run MATLAB test to verify if this behavior matches MATLAB's implementation.

## Potential Solutions

1. **Cap `error_adm_horizon` growth**: Limit the maximum growth factor per step
2. **Use `error_adm_Deltatopt` instead**: Use Run 2's `trueError` (which is typically smaller) for `error_adm_horizon`
3. **Early divergence detection**: Detect when `error_adm_horizon` is growing too rapidly and stop
4. **Adaptive time step reduction**: Reduce time step when `error_adm_horizon` grows too fast
