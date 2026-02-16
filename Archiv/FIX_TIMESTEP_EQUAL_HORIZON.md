# Fix for timeStepequalHorizon Logic

## Problem

Python Step 2 Run 2 does NOT use `timeStepequalHorizon` path, while MATLAB does.
This causes Python to compute a different `Rhom_tp`, leading to different reduction results.

## Root Cause Analysis

### Step-by-Step Flow

1. **Step 2 starts** (line 98):
   - `finitehorizon = prev_finitehorizon * (1 + prev_varphi - zetaphi_val)`
   - `options['timeStep'] = finitehorizon`

2. **Run 1 executes** (lines 599-602):
   - `options['timeStep'], _ = _aux_optimaldeltat(..., finitehorizon, ...)`
   - `options['timeStep'] = min(options['timeStep'], params['tFinal'] - options['t'])`

3. **After Run 1** (line 675):
   - `options['run'] += 1`

4. **Check** (line 682):
   - `if options['timeStep'] == finitehorizon:`
   - This compares the MODIFIED `timeStep` (after `_aux_optimaldeltat`) with the ORIGINAL `finitehorizon`

### The Issue

`_aux_optimaldeltat` returns `bestIdxnew = 8`, selecting `deltats[8] = 0.00709` instead of `deltats[0] = finitehorizon = 0.01648`.

So `timeStep = 0.00709 != finitehorizon = 0.01648`, and the check correctly fails.

### Why MATLAB Succeeds

MATLAB also has `bestIdxnew = 9` (1-based, same as Python's 8), so it should also fail. But MATLAB Step 2 Run 2 uses the path.

**Possible explanations:**
1. MATLAB's check happens at a different time (before `_aux_optimaldeltat`?)
2. MATLAB's `aux_optimaldeltat` returns exactly `finitehorizon` in some cases
3. There's a different code path in MATLAB
4. The check logic is different

## Investigation Needed

1. Check if MATLAB's `aux_optimaldeltat` returns exactly `finitehorizon` for Step 2 Run 1
2. Verify the exact sequence of operations in MATLAB
3. Check if there's a different condition or logic path

## Fix Implemented (logic, not tolerance)

- **`_aux_optimaldeltat`** now returns `(deltatest, kprimeest, bestIdxnew)`.
- After Run 1 we store `_optimaldeltat_bestIdx` and (for i > 1) `_timeStep_uncapped` in options.
- **timeStepequalHorizon** is set when:
  - `bestIdxnew == 0` (optimizer chose full horizon), or
  - uncapped timeStep equals finitehorizon (for floating-point / cap edge cases).
- This uses the optimizer’s decision instead of floating-point equality; it does not relax tolerances.

## Remaining divergence

If Python’s `_aux_optimaldeltat` returns `bestIdxnew = 8` while MATLAB returns `bestIdxnew = 1` for Step 2 Run 1, Python will still not take the horizon path. Fixing that requires aligning **upstream inputs** (Rstart, Rerror_h, varphi → rR, rerr1, varphimin) so the objective has the same minimum.

**Investigation:** See **UPSTREAM_INVESTIGATION_OPTIMALDELTAT.md** for:
- Where rR, rerr1, varphimin come from (Rstart, Rerror, step 1 varphi)
- Upstream chain: VerrorDyn → errorSolution_adaptive → Rerror → rerr1
- Sensitivity script: `investigate_optimaldeltat_sensitivity.py`
- RERR1_FIX_SUMMARY.md for rerr1/VerrorDyn differences

## Current Status

- ✅ Check uses optimizer’s chosen index (bestIdx == 0) and uncapped timeStep
- ⏳ Align upstream inputs (VerrorDyn / Rerror / varphi / Rstart) so Python gets bestIdxnew = 0 when MATLAB gets 1; follow UPSTREAM_INVESTIGATION_OPTIMALDELTAT.md
