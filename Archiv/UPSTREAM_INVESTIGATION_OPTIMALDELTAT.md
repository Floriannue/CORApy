# Upstream Investigation: optimaldeltat Inputs (Step 2 Run 1)

## Goal

Align **upstream inputs** so that Python’s `_aux_optimaldeltat` returns `bestIdxnew = 0` (full horizon) when MATLAB returns `bestIdxnew = 1`, enabling the `timeStepequalHorizon` path and matching Rlintp generator counts (4 vs 2).

## Objective Function (summary)

`bestIdxnew = argmin(objfuncset)` where:

- **objfuncset[i]** = `rR * (1+2*zetaZ)^k[i] * zetaP` + `rerr1/k[i] * varphiprod[i] * (sumallbutlast[i] + ...)`
- **k** = `decrFactor^(-kprime)`, **deltats** = `deltat * decrFactor^kprime`
- **zetaZ** = `redFactor * dHused` (dHused = 0.5 lin / 0.3 poly)
- **varphi / varphiprod** depend on **varphimin** and **deltats**

So the chosen index depends on **rR, rerr1, varphimin, zetaP, deltat** and options **decrFactor, redFactor, alg**.

## Inputs and Where They Come From

| Input       | Formula / source |
|------------|-------------------|
| **rR**     | `norm(sum(abs(Rstart.generators()), axis=1), 2)` → from **Rstart** (previous step) |
| **rerr1**  | `norm(sum(abs(Rerror.generators()), axis=1), 2)` → from **Rerror_h = Rerror** (current Run 1) |
| **varphimin** | Step 2: `options['varphi'][0]` = **step 1’s varphi** |
| **zetaP**  | From options (computed earlier) |
| **deltat** | `finitehorizon` for current step |

## Upstream Chain for rerr1

1. **Rerror** = `linsys.errorSolution_adaptive(options, VerrorDyn, VerrorStat)`
2. **VerrorDyn, VerrorStat, trueError** = `priv_abstractionError_adaptive(...)`
3. Inside abstraction error: **quadMap**, **reduce('adaptive')**, etc.

So to align **rerr1** between Python and MATLAB, compare (in order):

1. **VerrorDyn** (priv_abstractionError_adaptive)
2. **Rerror** before/after errorSolution_adaptive (and any internal reduce)
3. **quadMap** and **reduce('adaptive')** choices (RERR1_FIX_SUMMARY.md)

## Upstream Chain for varphimin (step 2)

- **varphimin** = step 1’s varphi = `options['varphi'][0]`
- Step 1’s varphi is set in **Run 2** of step 1 (or from aux_varphiest / max(abstrerr/abstrerr_h)).
- Compare: **options.varphi(1)** in MATLAB vs **options['varphi'][0]** in Python and the exact step where varphi is written.

## Upstream Chain for rR

- **Rstart** is the reach set passed into `linReach_adaptive` for the current step (e.g. from initReach / previous step).
- Compare **Rstart** (and thus **rR**) for step 2 between Python and MATLAB.

## Sensitivity (from investigate_optimaldeltat_sensitivity.py)

- With **decrFactor = 0.9** (default), the objective is very sensitive to **rR, rerr1, varphimin**.
- Reducing **rerr1** (e.g. scaling toward MATLAB’s value) can flip **bestIdx** from 8 to 0.
- Default **decrFactor** in reach is **0.9** (see reach.py / postProcessing.m), so **kprimemax** is large (~44) and **deltats** has many elements.

## Recommended Next Steps

1. **Run both with tracking**  
   Enable `trackOptimaldeltat` (and optionally `trackUpstream`) in both Python and MATLAB, run the same benchmark (e.g. jetEngine) for 2 steps.

2. **Compare Step 2 Run 1 inputs**  
   For the call to `_aux_optimaldeltat` / `aux_optimaldeltat`:
   - **rR, rerr1, varphimin, zetaP, deltat**
   - **decrFactor, redFactor, alg**

3. **Locate first divergence**  
   - If **rerr1** differs: compare **VerrorDyn** and **Rerror** (and **errorSolution_adaptive** / reduce) for step 2 Run 1.
   - If **varphimin** differs: compare step 1 varphi computation and storage.
   - If **rR** differs: compare **Rstart** (initReach / previous step).

4. **Fix at source**  
   Adjust the first place where Python and MATLAB differ (e.g. VerrorDyn, Rerror, varphi storage, or Rstart) so that Step 2 Run 1 inputs match and Python gets **bestIdxnew = 0** when MATLAB gets 1.

## Scripts

- **investigate_optimaldeltat_sensitivity.py** – Recomputes objective from log, tests sensitivity to rerr1/varphimin and documents upstream chain.
- **compare_optimaldeltat_inputs_step2.py** – Compares Python vs MATLAB optimaldeltat log for step 2 (needs MATLAB log file).
