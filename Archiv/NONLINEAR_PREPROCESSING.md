# Nonlinear Reach: Preprocessing (U and u/uTrans)

This document describes the preprocessing applied for **nonlinear (and all contDynamics) reach** so that the **input set U** and **input trajectory u** are in the form expected by downstream code (`linReach_adaptive`, `priv_abstractionError_adaptive`, `priv_precompStatError_adaptive`, `linearize`, etc.). Without this preprocessing, non-standard inputs (e.g. `U` as `Interval`, or user-provided `params.u`) can lead to problems later or to intermediate results not being passed correctly.

## MATLAB reference

- **postProcessing.m** (func = `'reach'`): `aux_convert_U`, then `aux_set_U_uTrans_uTransVec`.
- **config_nonlinearSys_reach.m**: `params.U` and `params.u` are **default** (optional); after postProcessing, `params.u` is removed and `params.uTrans` or `params.uTransVec` are set.

## Python implementation (contDynamics/reach.py)

After validation (either `sys.validateOptions` or the fallback block), the following runs **for all contDynamics** before any reach logic:

1. **Default U**  
   If `params` has no `'U'`, set  
   `params['U'] = Zonotope(zeros(nr_of_inputs, 1), empty generators)`.

2. **aux_convert_U**  
   If `params['U']` is an **Interval**, replace it with  
   `params['U'] = Zonotope(params['U'])`  
   so downstream code always receives a Zonotope (e.g. for `.center()`, `.interval()`, `cartProd_`).

3. **aux_set_U_uTrans_uTransVec**  
   If `params` contains `'u'` (and sys is not `nonlinearARX`):
   - `centerU = center(params['U'])`.
   - If `centerU` is non-zero:  
     `params['u'] = params['u'] + centerU`,  
     `params['U'] = params['U'] + (-centerU)`  
     (so **U is centered at the origin** and the shift is in the trajectory).
   - If `params['u']` has more than one column:  
     `params['uTransVec'] = params['u']` (time-varying input).  
     Else:  
     `params['uTrans'] = params['u']` (constant input).
   - Remove `params['u']`.

4. **Default uTrans**  
   If neither `params['uTrans']` nor `params['uTransVec']` is set and `params['U']` exists:  
   `params['uTrans'] = params['U'].center()`.

## Passing to subfunctions

- **reach_adaptive** receives the same `params` and `options` dicts (by reference). Preprocessing is done once in `reach()` before calling `reach_adaptive()`.
- **linReach_adaptive(nlnsys, Rstart, params, options)** uses `params['U']`, `params['tFinal']`, `params['tStart']`, etc., and passes `params['U']` into `priv_precompStatError_adaptive`, `priv_abstractionError_adaptive`, and `linearize`. So `params['U']` must be a Zonotope (not an Interval) and `params['uTrans']` (or `params['uTransVec']`) must be set as above.
- **Intermediate results** (e.g. `options['R']`, `options['varphi']`, `options['error_adm_horizon']`, `options['finitehorizon']`) are updated in place in `options` inside `linReach_adaptive` and subfunctions; because dicts are passed by reference, the next step and the main loop see the updated values.

## Summary

- **U**: Always converted from Interval to Zonotope; default U if missing.
- **u / uTrans / uTransVec**: If user provides `params.u`, it is split into origin-centered U and uTrans or uTransVec; then `params.u` is removed. Default uTrans = center(U) if no trajectory is given.
- This matches MATLAB postProcessing for `reach` and avoids issues with non-standard U or u when passing data into subfunctions.
