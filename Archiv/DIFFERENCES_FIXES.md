# Fixes for Python–MATLAB Intermediate Value Differences

## Comparison summary (Step 2 Run 1)

From `compare_intermediate_step2.py`, typical relative differences are:

- **deltat**: ~5e-5  
- **varphimin**: ~6e-5  
- **zetaP**: ~4e-6  
- **rR**: ~2e-6  
- **rerr1**: ~1.5e-3 (largest)  
- **deltatest**: ~5e-5  
- **Step 1 varphi**: ~1e-4  

**bestIdxnew** matches (Python 8 ↔ MATLAB 9, same reduced time step).

---

## Fixes applied

### 1. **errorSolution_adaptive.py**

- **RerrorInt_etanoF** and **_sum_abs_generators** use **float64** so the Taylor convergence (gainnoF, break condition) is numerically consistent with MATLAB and does not change Taylor order due to type/accuracy.
- **RerrorInt_etanoF** is explicitly cast to `np.float64` after each update.

### 2. **linReach_adaptive.py**

- **rR** and **rerr1** in `_aux_optimaldeltat` are computed with **float64** and returned as Python **float** to align with MATLAB `vecnorm(sum(abs(generators),2),2)` and avoid type/accuracy effects.
- **tt_err** (Taylor order per step) is stored in the upstream log so it can be compared (e.g. `compare_intermediate_step2.py`).

### 3. **MATLAB linReach_adaptive.m**

- **tt_err** is written into the upstream log (when `trackUpstream` is true) so Python and MATLAB can be compared on Taylor order.

### 4. **compare_intermediate_step2.py**

- **Taylor order (tt_err)** for step 2 is compared when present in both logs.
- **bestIdx** message corrected so that index 8/9 is described as “same time step index (reduced)”, not “full horizon”.

---

## Remaining small differences

- **rerr1** (~0.15%): Comes from **Rerror** generators, which depend on **VerrorDyn** and the Taylor expansion in **errorSolution_adaptive**. Float64 and consistent RerrorInt_etanoF reduce the chance of a different Taylor order; any remaining gap is from small differences in **VerrorDyn** or in the zonotope/interval operations (order of operations, expm, etc.).
- **varphimin / Step 1 varphi** (~1e-4): **varphi** is derived from **abstrerr** (from Rerror). So improving **rerr1** alignment also tends to reduce **varphi** differences.
- **deltat** (~5e-5): **finitehorizon** = prev_finitehorizon * (1 + prev_varphi - zetaphi). Small differences in **prev_varphi** or **prev_finitehorizon** explain the small **deltat** gap.

---

## What to do next

1. Re-run **track_upstream_python.py** and **track_upstream_matlab**, then **compare_intermediate_step2.py** and check whether **rerr1** and **tt_err** are closer.
2. If **tt_err** still differs for step 2, the Taylor break condition (gainnoF vs zetaTabs) differs; then compare **RerrorInt_etanoF** (and possibly **VerrorDyn**) at the break step.
3. If **rerr1** is still ~0.15% with the same **tt_err**, the remaining source is **VerrorDyn** or the zonotope/interval arithmetic in **errorSolution_adaptive** (e.g. IntervalMatrix * Zonotope, expm). Aligning those would require a more detailed step-by-step comparison of **VerrorDyn** and the Taylor terms.
