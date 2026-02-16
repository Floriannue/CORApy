# Run Python and MATLAB Tracking in Parallel and Compare

Same benchmark: **jetEngine**, `tFinal=8`, `R0=zonotope([1;1], 0.1*eye(2))`, `alg=lin-adaptive`, `trackUpstream=true`, `trackOptimaldeltat=true`.

## 1. Run Python (extract intermediate values)

From project root:

```bash
python track_upstream_python.py
```

- Writes **upstream_python_log.pkl** (upstreamLog, optimaldeltatLog, Rtp_tracking).
- Or use: `python run_both_tracking.py` (runs only Python unless you pass `--matlab`).

## 2. Run MATLAB (extract intermediate values)

**Option A – In MATLAB (recommended)**  
In MATLAB, from project root:

```matlab
run('track_upstream_matlab')
```

- Clears globals `upstreamLogGlobal` and `optimaldeltatLogGlobal`, runs reach, then saves **upstream_matlab_log.mat** (upstreamLog, optimaldeltatLog, Rtp_tracking).

**Option B – From shell (if `matlab` is on PATH)**  
From project root:

```bash
python run_both_tracking.py --matlab
```

- Runs Python first, then invokes MATLAB in batch to run `track_upstream_matlab`.

**Running in parallel:**  
Start Python and MATLAB in two terminals (same commands as above). After both finish, run the comparison (step 3).

## 3. Compare intermediate values (Step 2 Run 1)

From project root:

```bash
python compare_intermediate_step2.py
```

Or:

```bash
python run_both_tracking.py --compare
```

- Requires **upstream_python_log.pkl** and **upstream_matlab_log.mat**.
- Compares:
  - **optimaldeltat** (Step 2 Run 1): rR, rerr1, varphimin, zetaP, deltat, bestIdxnew, deltatest
  - **Rerror_before_optimaldeltat**: rerr1, radius_max
  - **VerrorDyn_before_errorsolution**: radius_max
  - **Step 1 varphi** (used as varphimin for step 2)
- Prints OK/DIFF and relative differences; suggests aligning Python with MATLAB where values differ (see UPSTREAM_INVESTIGATION_OPTIMALDELTAT.md).

## 4. Ensure Python mirrors MATLAB (exact translation per readme_florian2.md)

- **Same number of steps**: Both run 2 reach steps for jetEngine tFinal=8; same Run 1 / Run 2 logic per step.
- **Same logic**: bestIdx (time step index) is identical (Python 0-based 8 = MATLAB 1-based 9); timeStepequalHorizon path matches.
- **Numerical alignment**: rerr1/radius/rR/varphi use float64; errorSolution_adaptive RerrorInt_etanoF and gainnoF use float64; _aux_optimaldeltat k/deltats/varphiprod/objfuncset use float64.
- If **rerr1** or **VerrorDyn** differ: compare VerrorDyn and Rerror chain (priv_abstractionError_adaptive, errorSolution_adaptive, reduce). See RERR1_FIX_SUMMARY.md.
- If **varphimin** (step 1 varphi) differs: compare step 1 varphi computation and storage (options.varphi(1) in MATLAB vs options['varphi'][0] in Python).
- If **rR** differs: compare Rstart (previous step’s Rtp after reduction).

After fixing a divergence, re-run Python (and optionally MATLAB) and **compare_intermediate_step2.py** again until Step 2 Run 1 values match and Python takes the timeStepequalHorizon path when MATLAB does.
