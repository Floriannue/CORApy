"""
Compare intermediate values for Step 2 Run 1 between Python and MATLAB.

Loads:
  - upstream_python_log.pkl (upstreamLog, optimaldeltatLog)
  - upstream_matlab_log.mat (upstreamLog, optimaldeltatLog)

Compares:
  1. _aux_optimaldeltat / aux_optimaldeltat: rR, rerr1, varphimin, zetaP, deltat, bestIdxnew, deltatest
  2. Rerror_before_optimaldeltat: rerr1, radius_max, num_generators
  3. VerrorDyn_before_errorsolution: radius_max (if present)
  4. Step 1 varphi (used as varphimin for step 2)

Prints differences and suggests where Python should mirror MATLAB.
"""
import os
import pickle
import numpy as np

try:
    import scipy.io
except ImportError:
    scipy.io = None

ROOT = os.path.dirname(os.path.abspath(__file__))


def _get_py_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_mat_val(entry, field, default=None):
    if hasattr(entry, field):
        v = getattr(entry, field)
        if hasattr(v, "flatten"):
            v = np.asarray(v).flatten()
            if v.size == 1:
                v = float(v.flat[0])
        return v
    return default


def _rel_diff(a, b):
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if abs(a) < 1e-20 and abs(b) < 1e-20:
        return 0.0
    if abs(a) < 1e-20:
        return float("inf")
    return abs(a - b) / abs(a)


def main():
    py_log_path = os.path.join(ROOT, "upstream_python_log.pkl")
    ml_log_path = os.path.join(ROOT, "upstream_matlab_log.mat")

    if not os.path.isfile(py_log_path):
        print("ERROR: upstream_python_log.pkl not found. Run: python track_upstream_python.py")
        return 1
    if not os.path.isfile(ml_log_path):
        print("ERROR: upstream_matlab_log.mat not found. Run MATLAB: run('track_upstream_matlab')")
        return 1

    with open(py_log_path, "rb") as f:
        py_data = pickle.load(f)
    py_upstream = py_data.get("upstreamLog", [])
    py_opt = py_data.get("optimaldeltatLog", [])

    if not scipy.io:
        print("ERROR: scipy not installed; cannot load .mat")
        return 1
    ml_data = scipy.io.loadmat(ml_log_path, struct_as_record=False, squeeze_me=True)
    ml_upstream = ml_data.get("upstreamLog", None)
    if ml_upstream is None:
        print("ERROR: upstream_matlab_log.mat has no 'upstreamLog'")
        return 1
    if hasattr(ml_upstream, "shape") and ml_upstream.size == 1:
        ml_upstream = [ml_upstream.flat[0]]
    elif hasattr(ml_upstream, "shape"):
        ml_upstream = list(ml_upstream.flatten())
    else:
        ml_upstream = [ml_upstream] if not isinstance(ml_upstream, list) else ml_upstream

    ml_opt = ml_data.get("optimaldeltatLog", None)
    if ml_opt is not None and hasattr(ml_opt, "shape"):
        ml_opt = list(ml_opt.flatten()) if ml_opt.size > 1 else [ml_opt.flat[0]] if ml_opt.size == 1 else []
    elif ml_opt is None:
        ml_opt = []

    print("=" * 80)
    print("COMPARISON: Step 2 Run 1 intermediate values (Python vs MATLAB)")
    print("=" * 80)

    # --- optimaldeltat Step 2 ---
    py_opt2 = next((e for e in py_opt if isinstance(e, dict) and e.get("step") == 2), None)
    ml_opt2 = None
    for e in ml_opt:
        if hasattr(e, "step") and e.step == 2:
            ml_opt2 = e
            break
    if py_opt2 is None:
        print("\n[WARN] No Python optimaldeltat entry for step 2")
    if ml_opt2 is None and ml_opt:
        print("\n[WARN] No MATLAB optimaldeltat entry for step 2")

    if py_opt2 and ml_opt2:
        print("\n--- _aux_optimaldeltat / aux_optimaldeltat (Step 2 Run 1) ---")
        params = ["deltat", "varphimin", "zetaP", "rR", "rerr1", "deltatest"]
        for p in params:
            py_v = _get_py_val(py_opt2, p)
            ml_v = _get_mat_val(ml_opt2, p)
            if py_v is None and ml_v is None:
                continue
            rd = _rel_diff(py_v, ml_v)
            status = "OK" if rd is not None and rd < 1e-6 else "DIFF"
            if rd is not None and rd >= 1e-6:
                status += f" (rel_diff={rd:.4e})"
            print(f"  {p}:  Python={py_v}  MATLAB={ml_v}  [{status}]")
        py_best = _get_py_val(py_opt2, "bestIdxnew")
        ml_best = _get_mat_val(ml_opt2, "bestIdxnew")
        print(f"  bestIdxnew:  Python(0-based)={py_best}  MATLAB(1-based)={ml_best}")
        if py_best is not None and ml_best is not None:
            if py_best == 0 and ml_best == 1:
                print("  => Same choice: full horizon (index 0/1) -> timeStepequalHorizon path")
            elif py_best == ml_best - 1:
                print("  => Same choice: same time step index (reduced; full horizon would be 0/1)")
            else:
                print("  => DIVERGENCE: different time step chosen; align rR/rerr1/varphimin to mirror MATLAB")

    # --- upstream Step 2 Run 1 (Rerror_before_optimaldeltat, VerrorDyn) ---
    py_entry = next(
        (e for e in py_upstream if _get_py_val(e, "step") == 2 and _get_py_val(e, "run") == 1),
        None,
    )
    ml_entry = None
    for e in ml_upstream:
        if _get_mat_val(e, "step") == 2 and _get_mat_val(e, "run") == 1:
            ml_entry = e
            break
    if py_entry and ml_entry:
        print("\n--- Rerror_before_optimaldeltat (Step 2 Run 1) ---")
        for key in ["rerr1", "radius_max"]:
            py_r = py_entry.get("Rerror_before_optimaldeltat", {}) if isinstance(py_entry, dict) else {}
            if isinstance(py_r, dict):
                py_v = py_r.get(key)
            else:
                py_v = _get_mat_val(py_r, key) if py_r else None
            ml_r = _get_mat_val(ml_entry, "Rerror_before_optimaldeltat")
            if ml_r is not None and hasattr(ml_r, key):
                ml_v = getattr(ml_r, key, None)
            else:
                ml_v = None
            if py_v is None and ml_v is None:
                continue
            rd = _rel_diff(py_v, ml_v)
            status = "OK" if rd is not None and rd < 1e-5 else "DIFF"
            if rd is not None and rd >= 1e-5:
                status += f" (rel_diff={rd:.4e})"
            print(f"  {key}:  Python={py_v}  MATLAB={ml_v}  [{status}]")

        print("\n--- VerrorDyn_before_errorsolution (Step 2 Run 1) ---")
        py_vd = py_entry.get("VerrorDyn_before_errorsolution", {}) if isinstance(py_entry, dict) else {}
        py_vd_max = py_vd.get("radius_max") if isinstance(py_vd, dict) else None
        ml_vd = _get_mat_val(ml_entry, "VerrorDyn_before_errorsolution")
        ml_vd_max = getattr(ml_vd, "radius_max", None) if ml_vd is not None else None
        if py_vd_max is not None or ml_vd_max is not None:
            rd = _rel_diff(py_vd_max, ml_vd_max)
            status = "OK" if rd is not None and rd < 1e-5 else "DIFF"
            if rd is not None and rd >= 1e-5:
                status += f" (rel_diff={rd:.4e})"
            print(f"  radius_max:  Python={py_vd_max}  MATLAB={ml_vd_max}  [{status}]")

        # Taylor order (tt_err) for step 2 - from errorSolution_adaptive
        py_tt = py_entry.get("tt_err", []) if isinstance(py_entry, dict) else []
        ml_tt = _get_mat_val(ml_entry, "tt_err")
        if isinstance(py_tt, list) and len(py_tt) >= 2:
            py_tt2 = py_tt[1]
        else:
            py_tt2 = None
        if ml_tt is not None:
            ml_tt = np.asarray(ml_tt).flatten()
            ml_tt2 = int(ml_tt[1]) if len(ml_tt) >= 2 else None
        else:
            ml_tt2 = None
        if py_tt2 is not None or ml_tt2 is not None:
            print("\n--- Taylor order tt_err (step 2, from errorSolution_adaptive) ---")
            match = py_tt2 == ml_tt2 if (py_tt2 is not None and ml_tt2 is not None) else False
            print(f"  tt_err(2):  Python={py_tt2}  MATLAB={ml_tt2}  [{'OK' if match else 'DIFF'}]")

    # --- Step 1 varphi (used as varphimin for step 2) ---
    py_varphi_step1 = None
    for e in py_opt:
        if isinstance(e, dict) and e.get("step") == 1:
            py_varphi_step1 = e.get("varphimin")
            break
    ml_varphi_step1 = None
    for e in ml_opt:
        if hasattr(e, "step") and e.step == 1:
            ml_varphi_step1 = _get_mat_val(e, "varphimin")
            break
    if py_varphi_step1 is not None or ml_varphi_step1 is not None:
        print("\n--- Step 1 varphi (-> varphimin for step 2) ---")
        rd = _rel_diff(py_varphi_step1, ml_varphi_step1)
        status = "OK" if rd is not None and rd < 1e-5 else "DIFF"
        if rd is not None and rd >= 1e-5:
            status += f" (rel_diff={rd:.4e})"
        print(f"  varphimin:  Python={py_varphi_step1}  MATLAB={ml_varphi_step1}  [{status}]")

    print("\n" + "=" * 80)
    print("Next: If rerr1 or varphimin differ, fix Python to mirror MATLAB (VerrorDyn, Rerror, or varphi storage).")
    print("See UPSTREAM_INVESTIGATION_OPTIMALDELTAT.md and RERR1_FIX_SUMMARY.md.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
