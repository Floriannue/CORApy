"""
Investigate _aux_optimaldeltat sensitivity to rerr1, rR, varphimin for Step 2 Run 1.

Goal: Find how much we need to change rerr1 (or other inputs) so that bestIdx flips
from 8 to 0 (Python matches MATLAB and takes timeStepequalHorizon path).

Usage:
  python investigate_optimaldeltat_sensitivity.py

Requires: upstream_python_log.pkl with optimaldeltatLog (run reach with trackOptimaldeltat=True).
If missing, uses synthetic Step-2-like values for sensitivity formulas.
"""
import numpy as np
import pickle
import os

# Replicate objective from _aux_optimaldeltat (same formulas as linReach_adaptive.py)
def compute_objective_and_bestidx(rR, rerr1, varphimin, zetaP, deltat, mu=0.5, alg="lin", redFactor=0.1):
    dHused = 0.5 if alg == "lin" else 0.3
    zetaZ = float(redFactor) * dHused
    kprimemax = int(np.ceil(-np.log(100) / np.log(mu)))
    kprime = np.arange(0, kprimemax + 1)
    k = mu ** (-kprime)
    deltats = deltat * (mu ** kprime)
    floork = np.floor(k)

    varphimax = mu
    varphi_h = varphimax - varphimin
    varphi = (varphimin + (deltats[0] - deltats) / deltats[0] * varphi_h) / mu
    varphiprod = np.cumprod(varphi)

    sumallbutlast = np.zeros(len(floork))
    for i in range(len(floork)):
        n = int(floork[i])
        if n >= 1:
            firstfactor = (1 + 2 * zetaZ) ** (k[i] + 1 - np.arange(1, n + 1))
            secondfactor = zetaP ** (1 - np.arange(1, n + 1) / k[i])
            sumallbutlast[i] = np.sum(firstfactor * secondfactor)

    objfuncset = (
        rR * (1 + 2 * zetaZ) ** k * zetaP
        + rerr1 / k * varphiprod * (sumallbutlast + (1 + zetaZ) ** (k - kprime) * (k - floork))
    )
    bestIdxnew = int(np.argmin(objfuncset))
    return bestIdxnew, objfuncset, deltats


def main():
    print("=" * 80)
    print("INVESTIGATION: optimaldeltat inputs and bestIdx sensitivity (Step 2 Run 1)")
    print("=" * 80)

    # Try to load Python Step 2 optimaldeltat log
    py_entry = None
    if os.path.isfile("upstream_python_log.pkl"):
        try:
            with open("upstream_python_log.pkl", "rb") as f:
                data = pickle.load(f)
            log = data.get("optimaldeltatLog", [])
            for entry in log:
                if isinstance(entry, dict) and entry.get("step") == 2:
                    py_entry = entry
                    break
        except Exception as e:
            print(f"[WARN] Could not load upstream_python_log.pkl: {e}")

    if py_entry:
        rR = float(py_entry.get("rR", 0))
        rerr1 = float(py_entry.get("rerr1", 0))
        varphimin = float(py_entry.get("varphimin", 0))
        zetaP = float(py_entry.get("zetaP", 0))
        deltat = float(py_entry.get("deltat", 0))
        best_actual = py_entry.get("bestIdxnew", -1)
        print(f"\nLoaded Python Step 2 Run 1 from log:")
        print(f"  rR={rR:.6e}, rerr1={rerr1:.6e}, varphimin={varphimin:.6e}, zetaP={zetaP:.6e}, deltat={deltat:.6e}")
        print(f"  bestIdxnew (from log) = {best_actual}")
    else:
        # Synthetic Step-2-like values (from FIX_TIMESTEP_EQUAL_HORIZON.md / typical jetEngine)
        deltat = 0.01648
        rR = 1.0
        rerr1 = 0.5
        varphimin = 0.3
        zetaP = 0.1
        best_actual = None
        print("\nNo upstream_python_log.pkl (or no Step 2 entry). Using synthetic values:")
        print(f"  rR={rR}, rerr1={rerr1}, varphimin={varphimin}, zetaP={zetaP}, deltat={deltat}")

    # Default in reach is decrFactor=0.9 (not 0.5!)
    mu = 0.9
    redFactor = 0.1
    bestIdx, obj, deltats = compute_objective_and_bestidx(
        rR, rerr1, varphimin, zetaP, deltat, mu=mu, redFactor=redFactor
    )
    print(f"\nRecomputed bestIdx = {bestIdx} (index 0 = full horizon) [decrFactor={mu}, redFactor={redFactor}]")

    # If log has objfuncset, compare
    if py_entry and "objfuncset" in py_entry:
        log_obj = np.array(py_entry["objfuncset"])
        if log_obj.shape == obj.shape:
            diff = np.abs(obj - log_obj)
            print(f"  Objective vs log: max_abs_diff={np.max(diff):.6e}, match={np.allclose(obj, log_obj)}")
        log_best = py_entry.get("bestIdxnew", -1)
        if log_best != bestIdx:
            print(f"  => Log had bestIdxnew={log_best}; trying decrFactor in [0.5, 0.9].")
            for mu_try in [0.5, 0.6, 0.7, 0.8, 0.9]:
                b, o, _ = compute_objective_and_bestidx(
                    rR, rerr1, varphimin, zetaP, deltat, mu=mu_try, redFactor=redFactor
                )
                if b == log_best:
                    print(f"     match: decrFactor={mu_try} -> bestIdx={b}")
                    break

    # Sensitivity: scale rerr1 and see when bestIdx flips to 0 vs 8
    print("\n" + "-" * 80)
    print("SENSITIVITY: scale rerr1 (Python rerr1 * factor) -> bestIdx")
    print("-" * 80)
    factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    critical_scale_for_horizon = None
    for fac in factors:
        rerr1_scaled = rerr1 * fac
        bidx, _, _ = compute_objective_and_bestidx(rR, rerr1_scaled, varphimin, zetaP, deltat, mu=mu)
        marker = " <-- full horizon" if bidx == 0 else (" <-- reduced (like log)" if bidx == 8 else "")
        if bidx == 0 and critical_scale_for_horizon is None:
            critical_scale_for_horizon = fac
        print(f"  rerr1 * {fac:.2f} -> bestIdx = {bidx}{marker}")

    if critical_scale_for_horizon is not None:
        print(f"\n=> For bestIdx=0, rerr1 scale <= {critical_scale_for_horizon:.2f} gives full horizon in this run.")
    print("=> If MATLAB has smaller rerr1 (or different varphimin/rR), it can get bestIdx=1 (full horizon) while Python gets 8.")

    # Optional: sensitivity to varphimin
    print("\n" + "-" * 80)
    print("SENSITIVITY: varphimin (higher varphimin -> different varphi curve)")
    print("-" * 80)
    for vmin in [0.2, 0.3, 0.4, 0.5, 0.6]:
        bidx, _, _ = compute_objective_and_bestidx(rR, rerr1, vmin, zetaP, deltat, mu=mu)
        marker = " <-- full horizon" if bidx == 0 else ""
        print(f"  varphimin={vmin} -> bestIdx = {bidx}{marker}")

    print("\n" + "=" * 80)
    print("UPSTREAM CHAIN (to align rerr1 / varphimin)")
    print("=" * 80)
    print("  rerr1 = norm(sum(abs(Rerror_h.generators()), axis=1), 2)")
    print("  Rerror_h = Rerror (current Run 1); Rerror = errorSolution_adaptive(options, VerrorDyn, VerrorStat)")
    print("  => Compare: VerrorDyn (priv_abstractionError_adaptive), errorSolution_adaptive, reduce('adaptive')")
    print("  varphimin for step 2 = options['varphi'][0] = step 1's varphi")
    print("  => Compare: step 1 varphi computation and storage (options.varphi(1) in MATLAB)")
    print("  rR = norm(sum(abs(Rstart.generators()), axis=1), 2)")
    print("  => Compare: Rstart from previous step (initReach / reach flow)")
    print()


if __name__ == "__main__":
    main()
