"""Check when _aux_optimaldeltat returns exactly finitehorizon (bestIdxnew = 0)"""
import pickle
import numpy as np

print("=" * 80)
print("CHECKING WHEN _aux_optimaldeltat RETURNS EXACTLY finitehorizon")
print("=" * 80)

with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

optimaldeltat_log = python_log.get('optimaldeltatLog', [])

print(f"Found {len(optimaldeltat_log)} optimaldeltat entries")

# Check Step 2 Run 1
step2_run1 = None
for entry in optimaldeltat_log:
    if isinstance(entry, dict) and entry.get('step') == 2:
        step2_run1 = entry
        break

if step2_run1:
    print(f"\nStep 2 Run 1 optimaldeltat:")
    deltat = step2_run1.get('deltat')
    deltatest = step2_run1.get('deltatest')
    bestIdxnew = step2_run1.get('bestIdxnew')
    deltats = np.asarray(step2_run1.get('deltats', []))
    
    print(f"  Input deltat (finitehorizon): {deltat}")
    print(f"  Output deltatest: {deltatest}")
    print(f"  bestIdxnew: {bestIdxnew}")
    
    if len(deltats) > 0:
        print(f"  deltats[0] (should equal deltat if bestIdxnew=0): {deltats[0]}")
        print(f"  deltats: {deltats[:5]}")  # First 5
        
        if bestIdxnew == 0:
            print(f"  [OK] bestIdxnew = 0, so deltatest should equal deltat")
            print(f"  deltatest == deltat? {deltatest == deltat}")
        else:
            print(f"  [ISSUE] bestIdxnew = {bestIdxnew}, so deltatest != deltat")
            print(f"  This means _aux_optimaldeltat chose a different time step")
            print(f"  deltats[{bestIdxnew}] = {deltats[bestIdxnew]}")
    
    # Check objfuncset to see why bestIdxnew != 0
    objfuncset = np.asarray(step2_run1.get('objfuncset', []))
    if len(objfuncset) > 0:
        print(f"\n  Objective function values:")
        print(f"  objfuncset: {objfuncset[:5]}")  # First 5
        print(f"  Minimum at index: {np.argmin(objfuncset)}")
        print(f"  Value at index 0: {objfuncset[0] if len(objfuncset) > 0 else 'N/A'}")
        print(f"  Minimum value: {np.min(objfuncset) if len(objfuncset) > 0 else 'N/A'}")

# Check all Step 2 entries
step2_entries = [e for e in optimaldeltat_log if isinstance(e, dict) and e.get('step') == 2]
print(f"\nAll Step 2 optimaldeltat entries: {len(step2_entries)}")
for entry in step2_entries:
    deltat = entry.get('deltat')
    deltatest = entry.get('deltatest')
    bestIdxnew = entry.get('bestIdxnew')
    if deltat and deltatest:
        print(f"  deltat={deltat:.12e}, deltatest={deltatest:.12e}, bestIdxnew={bestIdxnew}, equal={deltat==deltatest}")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("If bestIdxnew = 0, then deltatest = deltats[0] = deltat * mu^0 = deltat")
print("So timeStep == finitehorizon and the check succeeds.")
print("If bestIdxnew != 0, then deltatest != deltat and the check fails.")
print("\nNeed to check why Python's bestIdxnew != 0 but MATLAB's might be 0.")
