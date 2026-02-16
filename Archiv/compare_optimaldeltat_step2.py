"""Compare _aux_optimaldeltat inputs and outputs for Step 2"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING _aux_optimaldeltat FOR STEP 2")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

if 'optimaldeltatLog' in python_log:
    od_log = python_log['optimaldeltatLog']
    step2_od = [entry for entry in od_log if entry.get('step') == 2]
    if step2_od:
        py_od = step2_od[0]
        print("Python Step 2 optimaldeltat:")
        print(f"  deltat (finitehorizon): {py_od.get('deltat')}")
        print(f"  deltatest (returned): {py_od.get('deltatest')}")
        print(f"  bestIdxnew: {py_od.get('bestIdxnew')}")
        print(f"  deltats: {py_od.get('deltats')}")
        if py_od.get('deltats'):
            deltats = np.asarray(py_od.get('deltats'))
            print(f"  deltats[0] (should equal finitehorizon): {deltats[0] if len(deltats) > 0 else 'N/A'}")
            print(f"  deltats[bestIdxnew]: {deltats[py_od.get('bestIdxnew')] if py_od.get('bestIdxnew') is not None and len(deltats) > py_od.get('bestIdxnew') else 'N/A'}")
        
        print(f"\n  Inputs:")
        print(f"    varphimin: {py_od.get('varphimin')}")
        print(f"    zetaP: {py_od.get('zetaP')}")
        print(f"    rR: {py_od.get('rR')}")
        print(f"    rerr1: {py_od.get('rerr1')}")

# MATLAB
try:
    matlab_data = scipy.io.loadmat('optimaldeltat_matlab_log.mat', struct_as_record=False, squeeze_me=True)
    if 'optimaldeltatLog' in matlab_data:
        od_log = matlab_data['optimaldeltatLog']
        if isinstance(od_log, np.ndarray):
            step2_od = [entry for entry in od_log if hasattr(entry, 'step') and entry.step == 2]
        else:
            step2_od = [od_log] if hasattr(od_log, 'step') and od_log.step == 2 else []
        
        if step2_od:
            mat_od = step2_od[0]
            print("\nMATLAB Step 2 optimaldeltat:")
            print(f"  deltat (finitehorizon): {getattr(mat_od, 'deltat', None)}")
            print(f"  deltatest (returned): {getattr(mat_od, 'deltatest', None)}")
            print(f"  bestIdxnew: {getattr(mat_od, 'bestIdxnew', None)}")
            
            if hasattr(mat_od, 'deltats'):
                deltats = np.asarray(mat_od.deltats).flatten()
                print(f"  deltats[0] (should equal finitehorizon): {deltats[0] if len(deltats) > 0 else 'N/A'}")
                bestIdx = getattr(mat_od, 'bestIdxnew', None)
                if bestIdx is not None and len(deltats) > bestIdx:
                    print(f"  deltats[bestIdxnew]: {deltats[bestIdx]}")
                    # MATLAB uses 1-based indexing, so bestIdxnew is 1-based
                    # Check if bestIdxnew == 1 (which means index 0 in 0-based)
                    if bestIdx == 1:
                        print(f"  [MATCH] bestIdxnew == 1 means it selected deltats[0] = finitehorizon")
                    else:
                        print(f"  [DIFFERENT] bestIdxnew == {bestIdx} means it selected a different value")
except FileNotFoundError:
    print("\n[INFO] optimaldeltat_matlab_log.mat not found")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("If _aux_optimaldeltat returns a value different from finitehorizon,")
print("then timeStep != finitehorizon and timeStepequalHorizon remains False.")
print("This is correct behavior - the optimization found a better time step.")
print("\nBut if MATLAB uses timeStepequalHorizon path, it means MATLAB's")
print("aux_optimaldeltat returned exactly finitehorizon (bestIdxnew == 1 in MATLAB, 0 in Python).")
print("\nThe question is: why does Python's _aux_optimaldeltat choose a different")
print("value than MATLAB's aux_optimaldeltat?")
