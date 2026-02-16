"""Compare _aux_optimaldeltat inputs between Python and MATLAB for Step 2 Run 1"""
import pickle
import scipy.io
import numpy as np

print("=" * 80)
print("COMPARING _aux_optimaldeltat INPUTS FOR STEP 2 RUN 1")
print("=" * 80)

# Python
with open('upstream_python_log.pkl', 'rb') as f:
    python_log = pickle.load(f)

optimaldeltat_log = python_log.get('optimaldeltatLog', [])

py_entry = None
for entry in optimaldeltat_log:
    if isinstance(entry, dict) and entry.get('step') == 2:
        py_entry = entry
        break

if py_entry is None:
    print("[ERROR] Could not find Python Step 2 optimaldeltat entry")
    exit(1)

print("[OK] Found Python Step 2 optimaldeltat entry")

print(f"\nPython Step 2 Run 1 _aux_optimaldeltat:")
print(f"  Input deltat (finitehorizon): {py_entry.get('deltat')}")
print(f"  varphimin: {py_entry.get('varphimin')}")
print(f"  zetaP: {py_entry.get('zetaP')}")
print(f"  rR: {py_entry.get('rR')}")
print(f"  rerr1: {py_entry.get('rerr1')}")
print(f"  Output deltatest: {py_entry.get('deltatest')}")
print(f"  bestIdxnew: {py_entry.get('bestIdxnew')}")

# MATLAB
try:
    matlab_data = scipy.io.loadmat('optimaldeltat_matlab_log.mat', struct_as_record=False, squeeze_me=True)
    if 'log' in matlab_data:
        log = matlab_data['log']
        if isinstance(log, np.ndarray):
            mat_entry = None
            for entry in log:
                if hasattr(entry, 'step') and entry.step == 2:
                    mat_entry = entry
                    break
        else:
            mat_entry = log if hasattr(log, 'step') and log.step == 2 else None
        
        if mat_entry:
            print(f"\nMATLAB Step 2 Run 1 aux_optimaldeltat:")
            print(f"  Input deltat (finitehorizon): {getattr(mat_entry, 'deltat', None)}")
            print(f"  varphimin: {getattr(mat_entry, 'varphimin', None)}")
            print(f"  zetaP: {getattr(mat_entry, 'zetaP', None)}")
            print(f"  rR: {getattr(mat_entry, 'rR', None)}")
            print(f"  rerr1: {getattr(mat_entry, 'rerr1', None)}")
            print(f"  Output deltatest: {getattr(mat_entry, 'deltatest', None)}")
            print(f"  bestIdxnew: {getattr(mat_entry, 'bestIdxnew', None)}")
            
            # Compare inputs
            print(f"\n" + "=" * 80)
            print("INPUT COMPARISON")
            print("=" * 80)
            
            params = ['deltat', 'varphimin', 'zetaP', 'rR', 'rerr1']
            for param in params:
                py_val = py_entry.get(param)
                mat_val = getattr(mat_entry, param, None)
                if py_val is not None and mat_val is not None:
                    diff = abs(py_val - mat_val)
                    rel_diff = diff / abs(py_val) if py_val != 0 else diff
                    print(f"  {param}:")
                    print(f"    Python: {py_val}")
                    print(f"    MATLAB: {mat_val}")
                    if rel_diff < 1e-10:
                        print(f"    [MATCH]")
                    else:
                        print(f"    [DIFFERENT] rel_diff={rel_diff:.6e}")
            
            # Compare outputs
            print(f"\n" + "=" * 80)
            print("OUTPUT COMPARISON")
            print("=" * 80)
            
            py_bestIdx = py_entry.get('bestIdxnew')
            mat_bestIdx = getattr(mat_entry, 'bestIdxnew', None)
            
            print(f"  bestIdxnew:")
            print(f"    Python (0-based): {py_bestIdx}")
            print(f"    MATLAB (1-based): {mat_bestIdx}")
            if mat_bestIdx is not None:
                # MATLAB 1-based index 1 = Python 0-based index 0
                if mat_bestIdx == 1 and py_bestIdx == 0:
                    print(f"    [MATCH] Both select deltats[0] = finitehorizon")
                elif mat_bestIdx == 1:
                    print(f"    [DIFFERENT] MATLAB selects finitehorizon, Python selects index {py_bestIdx}")
                else:
                    print(f"    [DIFFERENT] MATLAB selects index {mat_bestIdx-1} (0-based), Python selects {py_bestIdx}")
            
            py_deltatest = py_entry.get('deltatest')
            mat_deltatest = getattr(mat_entry, 'deltatest', None)
            
            print(f"\n  deltatest:")
            print(f"    Python: {py_deltatest}")
            print(f"    MATLAB: {mat_deltatest}")
            if py_deltatest and mat_deltatest:
                if py_deltatest == mat_deltatest:
                    print(f"    [MATCH]")
                else:
                    print(f"    [DIFFERENT] diff={abs(py_deltatest - mat_deltatest):.15e}")
        else:
            print("\n[WARNING] Could not find MATLAB Step 2 optimaldeltat entry")
    else:
        print("\n[WARNING] optimaldeltat_matlab_log.mat does not contain 'log'")
except FileNotFoundError:
    print("\n[INFO] optimaldeltat_matlab_log.mat not found")
    print("Run MATLAB with trackOptimaldeltat enabled to generate this file")
except Exception as e:
    print(f"\n[ERROR] Could not load MATLAB log: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("If the inputs (rR, rerr1, varphimin, zetaP) differ between Python and MATLAB,")
print("then _aux_optimaldeltat will compute different objective functions,")
print("leading to different bestIdxnew selections.")
print("\nIf MATLAB's bestIdxnew = 1 (selects finitehorizon) but Python's != 0,")
print("then Python's timeStep != finitehorizon and the check fails correctly.")
print("\nThe fix is to ensure the inputs to _aux_optimaldeltat match between Python and MATLAB.")
