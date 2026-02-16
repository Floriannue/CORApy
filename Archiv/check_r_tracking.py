"""check_r_tracking - Check if R tracking data is in the log"""

import pickle

# Load Python log
with open('upstream_python_log.pkl', 'rb') as f:
    data = pickle.load(f)

entries = data.get('upstreamLog', [])
print(f"Total entries: {len(entries)}")

# Find Step 3 entries
step3_entries = [e for e in entries if e.get('step') == 3]
print(f"\nStep 3 entries: {len(step3_entries)}")

if step3_entries:
    # Get the last one (converged)
    e = step3_entries[-1]
    print(f"\nLast Step 3 entry keys: {list(e.keys())}")
    
    has_r_before = 'R_before_reduction' in e
    has_rred_after = 'Rred_after_reduction' in e
    
    print(f"\nHas R_before_reduction: {has_r_before}")
    print(f"Has Rred_after_reduction: {has_rred_after}")
    
    if has_r_before:
        r = e['R_before_reduction']
        print(f"\nR before reduction:")
        print(f"  Type: {type(r)}")
        if isinstance(r, dict):
            print(f"  Keys: {list(r.keys())}")
            print(f"  num_generators: {r.get('num_generators', 'N/A')}")
            print(f"  redFactor: {r.get('redFactor', 'N/A')}")
            print(f"  diagpercent: {r.get('diagpercent', 'N/A')}")
    
    if has_rred_after:
        rred = e['Rred_after_reduction']
        print(f"\nRred after reduction:")
        print(f"  Type: {type(rred)}")
        if isinstance(rred, dict):
            print(f"  Keys: {list(rred.keys())}")
            print(f"  num_generators: {rred.get('num_generators', 'N/A')}")
    
    # Check all Step 3 entries
    print(f"\nChecking all Step 3 entries:")
    for i, entry in enumerate(step3_entries):
        has_r_before = 'R_before_reduction' in entry
        has_rred_after = 'Rred_after_reduction' in entry
        print(f"  Entry {i+1} (run={entry.get('run', 'N/A')}): R_before={has_r_before}, Rred_after={has_rred_after}")
