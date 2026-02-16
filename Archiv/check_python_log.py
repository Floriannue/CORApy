"""Check Python log for Step 3 entries"""
import pickle

data = pickle.load(open('upstream_python_log.pkl', 'rb'))
log = data.get('upstreamLog', [])
step3 = [e for e in log if e.get('step') == 3]
print(f'Step 3 entries: {len(step3)}')
print(f'Has Rlinti_tracking: {sum(1 for e in step3 if "Rlinti_tracking" in e)}')
print(f'Has Rmax_before_reduction: {sum(1 for e in step3 if "Rmax_before_reduction" in e)}')
print(f'Has R_before_reduction: {sum(1 for e in step3 if "R_before_reduction" in e)}')

if step3:
    print(f"\nFirst Step 3 entry keys: {list(step3[0].keys())}")
