"""track_upstream_python - Track upstream computations in Python"""

import numpy as np
from cora_python.contDynamics.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.models.Cora.contDynamics.nonlinearSys.models.jetEngine import jetEngine
import pickle

dim_x = 2
params = {}
params['tFinal'] = 8.0
params['R0'] = Zonotope(np.array([[1], [1]]), 0.1 * np.eye(dim_x))
params['U'] = Zonotope(np.zeros((1, 1)), np.array([]).reshape(1, 0))
params['tStart'] = 0.0

options = {}
options['alg'] = 'lin-adaptive'
options['trackUpstream'] = True  # Enable upstream tracking
options['trackOptimaldeltat'] = True  # Also track optimaldeltat
options['progress'] = True  # Enable progress output for debugging

sys = NonlinearSys(name='jetEngine', fun=jetEngine, states=dim_x, inputs=1)

print("Running Python with upstream tracking...")
R = sys.reach(params, options)

# Extract logs
upstream_log = options.get('upstreamLog', [])
optimaldeltat_log = options.get('optimaldeltatLog', [])

# Extract Rtp tracking from options (stored per step during reach_adaptive)
Rtp_tracking = options.get('Rtp_tracking_by_step', {})

print(f"\nCaptured {len(upstream_log)} upstream computation entries")
print(f"Captured {len(optimaldeltat_log)} optimaldeltat entries")

# Save to file
with open('upstream_python_log.pkl', 'wb') as f:
    pickle.dump({
        'upstreamLog': upstream_log,
        'optimaldeltatLog': optimaldeltat_log,
        'Rtp_tracking': Rtp_tracking,
    }, f)

print("Saved to upstream_python_log.pkl")

# Show first few entries
print("\nFirst 3 upstream entries:")
for i, entry in enumerate(upstream_log[:3]):
    print(f"\nEntry {i+1} (Step {entry.get('step', 'N/A')}, Run {entry.get('run', 'N/A')}):")
    if 'Z_before_quadmap' in entry and entry['Z_before_quadmap']:
        z = entry['Z_before_quadmap']
        print(f"  Z before quadMap: radius_max={z.get('radius_max', 'N/A')}")
    if 'errorSec_before_combine' in entry and entry['errorSec_before_combine']:
        es = entry['errorSec_before_combine']
        print(f"  errorSec before combine: radius_max={es.get('radius_max', 'N/A')}")
    if 'VerrorDyn_before_reduce' in entry and entry['VerrorDyn_before_reduce']:
        vd = entry['VerrorDyn_before_reduce']
        print(f"  VerrorDyn before reduce: radius_max={vd.get('radius_max', 'N/A')}")
    if 'VerrorDyn_after_reduce' in entry and entry['VerrorDyn_after_reduce']:
        vd = entry['VerrorDyn_after_reduce']
        print(f"  VerrorDyn after reduce: radius_max={vd.get('radius_max', 'N/A')}")
    if 'Rerror_before_optimaldeltat' in entry and entry['Rerror_before_optimaldeltat']:
        re = entry['Rerror_before_optimaldeltat']
        print(f"  Rerror before optimaldeltat: rerr1={re.get('rerr1', 'N/A')}")
