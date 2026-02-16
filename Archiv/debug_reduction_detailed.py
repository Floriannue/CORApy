"""debug_reduction_detailed - Detailed debugging of reduction algorithm"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.zonotope.private.priv_reduceAdaptive import priv_reduceAdaptive

print("=" * 80)
print("DETAILED REDUCTION ALGORITHM DEBUGGING")
print("=" * 80)

# Load actual R from Step 3
import pickle
with open('upstream_python_log.pkl', 'rb') as f:
    data = pickle.load(f)

entries = data.get('upstreamLog', [])
step3_entries = [e for e in entries if e.get('step') == 3 and 'R_before_reduction' in e]

if step3_entries:
    entry = step3_entries[-1]
    R_before = entry['R_before_reduction']
    
    center = np.asarray(R_before['center'])
    generators = np.asarray(R_before['generators'])
    redFactor = R_before['redFactor']
    diagpercent = R_before['diagpercent']
    
    print(f"\nStep 3 R (before reduction):")
    print(f"  Center shape: {center.shape}")
    print(f"  Generators shape: {generators.shape}")
    print(f"  Number of generators: {generators.shape[1]}")
    print(f"  redFactor: {redFactor}")
    print(f"  diagpercent: {diagpercent}")
    
    # Reconstruct Z
    Z = Zonotope(center, generators)
    
    # Manually trace through the reduction algorithm
    print(f"\n" + "=" * 80)
    print("MANUAL TRACE OF REDUCTION ALGORITHM")
    print("=" * 80)
    
    G = Z.generators()
    Gabs = np.abs(G)
    n, nrG = G.shape
    
    # Compute dHmax
    Gbox = np.sum(Gabs, axis=1, keepdims=True)
    dHmax = (diagpercent * 2) * np.sqrt(np.sum(Gbox ** 2))
    
    print(f"\nStep 1: Compute dHmax")
    print(f"  Gbox shape: {Gbox.shape}")
    print(f"  Gbox sum: {np.sum(Gbox)}")
    print(f"  dHmax: {dHmax:.10e}")
    
    # Girard method
    print(f"\nStep 2: Girard method")
    norminf = np.max(Gabs, axis=0)
    normsum = np.sum(Gabs, axis=0)
    diff = normsum - norminf
    
    print(f"  norminf shape: {norminf.shape}, max: {np.max(norminf):.10e}")
    print(f"  normsum shape: {normsum.shape}, max: {np.max(normsum):.10e}")
    print(f"  diff shape: {diff.shape}, max: {np.max(diff):.10e}")
    
    # Sort indices
    idx = np.argpartition(diff, nrG - 1)[:nrG]
    idx = idx[np.argsort(diff[idx])]
    h = diff[idx]
    
    print(f"  idx: {idx}")
    print(f"  h: {h}")
    print(f"  h <= dHmax: {h <= dHmax}")
    
    if not np.any(h):
        print(f"  All h are zero - early return")
    else:
        # Box generators with h = 0
        hzeroIdx = idx[h == 0]
        Gzeros = np.sum(Gabs[:, hzeroIdx], axis=1, keepdims=True)
        last0Idx = len(hzeroIdx)
        gensred = Gabs[:, idx[last0Idx:]]
        
        print(f"\nStep 3: Process generators")
        print(f"  hzeroIdx: {hzeroIdx}")
        print(f"  last0Idx: {last0Idx}")
        print(f"  gensred shape: {gensred.shape}")
        
        # Compute mugensred
        maxidx = np.argmax(gensred, axis=0)
        maxval = np.max(gensred, axis=0)
        nrG_red = nrG - last0Idx
        mugensred = np.zeros((n, nrG_red))
        cols = n * np.arange(nrG_red)
        mugensred.flat[cols + maxidx] = maxval
        
        print(f"  maxidx: {maxidx}")
        print(f"  maxval: {maxval}")
        
        # Compute gensdiag and h
        gensdiag = np.cumsum(gensred - mugensred, axis=1)
        h = 2 * np.linalg.norm(gensdiag, axis=0, ord=2)
        
        print(f"\nStep 4: Compute h values")
        print(f"  gensdiag shape: {gensdiag.shape}")
        print(f"  h: {h}")
        print(f"  h <= dHmax: {h <= dHmax}")
        
        # Find redIdx
        redIdx_arr = np.where(h <= dHmax)[0]
        if len(redIdx_arr) == 0:
            redIdx = 0
            print(f"  No valid indices - redIdx = 0")
        else:
            redIdx_0based = redIdx_arr[-1]
            redIdx = redIdx_0based + 1
            print(f"  redIdx_arr: {redIdx_arr}")
            print(f"  redIdx_0based: {redIdx_0based}")
            print(f"  redIdx (1-based): {redIdx}")
        
        # Compute final generators
        if redIdx > 0:
            Gred = np.sum(gensred[:, :redIdx], axis=1, keepdims=True)
        else:
            Gred = np.zeros((n, 1))
        
        if last0Idx + redIdx < len(idx):
            gunred_idx = idx[last0Idx + redIdx:]
            gunred_idx_sorted = np.sort(gunred_idx)
            Gunred = G[:, gunred_idx_sorted]
        else:
            Gunred = np.array([]).reshape(n, 0)
        
        Gred_total = (Gred + Gzeros).flatten()
        G_diag = np.diag(Gred_total)
        G_diag = G_diag[:, np.any(G_diag, axis=0)]
        G_new = np.hstack([Gunred, G_diag]) if Gunred.size > 0 else G_diag
        
        print(f"\nStep 5: Final generators")
        print(f"  Gred shape: {Gred.shape}")
        print(f"  Gunred shape: {Gunred.shape}")
        print(f"  G_diag shape: {G_diag.shape}")
        print(f"  G_new shape: {G_new.shape}")
        print(f"  Final number of generators: {G_new.shape[1]}")
        
        # Now run the actual function
        print(f"\n" + "=" * 80)
        print("ACTUAL FUNCTION RESULT")
        print("=" * 80)
        Z_red, dHerror, gredIdx = priv_reduceAdaptive(Z, diagpercent, 'girard')
        G_red = Z_red.generators()
        print(f"  Reduced generators: {G_red.shape[1]}")
        print(f"  dHerror: {dHerror:.10e}")
        print(f"  gredIdx length: {len(gredIdx)}")
        print(f"  gredIdx: {gredIdx}")
        
        if G_red.shape[1] != G_new.shape[1]:
            print(f"\n  *** MISMATCH: Manual trace = {G_new.shape[1]}, Function = {G_red.shape[1]} ***")
        else:
            print(f"\n  Match!")

else:
    print("No Step 3 entry with R_before_reduction found")

print("\n" + "=" * 80)
