import numpy as np
import warnings
from cora_python.contSet.polytope.private.priv_equalityToInequality import priv_equalityToInequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.auxiliary import combinator

# Test case from the failing test
A_outside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_outside = np.array([[7], [-2], [-4]])

print("=== POLYTOPE VERTICES DEBUG ===")
print(f"Original constraints:")
print(f"A: {A_outside}")
print(f"b: {b_outside.flatten()}")
print()

# Step 1: Convert equality to inequality
A, b = priv_equalityToInequality(A_outside, b_outside, np.array([]).reshape(0,2), np.array([]).reshape(0,1))
print(f"After equality to inequality:")
print(f"A: {A}")
print(f"b: {b.flatten()}")
print()

# Step 2: Normalize constraints
A, b, _, _ = priv_normalizeConstraints(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 'A')
print(f"After normalization:")
print(f"A: {A}")
print(f"b: {b.flatten()}")
print()

# Step 3: Compact all
tol = 1e-12
A, b, _, _, empty, _ = priv_compact_all(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 2, tol)
print(f"After compact_all:")
print(f"A: {A}")
print(f"b: {b.flatten()}")
print(f"empty: {empty}")
print()

if empty:
    print("Polytope is empty!")
    exit()

# Step 4: Get dimensions
nrCon, n = A.shape
print(f"Number of constraints: {nrCon}, dimension: {n}")
print()

# Step 5: Generate combinations
comb = combinator(nrCon, n, 'c')
nrComb = comb.shape[0]
print(f"Number of combinations: {nrComb}")
print(f"Combinations: {comb}")
print()

# Step 6: Initialize arrays
V = np.zeros((n, nrComb))
idxKeep = np.ones(nrComb, dtype=bool)

print("=== PROCESSING COMBINATIONS ===")

# Step 7: Process each combination
for i in range(nrComb):
    print(f"\n--- Combination {i+1} ---")
    indices = comb[i, :] - 1  # Convert 1-indexed to 0-indexed
    print(f"Indices: {indices}")
    
    A_sub = A[list(indices), :]
    b_sub = b[list(indices)]
    print(f"A_sub: {A_sub}")
    print(f"b_sub: {b_sub.flatten()}")
    
    # Check rank
    rank = np.linalg.matrix_rank(A_sub, tol=1e-8)
    print(f"Rank: {rank}, n: {n}")
    if rank < n:
        print("Rank < n, skipping")
        idxKeep[i] = False
        continue
    
    # Compute intersection point
    try:
        v = np.linalg.solve(A_sub, b_sub)
        print(f"Intersection point: {v.flatten()}")
    except np.linalg.LinAlgError:
        print("LinAlgError in solve, trying pseudo-inverse")
        try:
            v = np.linalg.pinv(A_sub) @ b_sub
            print(f"Intersection point (pseudo-inverse): {v.flatten()}")
        except np.linalg.LinAlgError:
            print("LinAlgError in pseudo-inverse, skipping")
            idxKeep[i] = False
            continue
    
    # Ensure v is column vector
    v = v.reshape(-1, 1)
    V[:, i] = v.flatten()
    
    # Check if vertex is contained in polytope
    val = A @ v
    print(f"Constraint values: {val.flatten()}")
    print(f"Constraint bounds: {b.flatten()}")
    
    satisfied = (val.flatten() < b.flatten() + 1e-8) | withinTol(val.flatten(), b.flatten(), 1e-8)
    print(f"Constraints satisfied: {satisfied}")
    print(f"All satisfied: {np.all(satisfied)}")
    
    if not np.all(satisfied):
        print("Vertex outside polytope, skipping")
        idxKeep[i] = False
        continue
    
    # Check for duplicates
    if i > 0:
        existing_vertices = V[:, :i][:, idxKeep[:i]]
        print(f"Existing vertices shape: {existing_vertices.shape}")
        if existing_vertices.shape[1] > 0:
            distances = np.linalg.norm(existing_vertices - v, axis=0)
            print(f"Distances to existing vertices: {distances}")
            if np.any(withinTol(distances, 0, 1e-14)):
                print("Duplicate vertex found, skipping")
                idxKeep[i] = False
                continue
    
    print("Vertex accepted!")

print(f"\n=== FINAL RESULTS ===")
print(f"idxKeep: {idxKeep}")
print(f"Final vertices: {V[:, idxKeep]}")
print(f"Final vertices shape: {V[:, idxKeep].shape}")

# Remove vertices with Inf/NaN values
V_final = V[:, idxKeep]
V_final = V_final[:, np.all(np.isfinite(V_final), axis=0)]
print(f"After removing Inf/NaN: {V_final}")
print(f"Final shape: {V_final.shape}") 