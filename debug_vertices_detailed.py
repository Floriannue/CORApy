import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.check.auxiliary import combinator

# Test the complete vertex computation process
A = np.array([[-0.70710678, -0.70710678],
              [1.0, 0.0],
              [-0.70710678, 0.70710678]])
b = np.array([-2.82842712, 7.0, -1.41421356])

print("Testing complete vertex computation process")
print("A:", A)
print("b:", b)

nrCon, n = A.shape
print(f"Number of constraints: {nrCon}, dimension: {n}")

# Generate combinations
comb = combinator(nrCon, n, 'c')
nrComb = comb.shape[0]
print(f"Number of combinations: {nrComb}")
print("Combinations:", comb)

# Init vertices
V = np.zeros((n, nrComb))
idxKeep = np.ones(nrComb, dtype=bool)

print("\nProcessing each combination:")
for i in range(nrComb):
    indices = comb[i, :] - 1  # Convert 1-indexed to 0-indexed
    print(f"\nCombination {i+1}: indices {indices}")
    
    A_sub = A[list(indices), :]
    b_sub = b[list(indices)]
    print(f"A_sub: {A_sub}")
    print(f"b_sub: {b_sub}")
    
    # Check rank
    rank = np.linalg.matrix_rank(A_sub, tol=1e-8)
    print(f"Rank: {rank}")
    if rank < n:
        print("Rank too low, skipping")
        idxKeep[i] = False
        continue
    
    # Compute intersection point
    try:
        v = np.linalg.solve(A_sub, b_sub)
        print(f"Intersection point: {v}")
    except np.linalg.LinAlgError:
        print("Singular matrix, trying pseudo-inverse")
        try:
            v = np.linalg.pinv(A_sub) @ b_sub
            print(f"Pseudo-inverse result: {v}")
        except np.linalg.LinAlgError:
            print("Pseudo-inverse failed, skipping")
            idxKeep[i] = False
            continue
    
    # Ensure v is column vector
    v = v.reshape(-1, 1)
    V[:, i] = v.flatten()
    
    # Check if vertex is contained in polytope
    val = A @ v
    print(f"Constraint values: {val.flatten()}")
    print(f"Constraint bounds: {b}")
    contained = np.all((val.flatten() < b + 1e-8) | withinTol(val.flatten(), b, 1e-8))
    print(f"Contained in polytope: {contained}")
    
    if not contained:
        print("Vertex outside polytope, skipping")
        idxKeep[i] = False
        continue
    
    # Check for duplicates
    if i > 0:
        existing_vertices = V[:, :i][:, idxKeep[:i]]
        if existing_vertices.shape[1] > 0:
            distances = np.linalg.norm(existing_vertices - v, axis=0)
            duplicate = np.any(withinTol(distances[idxKeep[:i]], 0, 1e-14))
            print(f"Duplicate check: {duplicate}")
            if duplicate:
                print("Duplicate vertex, skipping")
                idxKeep[i] = False
                continue
    
    print(f"Keeping vertex {i+1}")

print(f"\nFinal idxKeep: {idxKeep}")

# Remove unwanted vertices
V = V[:, idxKeep]
print(f"Vertices after idxKeep filter: {V}")
print(f"Vertices shape after idxKeep: {V.shape}")

# Remove vertices with Inf/NaN values
V = V[:, np.all(np.isfinite(V), axis=0)]
print(f"Final vertices: {V}")
print(f"Final vertices shape: {V.shape}") 