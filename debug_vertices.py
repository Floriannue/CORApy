import numpy as np
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.polytope.private import priv_compact_all, priv_equality_to_inequality, priv_normalize_constraints
from cora_python.g.functions.matlab.validate.check.auxiliary import combinator
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Test case from the failing test
A_outside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_outside = np.array([[7], [-2], [-4]])

print('Original constraints:')
print('A:', A_outside)
print('b:', b_outside)

# Step 1: Convert equality to inequality
A, b = priv_equality_to_inequality(A_outside, b_outside, np.array([]).reshape(0,2), np.array([]).reshape(0,1))
print('\nAfter equality to inequality:')
print('A:', A)
print('b:', b)

# Step 2: Normalize constraints
A, b, _, _ = priv_normalize_constraints(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 'A')
print('\nAfter normalization:')
print('A:', A)
print('b:', b)

# Step 3: Compact all
A, b, _, _, empty, _ = priv_compact_all(A, b, np.array([]).reshape(0,2), np.array([]).reshape(0,1), 2, 1e-12)
print('\nAfter compact_all:')
print('A:', A)
print('b:', b)
print('empty:', empty)

# Step 4: Manual vertex computation
if not empty and A.size > 0:
    nrCon, n = A.shape
    print(f'\nNumber of constraints: {nrCon}, dimension: {n}')
    
    # Generate combinations
    comb = combinator(nrCon, n, 'c')
    nrComb = comb.shape[0]
    print(f'Number of combinations: {nrComb}')
    print('Combinations:', comb)
    
    # Init vertices
    V = np.zeros((n, nrComb))
    idxKeep = np.ones(nrComb, dtype=bool)
    
    # Loop over all combinations
    for i in range(nrComb):
        indices = comb[i, :] - 1  # Convert 1-indexed to 0-indexed
        print(f'\nCombination {i+1}: indices {indices}')
        
        A_sub = A[list(indices), :]
        b_sub = b[list(indices)]
        print(f'A_sub: {A_sub}')
        print(f'b_sub: {b_sub}')
        
        # Check rank
        rank = np.linalg.matrix_rank(A_sub, tol=1e-8)
        print(f'Rank: {rank}')
        if rank < n:
            print('Rank too low, skipping')
            idxKeep[i] = False
            continue
        
        # Compute intersection point
        try:
            v = np.linalg.solve(A_sub, b_sub)
            print(f'Intersection point: {v}')
        except np.linalg.LinAlgError:
            print('Singular matrix, trying pseudo-inverse')
            try:
                v = np.linalg.pinv(A_sub) @ b_sub
                print(f'Pseudo-inverse result: {v}')
            except np.linalg.LinAlgError:
                print('Pseudo-inverse failed, skipping')
                idxKeep[i] = False
                continue
        
        # Ensure v is column vector
        v = v.reshape(-1, 1)
        V[:, i] = v.flatten()
        
        # Check if vertex is contained in polytope
        val = A @ v
        print(f'Constraint values: {val.flatten()}')
        print(f'Constraint bounds: {b.flatten()}')
        contained = np.all((val < b + 1e-12) | withinTol(val, b, 1e-12))
        print(f'Contained in polytope: {contained}')
        
        if not contained:
            print('Vertex outside polytope, skipping')
            idxKeep[i] = False
            continue
        
        # Check for duplicates
        if i > 0:
            existing_vertices = V[:, :i][:, idxKeep[:i]]
            if existing_vertices.shape[1] > 0:
                distances = np.linalg.norm(existing_vertices - v, axis=0)
                duplicate = np.any(withinTol(distances[idxKeep[:i]], 0, 1e-14))
                print(f'Duplicate check: {duplicate}')
                if duplicate:
                    print('Duplicate vertex, skipping')
                    idxKeep[i] = False
                    continue
        
        print(f'Keeping vertex {i+1}')
    
    # Remove unwanted vertices
    V = V[:, idxKeep]
    V = V[:, np.all(np.isfinite(V), axis=0)]
    
    print(f'\nFinal vertices: {V}')
    print(f'Final vertices shape: {V.shape}')
else:
    print('\nPolytope is empty or has no constraints') 