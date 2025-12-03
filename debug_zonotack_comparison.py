"""
Debug script to compare zonotack attack computation between MATLAB and Python
This will help identify why different counterexamples are produced
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.neuralNetwork import NeuralNetwork
from nn.layers.linear.nnLinearLayer import nnLinearLayer
from nn.layers.nonlinear.nnReLULayer import nnReLULayer

# Create the neural network (same as test)
layers = [
    nnLinearLayer(
        np.array([[0.6294, 0.2647], [0.8116, -0.8049], [-0.7460, -0.4430], [0.8268, 0.0938]], dtype=np.float32),
        np.array([[0.9150], [0.9298], [-0.6848], [0.9412]], dtype=np.float32)
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[0.9143, -0.1565, 0.3115, 0.3575], [-0.0292, 0.8315, -0.9286, 0.5155], 
                 [0.6006, 0.5844, 0.6983, 0.4863], [-0.7162, 0.9190, 0.8680, -0.2155]], dtype=np.float32),
        np.array([[0.3110], [-0.6576], [0.4121], [-0.9363]], dtype=np.float32)
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[-0.4462, -0.8057, 0.3897, 0.9004], [-0.9077, 0.6469, -0.3658, -0.9311]], dtype=np.float32),
        np.array([[-0.1225], [-0.2369]], dtype=np.float32)
    ),
]
nn = NeuralNetwork(layers)

# Test parameters
xi = np.array([[0.0], [0.0]], dtype=np.float32)
ri = np.array([[1.0], [1.0]], dtype=np.float32)
A = np.array([[-1, 1]], dtype=np.float32)
b = np.array([[-1.27]], dtype=np.float32)
safeSet = False

# Options
options = {
    'nn': {
        'use_approx_error': True,
        'poly_method': 'bounds',
        'train': {
            'backprop': False,
            'mini_batch_size': 512,
            'num_init_gens': 2
        },
        'interval_center': False,
        'falsification_method': 'zonotack',
        'refinement_method': 'zonotack',
        'input_generator_heuristic': 'sensitivity'
    }
}

print("=== Zonotack Attack Computation Debug ===\n")
print(f"Input: xi = {xi.flatten()}, ri = {ri.flatten()}")
print(f"Bounds: [{xi[0,0]-ri[0,0]}, {xi[0,0]+ri[0,0]}] x [{xi[1,0]-ri[1,0]}, {xi[1,0]+ri[1,0]}]")
print(f"A = {A.flatten()}, b = {b.flatten()}")
print()

# Construct input zonotope (simplified - we'll need to call the actual function)
from nn.neuralNetwork import verify_helpers
n0, cbSz = xi.shape
numInitGens = min(options['nn']['train']['num_init_gens'], n0)

# Prepare network for zonotope evaluation to get proper batchG size
idxLayer = list(range(len(nn.layers)))
q = nn.prepareForZonoBatchEval(xi, options, idxLayer)
batchG = np.zeros((n0, q, cbSz), dtype=np.float32)

print(f"numInitGens = {numInitGens}")
print(f"q (num generators) = {q}")
print()

# Construct input zonotope
# For numInitGens >= n0, we don't need sens/grad
# But for numInitGens < n0, we'd need them for the heuristic
# Since numInitGens=2 and n0=2, we have numInitGens >= n0, so we can pass None

# Debug: Let's manually trace through the construction
print("=== Manual Generator Construction Debug ===\n")
from nn.layers.linear.nnGeneratorReductionLayer import sub2ind, repelem

# Simulate what _aux_constructInputZonotope does
Gxi_debug = batchG[:, :, :cbSz]  # (n0, q, cbSz)
if numInitGens >= n0:
    dimIdx_debug = np.tile(np.arange(1, n0 + 1).reshape(-1, 1), (1, cbSz))  # 1-based: (n0, bSz)
else:
    # Would compute heuristic here
    dimIdx_debug = None

print(f"dimIdx_debug:\n{dimIdx_debug}")
print(f"Gxi_debug shape before: {Gxi_debug.shape}")

# Compute indices
dimIdx_flat = dimIdx_debug.flatten('F')  # Column-major flatten, 1-based
genIdx_flat = np.tile(np.arange(1, numInitGens + 1), (1, cbSz)).flatten('F')  # 1-based
batchIdx_flat = repelem(np.arange(1, cbSz + 1), numInitGens, 1).flatten('F')  # 1-based

print(f"dimIdx_flat: {dimIdx_flat}")
print(f"genIdx_flat: {genIdx_flat}")
print(f"batchIdx_flat: {batchIdx_flat}")

gIdx = sub2ind(Gxi_debug.shape, dimIdx_flat, genIdx_flat, batchIdx_flat)  # 1-based linear indices
print(f"gIdx from sub2ind: {gIdx}")
print(f"gIdx - 1 (0-based): {gIdx - 1}")

# Now call the actual function
cxi, Gxi, dimIdx = verify_helpers._aux_constructInputZonotope(
    options, 'sensitivity', xi, ri, batchG, None, None, numInitGens
)

print("Input Zonotope:")
print(f"  cxi shape: {cxi.shape}")
print(f"  Gxi shape: {Gxi.shape}")
print(f"  dimIdx: {dimIdx}")
print(f"  Gxi[:,:numInitGens,:]:")
print(Gxi[:, :numInitGens, :])
print(f"  sum(|Gxi|, axis=1): {np.sum(np.abs(Gxi[:, :numInitGens, :]), axis=1).flatten()}")
print(f"  ri: {ri.flatten()}")
print()

# Evaluate zonotope through network
yi, Gyi = nn.evaluateZonotopeBatch_(cxi, Gxi, options, idxLayer)

print("Output Zonotope:")
print(f"  yi shape: {yi.shape}, yi = {yi.flatten()}")
print(f"  Gyi shape: {Gyi.shape}")
print()

# Compute logit difference
yic = yi  # interval_center is False
# yic has shape (2, 1, 1), need to reshape to (2, 1) for matrix multiplication
yic_reshaped = yic.reshape(yic.shape[0], -1)  # (2, 1)
ld_yi = A @ yic_reshaped  # (1, 1)
ld_Gyi = np.einsum('ij,jkl->ikl', A, Gyi)  # (1, num_gens, batch)

print("Logit Difference:")
print(f"  ld_yi: {ld_yi.flatten()}")
print(f"  ld_Gyi shape: {ld_Gyi.shape}")
print(f"  ld_Gyi[:,:numInitGens,:]: {ld_Gyi[:, :numInitGens, :].flatten()}")
print()

# Zonotack attack
p = A.shape[0]
ld_Gyi_subset = ld_Gyi[:, :numInitGens, :]  # (p, numInitGens, cbSz)
beta_ = -np.sign(ld_Gyi_subset)  # (p, numInitGens, cbSz)

print("Zonotack Attack Computation:")
print(f"  p (num constraints) = {p}")
print(f"  ld_Gyi_subset shape: {ld_Gyi_subset.shape}")
print(f"  ld_Gyi_subset values: {ld_Gyi_subset.flatten()}")
print(f"  sign(ld_Gyi_subset): {np.sign(ld_Gyi_subset).flatten()}")
print(f"  beta_ (before permute): {beta_.flatten()}")
print()

# permute [2 4 1 3]: (p, numInitGens, cbSz) -> (numInitGens, 1, p, cbSz)
beta_ = beta_[:, np.newaxis, :, :]  # (p, 1, numInitGens, cbSz)
beta_ = np.transpose(beta_, (2, 1, 0, 3))  # (numInitGens, 1, p, cbSz)

if safeSet:
    numUnionConst = A.shape[0]
    beta_[:, :, :numUnionConst, :] = -beta_[:, :, :numUnionConst, :]

beta = beta_.reshape(numInitGens, 1, p * cbSz)

print(f"  beta_ (after permute) shape: {beta_.shape}")
print(f"  beta_ values: {beta_.flatten()}")
print(f"  beta shape: {beta.shape}")
print(f"  beta values: {beta.flatten()}")
print()

# Compute attack: delta = pagemtimes(repelem(Gxi(:,1:numInitGens,:),1,1,p),beta)
Gxi_subset = Gxi[:, :numInitGens, :]  # (n0, numInitGens, cbSz)
Gxi_repeated = np.repeat(Gxi_subset, p, axis=2)  # (n0, numInitGens, p*cbSz)

print(f"  Gxi_subset shape: {Gxi_subset.shape}")
print(f"  Gxi_subset:\n{Gxi_subset[:, :, 0]}")
print(f"  Gxi_repeated shape: {Gxi_repeated.shape}")
print()

# pagemtimes: (n0, numInitGens, p*cbSz) @ (numInitGens, 1, p*cbSz) = (n0, 1, p*cbSz)
delta = np.einsum('ijk,jlk->ilk', Gxi_repeated, beta)  # (n0, 1, p*cbSz)
delta = delta.squeeze(1)  # (n0, p*cbSz)

print(f"  delta shape: {delta.shape}")
print(f"  delta:\n{delta}")
print(f"  |delta|: {np.abs(delta).flatten()}")
print(f"  ri: {ri.flatten()}")
print(f"  |delta| <= ri: {np.all(np.abs(delta) <= ri, axis=0)}")
print()

# xi_ = repelem(xi,1,p) + delta(:,:)
xi_repeated = np.repeat(xi, p, axis=1)  # (n0, p*cbSz)
zi = xi_repeated + delta

print("Attack Points (zi):")
print(f"  zi shape: {zi.shape}")
print(f"  zi:\n{zi}")
print(f"  zi in bounds: {np.all(zi >= xi - ri) and np.all(zi <= xi + ri)}")
print()

# Evaluate attack points
yi_attack = nn.evaluate_(zi, options, idxLayer)
# yi_attack has shape (2, p*cbSz), need to reshape for proper indexing
if yi_attack.ndim == 2 and yi_attack.shape[0] == 2:
    pass  # Already correct shape
else:
    yi_attack = yi_attack.reshape(2, -1)
ld_yi_attack = A @ yi_attack

print("Attack Point Evaluations:")
for i in range(zi.shape[1]):
    print(f"  Point {i}: zi = {zi[:, i]}, yi = {yi_attack[:, i]}, A*yi = {ld_yi_attack[0, i]:.6f}")
    print(f"    Violation (A*yi <= b): {ld_yi_attack[0, i] <= b[0, 0]}")
print()

# Check which points are counterexamples
if safeSet:
    checkSpecs = np.any(ld_yi_attack > b, axis=0)
else:
    checkSpecs = np.all(ld_yi_attack <= b, axis=0)

print("Counterexample Check:")
print(f"  checkSpecs: {checkSpecs}")
print(f"  First counterexample index: {np.where(checkSpecs)[0][0] if np.any(checkSpecs) else None}")
if np.any(checkSpecs):
    id_ = np.where(checkSpecs)[0][0]
    print(f"  Selected counterexample: zi[:, {id_}] = {zi[:, id_]}")
    print(f"  MATLAB counterexample: [1; 1]")
    print(f"  Match: {np.allclose(zi[:, id_], np.array([[1.0], [1.0]]))}")

