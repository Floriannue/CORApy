"""
Debug script to trace step-by-step through MATLAB verify.m and compare with Python verify.py
"""
import numpy as np
import sys
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.neuralNetwork import NeuralNetwork
from nn.layers.linear.nnLinearLayer import nnLinearLayer
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.validateNNoptions import validateNNoptions

# Create the exact network from the test
layers = [
    nnLinearLayer(
        np.array([[0.6294, 0.2647], [0.8116, -0.8049], [-0.7460, -0.4430], [0.8268, 0.0938]], dtype=np.float32),     
        np.array([[0.9150], [0.9298], [-0.6848], [0.9412]], dtype=np.float32)
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[0.9143, -0.1565, 0.3115, 0.3575],
                  [-0.0292, 0.8315, -0.9286, 0.5155],
                  [0.6006, 0.5844, 0.6983, 0.4863],
                  [-0.7162, 0.9190, 0.8680, -0.2155]], dtype=np.float32),
        np.array([[0.3110], [-0.6576], [0.4121], [-0.9363]], dtype=np.float32)
    ),
    nnReLULayer(),
    nnLinearLayer(
        np.array([[-0.4462, -0.8057, 0.3897, 0.9004],
                  [-0.9077, 0.6469, -0.3658, -0.9311]], dtype=np.float32),
        np.array([[-0.1225], [-0.2369]], dtype=np.float32)
    ),
]
nn = NeuralNetwork(layers)

# Test inputs
x = np.array([[0], [0]], dtype=np.float32)  # center
r = np.array([[1], [1]], dtype=np.float32)  # radius
A = np.array([[-1, 1]], dtype=np.float32)  # Shape: (1, 2)
b = np.array([[-2.27]], dtype=np.float32)
safeSet = False

# Options
options = {}
options['nn'] = {
    'use_approx_error': True,
    'poly_method': 'bounds',
    'train': {
        'backprop': False,
        'mini_batch_size': 512,
        'use_gpu': False
    }
}
options = validateNNoptions(options, True)
options['nn']['interval_center'] = False

print("=" * 80)
print("STEP-BY-STEP DEBUG: MATLAB verify.m vs Python verify.py")
print("=" * 80)

# Step 1: Initialization (MATLAB lines 38-74)
print("\n=== STEP 1: Initialization ===")
nSplits = 5
nDims = 1
totalNumSplits = 0
verifiedPatches = 0
bs = options.get('nn', {}).get('train', {}).get('mini_batch_size', 32)
inputDataClass = np.float32
n0 = x.shape[0]
xs = x.copy()
rs = r.copy()
print(f"nSplits={nSplits}, nDims={nDims}, bs={bs}, n0={n0}")
print(f"xs shape: {xs.shape}, rs shape: {rs.shape}")

# Step 2: First iteration - Pop batch (MATLAB lines 96-100)
print("\n=== STEP 2: Pop batch (MATLAB aux_pop) ===")
bs_actual = min(bs, xs.shape[1])
idx = np.arange(bs_actual)
xi = xs[:, idx].copy()
xs = np.delete(xs, idx, axis=1)
ri = rs[:, idx].copy()
rs = np.delete(rs, idx, axis=1)
print(f"Popped: xi shape={xi.shape}, ri shape={ri.shape}")
print(f"xi:\n{xi}")
print(f"ri:\n{ri}")

# Step 3: Falsification - Compute sensitivity (MATLAB lines 105-109)
print("\n=== STEP 3: Falsification - Compute sensitivity ===")
import torch
device = torch.device('cpu')
xi_torch = torch.tensor(xi, dtype=torch.float32, device=device)
ri_torch = torch.tensor(ri, dtype=torch.float32, device=device)
idxLayer = list(range(len(nn.layers)))

S, _ = nn.calcSensitivity(xi_torch, options, store_sensitivity=False)
print(f"S shape: {S.shape if isinstance(S, np.ndarray) else S.shape}")
if isinstance(S, np.ndarray):
    S = torch.tensor(S, dtype=torch.float32, device=device)
S = torch.maximum(S, torch.tensor(1e-3, dtype=torch.float32, device=device))
print(f"S after max(1e-3) shape: {S.shape}")
print(f"S[:,:,0] (first batch):\n{S[:,:,0].cpu().numpy()}")

# MATLAB: sens = permute(sum(abs(S)),[2 1 3]);
S_abs = torch.abs(S)
sens_sum = torch.sum(S_abs, dim=0)  # (n0, cbSz)
sens = sens_sum.permute(1, 0)  # (cbSz, n0)
sens_np = sens.cpu().numpy()
print(f"sens shape: {sens_np.shape}")
print(f"sens:\n{sens_np}")

# Step 4: Compute adversarial attack (MATLAB line 112)
print("\n=== STEP 4: Compute adversarial attack (MATLAB line 112) ===")
# MATLAB: zi = xi + ri.*sign(sens);
sens_sign = torch.sign(sens)  # (cbSz, n0)
sens_sign_T = sens_sign.T  # (n0, cbSz)
print(f"sign(sens) shape: {sens_sign.shape}")
print(f"sign(sens):\n{sens_sign.cpu().numpy()}")
print(f"sign(sens).T shape: {sens_sign_T.shape}")
print(f"sign(sens).T:\n{sens_sign_T.cpu().numpy()}")
zi = xi_torch + ri_torch * sens_sign_T
print(f"zi shape: {zi.shape}")
print(f"zi:\n{zi.cpu().numpy()}")

# Step 5: Check adversarial examples (MATLAB lines 114-119)
print("\n=== STEP 5: Check adversarial examples ===")
yi = nn.evaluate_(zi, options, idxLayer)
if isinstance(yi, torch.Tensor):
    yi_np = yi.cpu().numpy()
else:
    yi_np = yi
print(f"yi shape: {yi_np.shape}")
print(f"yi:\n{yi_np}")

ld_yi = A @ yi_np + b
print(f"A*yi + b shape: {ld_yi.shape}")
print(f"A*yi + b:\n{ld_yi}")

# MATLAB: checkSpecs = all(A*yi + b <= 0,1); (for safeSet=False)
checkSpecs = np.all(ld_yi <= 0, axis=0)
print(f"checkSpecs (all(A*yi + b <= 0)): {checkSpecs}")

if np.any(checkSpecs):
    print("\n*** COUNTEREXAMPLE FOUND IN FIRST ITERATION ***")
    idNzEntry = np.where(checkSpecs)[0]
    id_ = idNzEntry[0]
    print(f"Counterexample index: {id_}")
    print(f"Counterexample input: {zi[:, id_].cpu().numpy()}")
    print(f"Counterexample output: {yi_np[:, id_]}")
    print(f"A*y + b for counterexample: {ld_yi[:, id_]}")
else:
    print("\n*** NO COUNTEREXAMPLE - Continue to verification ***")

print("\n" + "=" * 80)
print("End of Step-by-Step Debug")
print("=" * 80)

