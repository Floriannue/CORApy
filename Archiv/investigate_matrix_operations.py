"""
Investigate order of operations in matrix multiplications
Compare step-by-step with MATLAB to find precision differences
"""
import numpy as np
from cora_python.models.Cora.tank.tank6Eq import tank6Eq
from cora_python.contDynamics.nonlinearSys.nonlinearSys import NonlinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contDynamics.nonlinearSys.linearize import linearize
from cora_python.contDynamics.linearSys.particularSolution_timeVarying import particularSolution_timeVarying

# Setup
dim_x = 6
params = {
    'R0': Zonotope(np.array([[2], [4], [4], [2], [10], [4]]), 0.2 * np.eye(dim_x)),
    'U': Zonotope(np.zeros((1, 1)), 0.005 * np.eye(1)),
    'tFinal': 4,
    'uTrans': np.zeros((1, 1))
}

options = {
    'timeStep': 4,
    'taylorTerms': 4,
    'zonotopeOrder': 50,
    'alg': 'lin',
    'tensorOrder': 2,
    'maxError': np.full((dim_x, 1), np.inf)
}

# System
tank = NonlinearSys(tank6Eq, states=6, inputs=1)

# Compute derivatives
from cora_python.contDynamics.contDynamics.derivatives import derivatives
derivatives(tank, options)

# Linearize
sys, linsys, linParams, linOptions = linearize(tank, params['R0'], params, options)

# Get inputs
U = linParams['U']
timeStep = options['timeStep']
truncationOrder = options['taylorTerms']

print("=" * 80)
print("INVESTIGATING MATRIX OPERATIONS ORDER")
print("=" * 80)

# Decompose U
blocks = np.array([[1, linsys.nr_of_dims]])
U_decomp = U.decompose(blocks)

# Initialize Ptp
Ptp = timeStep * U
Ptp = Ptp.decompose(blocks)

print(f"\nInitial Ptp:")
print(f"  Center: {Ptp.c.flatten()}")
print(f"  Generators shape: {Ptp.G.shape}")
print(f"  First generator: {Ptp.G[:, 0] if Ptp.G.size > 0 else 'N/A'}")

# Trace through the loop
options_dict = {'timeStep': timeStep, 'ithpower': 1}
for eta in range(1, truncationOrder + 1):
    print(f"\n{'='*80}")
    print(f"ETA = {eta}")
    print(f"{'='*80}")
    
    # Get Apower_mm
    options_dict['ithpower'] = eta
    Apower_mm = linsys.getTaylor('Apower', options_dict)
    print(f"\nApower_mm (A^{eta}):")
    print(f"  Shape: {Apower_mm.shape}")
    print(f"  First row: {Apower_mm[0, :]}")
    print(f"  Max abs value: {np.max(np.abs(Apower_mm))}")
    
    # Get dtoverfac
    options_dict['ithpower'] = eta + 1
    dtoverfac = linsys.getTaylor('dtoverfac', options_dict)
    print(f"\ndtoverfac (dt^{eta+1}/{eta+1}!):")
    print(f"  Value: {dtoverfac:.15e}")
    print(f"  Type: {type(dtoverfac)}")
    
    # Compute addTerm = Apower_mm * dtoverfac
    # In MATLAB: addTerm = Apower_mm * dtoverfac (element-wise multiplication)
    # In Python: Apower_mm * dtoverfac (element-wise multiplication)
    addTerm = Apower_mm * dtoverfac
    print(f"\naddTerm = Apower_mm * dtoverfac:")
    print(f"  Shape: {addTerm.shape}")
    print(f"  First row: {addTerm[0, :]}")
    print(f"  Max abs value: {np.max(np.abs(addTerm))}")
    print(f"  Type: {type(addTerm)}")
    
    # Check if this matches manual computation
    addTerm_manual = Apower_mm * dtoverfac
    if np.allclose(addTerm, addTerm_manual):
        print(f"  [OK] Matches manual computation")
    else:
        print(f"  [ERROR] Does NOT match manual computation!")
        print(f"    Max diff: {np.max(np.abs(addTerm - addTerm_manual))}")
    
    # Compute Ptp_eta = block_mtimes(addTerm, U_decomp)
    from cora_python.g.functions.helper.sets.contSet.contSet import block_mtimes
    Ptp_eta = block_mtimes(addTerm, U_decomp)
    print(f"\nPtp_eta = block_mtimes(addTerm, U_decomp):")
    print(f"  Type: {type(Ptp_eta)}")
    if hasattr(Ptp_eta, 'c'):
        print(f"  Center: {Ptp_eta.c.flatten()}")
        print(f"  Generators shape: {Ptp_eta.G.shape}")
        print(f"  First generator: {Ptp_eta.G[:, 0] if Ptp_eta.G.size > 0 else 'N/A'}")
    
    # Check manual computation: addTerm @ U
    # In MATLAB: block_mtimes performs matrix multiplication
    # For non-decomposed case: addTerm @ U
    Ptp_eta_manual = addTerm @ U
    print(f"\nPtp_eta_manual = addTerm @ U (direct matrix multiplication):")
    if hasattr(Ptp_eta_manual, 'c'):
        print(f"  Center: {Ptp_eta_manual.c.flatten()}")
        print(f"  Generators shape: {Ptp_eta_manual.G.shape}")
        print(f"  First generator: {Ptp_eta_manual.G[:, 0] if Ptp_eta_manual.G.size > 0 else 'N/A'}")
        
        # Compare centers
        center_diff = np.abs(Ptp_eta.c - Ptp_eta_manual.c).flatten()
        print(f"  Center difference: {center_diff}")
        print(f"  Max center diff: {np.max(center_diff)}")
        
        # Compare generators
        if Ptp_eta.G.shape == Ptp_eta_manual.G.shape:
            gen_diff = np.abs(Ptp_eta.G - Ptp_eta_manual.G)
            print(f"  Max generator diff: {np.max(gen_diff)}")
    
    # Store Ptp before addition
    Ptp_before = Ptp.c.copy() if hasattr(Ptp, 'c') else None
    
    # Compute Ptp = block_operation(@plus, Ptp, Ptp_eta)
    from cora_python.g.functions.helper.sets.contSet.contSet import block_operation
    Ptp = block_operation(lambda a, b: a + b, Ptp, Ptp_eta)
    
    print(f"\nPtp after addition:")
    print(f"  Center: {Ptp.c.flatten()}")
    print(f"  Generators shape: {Ptp.G.shape}")
    if Ptp_before is not None:
        center_change = (Ptp.c - Ptp_before).flatten()
        print(f"  Center change from previous: {center_change}")
        print(f"  Max center change: {np.max(np.abs(center_change))}")

# Check remainder term
print(f"\n{'='*80}")
print("REMAINDER TERM")
print(f"{'='*80}")
if not np.isinf(truncationOrder):
    from cora_python.contDynamics.linearSys.private.priv_expmRemainder import priv_expmRemainder
    E = priv_expmRemainder(linsys, timeStep, truncationOrder)
    print(f"\nE (remainder term):")
    print(f"  Type: {type(E)}")
    if hasattr(E, 'inf'):
        print(f"  Inf shape: {E.inf.shape}")
        print(f"  Sup shape: {E.sup.shape}")
        print(f"  Inf first row: {E.inf[0, :] if len(E.inf.shape) > 1 else E.inf[0]}")
        print(f"  Sup first row: {E.sup[0, :] if len(E.sup.shape) > 1 else E.sup[0]}")
    
    E_times_dt = E * timeStep
    print(f"\nE * timeStep:")
    print(f"  Type: {type(E_times_dt)}")
    if hasattr(E_times_dt, 'inf'):
        print(f"  Inf first row: {E_times_dt.inf[0, :] if len(E_times_dt.inf.shape) > 1 else E_times_dt.inf[0]}")
        print(f"  Sup first row: {E_times_dt.sup[0, :] if len(E_times_dt.sup.shape) > 1 else E_times_dt.sup[0]}")

print(f"\n{'='*80}")
print("FINAL RESULT")
print(f"{'='*80}")
print(f"Ptp center: {Ptp.c.flatten()}")
print(f"Ptp generators shape: {Ptp.G.shape}")

# Compute delta
from cora_python.g.functions.helper.precision.kahan_sum import kahan_sum_abs
delta = kahan_sum_abs(Ptp.G, axis=1)
print(f"Delta (sum of abs generators): {delta}")

print("\n" + "=" * 80)
