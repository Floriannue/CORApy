"""
example_linear_reach_01_5dim - example of linear reachability analysis
   with uncertain inputs, can be found in [1, Sec. 3.2.3].

Syntax:
    res = example_linear_reach_01_5dim()

Inputs:
    -

Outputs:
    res - true/false 

References:
    [1] M. Althoff, "Reachability analysis and its application to the 
        safety assessment of autonomous cars", Dissertation, TUM 2010

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       17-August-2016 (MATLAB)
Last update:   23-April-2020 (restructure params/options) (MATLAB)
Last revision: ---
Python translation: 2025
"""

import sys
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

# Configure matplotlib for smooth rendering like MATLAB
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.antialiased'] = True
plt.rcParams['patch.antialiased'] = True
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.solid_capstyle'] = 'round'
plt.rcParams['lines.solid_joinstyle'] = 'round'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['patch.linewidth'] = 0.5

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')
sys.path.insert(0, project_root)

# Import the required modules
from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval


def example_linear_reach_01_5dim():
    """
    Example of linear reachability analysis with uncertain inputs
    
    This example demonstrates reachability analysis for a 5-dimensional linear system
    with uncertain inputs, corresponding to [1, Sec. 3.2.3].
    
    Returns:
        bool: True if example completed successfully
    """
    
    print("=== Linear Reachability Analysis Example (5D) ===\n")
    
    # Parameters --------------------------------------------------------------
    
    params = {}
    params['tFinal'] = 5
    params['R0'] = Zonotope(np.ones((5, 1)), 0.1 * np.diag(np.ones(5)))
    params['U'] = Zonotope(Interval(np.array([0.9, -0.25, -0.1, 0.25, -0.75]), 
                                   np.array([1.1, 0.25, 0.1, 0.75, -0.25])))
    
    
    # Reachability Settings ---------------------------------------------------
    
    options = {}
    options['timeStep'] = 0.02
    options['taylorTerms'] = 4
    options['zonotopeOrder'] = 20
    options['linAlg'] = 'standard'  # Use standard reachability algorithm
    
    
    # System Dynamics ---------------------------------------------------------
    
    A = np.array([[-1, -4, 0, 0, 0], 
                  [4, -1, 0, 0, 0], 
                  [0, 0, -3, 1, 0], 
                  [0, 0, -1, -3, 0], 
                  [0, 0, 0, 0, -2]])
    B = 1
    
    fiveDimSys = LinearSys('fiveDimSys', A, B)
    
    
    # Reachability Analysis ---------------------------------------------------
    
    print("Computing reachable set...")
    tic = time.time()
    R = fiveDimSys.reach(params, options)
    tComp = time.time() - tic
    
    print(f'Computation time of reachable set: {tComp:.4f} seconds')
    
    
    # Simulation --------------------------------------------------------------
    
    print("Running random simulations...")
    simOpt = {}
    simOpt['points'] = 25  # Match MATLAB exactly
    simOpt['type'] = 'gaussian'
    
    simRes = fiveDimSys.simulateRandom(params, simOpt)
    
    
    # Visualization -----------------------------------------------------------
    
    # Plot different projections
    dims = [[0, 1], [2, 3]]  # Python uses 0-based indexing
    
    for k, projDims in enumerate(dims):
        
        plt.figure()
        plt.hold = True if hasattr(plt, 'hold') else None
        
        # Plot reachable sets (MATLAB-style simplicity)
        R.plot(projDims, DisplayName='Reachable set', Unify=True)
        
        # Plot initial set (MATLAB-style simplicity)
        R.R0.plot(projDims, DisplayName='Initial set')
        
        # Plot simulation results (MATLAB-style simplicity)
        simRes.plot(projDims, DisplayName='Simulations')
        
        # Label plot (MATLAB-style with proper LaTeX formatting)
        plt.xlabel(f'$x_{{{projDims[0]+1}}}$')
        plt.ylabel(f'$x_{{{projDims[1]+1}}}$')
        plt.legend(loc='upper left')  # Position legend outside plot area
        plt.tight_layout()  # Adjust layout to prevent overlap
        
        # Save plot instead of showing it
        plt.savefig(f'example_5dim_projection_{projDims[0]+1}_{projDims[1]+1}.png', dpi=300, bbox_inches='tight')
        print(f'Saved plot for projection [{projDims[0]+1}, {projDims[1]+1}]')
    
    print("Plots saved successfully!")
    plt.show()
    plt.close()
    
    # Example completed
    print("\n=== Example completed successfully! ===")
    return True


if __name__ == "__main__":
    example_linear_reach_01_5dim() 