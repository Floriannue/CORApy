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
import matplotlib.pyplot as plt
import time

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Try multiple import approaches
try:
    # Try as module import
    from cora_python.contDynamics.linearSys import LinearSys
    from cora_python.contSet.zonotope import Zonotope
    from cora_python.contSet.interval import Interval
except ImportError:
    try:
        # Try relative import
        from contDynamics.linearSys import LinearSys
        from contSet.zonotope import Zonotope
        from contSet.interval import Interval
    except ImportError:
        # Final fallback
        import sys
        import os
        # Add parent directories to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sys.path.insert(0, parent_dir)
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
    simOpt['points'] = 25
    simOpt['type'] = 'gaussian'
    
    simRes = fiveDimSys.simulateRandom(params, simOpt)
    
    
    # Visualization -----------------------------------------------------------
    
    # Plot different projections
    dims = [[0, 1], [2, 3]]  # Python uses 0-based indexing
    
    for k, projDims in enumerate(dims):
        
        plt.figure(figsize=(8, 6))
        plt.hold = True  # Enable hold for multiple plots
        plt.box(True)
        
        # Note: useCORAcolors functionality would be implemented separately
        # For now, we'll use matplotlib's default color cycle
        
        # Plot reachable sets
        if hasattr(R, 'plot'):
            R.plot(projDims, label='Reachable set', Unify=True)
        else:
            print(f"Warning: Plot method not yet implemented for reachSet")
            # Fallback: plot time-point sets if available
            if hasattr(R, 'timePoint') and 'set' in R.timePoint:
                for i, rset in enumerate(R.timePoint['set']):
                    if hasattr(rset, 'plot'):
                        if i == 0:
                            rset.plot(projDims, label='Reachable set', alpha=0.7)
                        else:
                            rset.plot(projDims, alpha=0.7)
        
        # Plot initial set
        if hasattr(R, 'R0'):
            if hasattr(R.R0, 'plot'):
                R.R0.plot(projDims, label='Initial set', facecolor='green', alpha=0.8)
        elif hasattr(params['R0'], 'plot'):
            params['R0'].plot(projDims, label='Initial set', facecolor='green', alpha=0.8)
        else:
            print(f"Warning: Plot method not yet implemented for initial set")
        
        # Plot simulation results
        if hasattr(simRes[0], 'plot'):
            simRes[0].plot(projDims, label='Simulations')
        else:
            print(f"Warning: Plot method not yet implemented for simResult")
            # Fallback: plot trajectories manually
            for sim in simRes:
                if hasattr(sim, 'x') and hasattr(sim, 't'):
                    traj = sim.x[0]  # First trajectory
                    if traj.shape[1] > max(projDims):
                        plt.plot(traj[:, projDims[0]], traj[:, projDims[1]], 
                                'b-', alpha=0.6, linewidth=0.8)
        
        # Label plot
        plt.xlabel(f'x_{projDims[0]+1}')  # Display 1-based indexing for user
        plt.ylabel(f'x_{projDims[1]+1}')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.title(f'Reachable Set Projection: Dimensions {projDims[0]+1}-{projDims[1]+1}')
    
    plt.show()
    
    # Example completed
    print("\n=== Example completed successfully! ===")
    return True


if __name__ == "__main__":
    example_linear_reach_01_5dim() 