"""
example_linear_reach_adaptive - example of adaptive reachability analysis
for linear time-invariant systems

This example demonstrates the adaptive reachability algorithm that
automatically adjusts the time step size to maintain a specified error bound.

Syntax:
    example_linear_reach_adaptive()

Inputs:
    -

Outputs:
    -

References:
    [1] M. Wetzlinger et al. "Fully automated verification of linear
        systems using inner-and outer-approximations of reachable sets",
        TAC, 2023.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 08-July-2021 (MATLAB)
Last update: 08-October-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')
sys.path.insert(0, project_root)

from cora_python.contDynamics.linearSys import LinearSys
from cora_python.contSet.zonotope import Zonotope





def example_linear_reach_adaptive():
    """
    Example of adaptive reachability analysis for linear systems
    """
    
    # System Dynamics --------------------------------------------------------
    
    # system matrix
    A = np.array([[-1, -4, 0, 0, 0],
                  [4, -1, 0, 0, 0],
                  [0, 0, -3, 1, 0],
                  [0, 0, -1, -3, 0],
                  [0, 0, 0, 0, -2]])
    
    # input matrix
    B = np.array([[1, 0],
                  [0, 0],
                  [0, 1],
                  [0, 0],
                  [0, 0]])
    
    # output matrix
    C = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]])

    # constant input offset (c) – zero vector (matches MATLAB default [])
    c = np.zeros((A.shape[0], 1))

    # create linear system object (A, B, c, C)
    sys = LinearSys('sys', A, B, c, C)
    
    # Parameters -------------------------------------------------------------
    
    # time horizon
    params = {}
    params['tStart'] = 0.0
    params['tFinal'] = 5.0
    
    # initial set
    params['R0'] = Zonotope(np.array([2, 3, 1, 1, 1]), 0.5 * np.eye(5))
    
    # input set
    params['U'] = Zonotope(np.zeros(2), 0.1 * np.eye(2))
    
    # Reachability Settings -------------------------------------------------
    
    # settings for adaptive algorithm
    options = {}
    options['linAlg'] = 'adaptive'
    options['error'] = 0.01  # maximum allowed error
    
    # Reachability Analysis -------------------------------------------------
    
    print("Computing reachable set using adaptive algorithm...")
    
    # compute reachable set
    R = sys.reach(params, options)
    
    print(f"Computation completed with {len(R.timeInterval.set)} time intervals")
    
    # Visualization ----------------------------------------------------------
    
    # plot reachable set in output space
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.sca(ax)  # contSet.plot() uses plt.gca(), so set current axes

    # plot time-interval reachable sets (dims [0,1] = first two output dimensions)
    for i, (Rset, time_interval) in enumerate(zip(R.timeInterval.set, R.timeInterval.time)):
        if hasattr(Rset, 'plot'):
            Rset.plot(color='lightblue', alpha=0.7)

    # plot time-point reachable sets
    for i, (Rset, time_point) in enumerate(zip(R.timePoint.set, R.timePoint.time)):
        if hasattr(Rset, 'plot'):
            if i == 0:  # Initial set
                Rset.plot(color='green', alpha=0.8, label='Initial set')
            elif i == len(R.timePoint.set) - 1:  # Final set
                Rset.plot(color='red', alpha=0.8, label='Final set')
    
    ax.set_xlabel('y₁')
    ax.set_ylabel('y₂')
    ax.set_title('Adaptive Reachability Analysis - Output Space')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Comparison with Standard Algorithm -------------------------------------
    
    print("\nComparing with standard algorithm...")
    
    # compute reachable set using standard algorithm
    options_std = {}
    options_std['linAlg'] = 'standard'
    options_std['timeStep'] = 0.1  # Fixed time step
    
    R_std = sys.reach(params, options_std)
    
    print(f"Standard algorithm used {len(R_std.timeInterval.set)} time intervals")
    print(f"Adaptive algorithm used {len(R.timeInterval.set)} time intervals")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Standard algorithm
    plt.sca(ax1)
    for Rset in R_std.timeInterval.set:
        if hasattr(Rset, 'plot'):
            Rset.plot(color='lightcoral', alpha=0.7)
    ax1.set_title('Standard Algorithm (Fixed Time Step)')
    ax1.set_xlabel('y₁')
    ax1.set_ylabel('y₂')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Adaptive algorithm
    plt.sca(ax2)
    for Rset in R.timeInterval.set:
        if hasattr(Rset, 'plot'):
            Rset.plot(color='lightblue', alpha=0.7)
    ax2.set_title('Adaptive Algorithm (Variable Time Step)')
    ax2.set_xlabel('y₁')
    ax2.set_ylabel('y₂')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Error Analysis ---------------------------------------------------------
    
    if hasattr(R.timeInterval, 'error') and R.timeInterval.error:
        print("\nError analysis:")
        errors = np.array(R.timeInterval.error)
        print(f"Maximum error: {np.max(errors):.6f}")
        print(f"Average error: {np.mean(errors):.6f}")
        print(f"Error bound: {options['error']:.6f}")
        
        # Plot error over time
        times = [np.mean(time_int) for time_int in R.timeInterval.time]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(times, errors, 'b-', linewidth=2, label='Actual error')
        ax.axhline(y=options['error'], color='r', linestyle='--', 
                  linewidth=2, label='Error bound')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')
        ax.set_title('Error Evolution Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    print("\nExample completed successfully!")


def example_linear_reach_adaptive_different_errors():
    """
    Example comparing adaptive reachability with different error bounds
    """
    
    print("Comparing adaptive algorithm with different error bounds...")
    
    # System Dynamics
    A = np.array([[-0.5, -2], [2, -0.5]])
    B = np.array([[1], [1]])
    sys = LinearSys('sys', A, B)
    
    # Parameters
    params = {}
    params['tStart'] = 0.0
    params['tFinal'] = 3.0
    params['R0'] = Zonotope(np.array([1, 1]), 0.1 * np.eye(2))
    params['U'] = Zonotope(np.zeros(2), 0.05 * np.eye(2))
    
    # Different error bounds
    errors = [0.1, 0.05, 0.01]
    results = []
    
    for error in errors:
        options = {}
        options['linAlg'] = 'adaptive'
        options['error'] = error
        
        R = sys.reach(params, options)
        results.append(R)
        
        print(f"Error bound {error}: {len(R.timeInterval.set)} time intervals")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (R, error) in enumerate(zip(results, errors)):
        plt.sca(axes[i])
        for Rset in R.timeInterval.set:
            if hasattr(Rset, 'plot'):
                Rset.plot(color='lightblue', alpha=0.7)

        axes[i].set_title(f'Error Bound: {error}')
        axes[i].set_xlabel('x₁')
        axes[i].set_ylabel('x₂')
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("Comparison completed!")


if __name__ == '__main__':
    # Run main example
    example_linear_reach_adaptive()
    
    # Run comparison example
    example_linear_reach_adaptive_different_errors() 