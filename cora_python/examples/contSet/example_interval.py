"""
example_interval - example instantiation of interval objects

This example demonstrates basic interval operations including creation,
intersection, and visualization.

Syntax:
    completed = example_interval()

Inputs:
    -

Outputs:
    completed - true/false

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       21-April-2018 (MATLAB)
Last update:   ---
Last revision: ---
Python translation: 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Use absolute import with fallback
try:
    from cora_python.contSet.interval import Interval
except ImportError:
    from contSet.interval import Interval


def example_interval():
    """
    Example instantiation of interval objects
    
    This example demonstrates basic interval operations including:
    - Creating intervals
    - Computing radius and center
    - Intersection operations
    - Visualization
    
    Returns:
        bool: True if example completed successfully
    """
    
    print("=== Interval Example ===\n")
    
    # Create intervals
    I1 = Interval([0, -1], [3, 1])  # create interval I1
    I2 = Interval([-1, -1.5], [1, -0.5])  # create interval I2
    
    print("I1 =", I1)
    print("I2 =", I2)
    print()
    
    # Obtain and display radius of I1
    r = I1.rad()  # obtain and display radius of I1
    print("Radius of I1:", r)
    print()
    
    # Compute intersection of I1 and I2
    I3 = I1 & I2  # computes the intersection of I1 and I2
    print("I3 = I1 & I2 =", I3)
    print()
    
    # Return and display the center of I3
    c = I3.center()  # returns and displays the center of I3
    print("Center of I3:", c)
    print()
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.hold = True  # Enable hold for multiple plots
    
    # Plot I3 in blue
    if hasattr(I3, 'plot'):
        I3.plot([1, 2], facecolor='blue', alpha=0.7, label='I3 (intersection)')
    else:
        # Fallback plotting if plot method not available
        print("Warning: Plot method not implemented for intervals")
        # Manual plotting
        inf_vals = I3.inf
        sup_vals = I3.sup
        if len(inf_vals) >= 2:
            x_coords = [inf_vals[0], sup_vals[0], sup_vals[0], inf_vals[0], inf_vals[0]]
            y_coords = [inf_vals[1], inf_vals[1], sup_vals[1], sup_vals[1], inf_vals[1]]
            plt.fill(x_coords, y_coords, color='blue', alpha=0.7, label='I3 (intersection)')
    
    # Plot I1
    if hasattr(I1, 'plot'):
        I1.plot([1, 2], facecolor='red', alpha=0.5, label='I1')
    else:
        # Manual plotting
        inf_vals = I1.inf
        sup_vals = I1.sup
        if len(inf_vals) >= 2:
            x_coords = [inf_vals[0], sup_vals[0], sup_vals[0], inf_vals[0], inf_vals[0]]
            y_coords = [inf_vals[1], inf_vals[1], sup_vals[1], sup_vals[1], inf_vals[1]]
            plt.plot(x_coords, y_coords, 'r-', linewidth=2, label='I1')
    
    # Plot I2
    if hasattr(I2, 'plot'):
        I2.plot([1, 2], facecolor='green', alpha=0.5, label='I2')
    else:
        # Manual plotting
        inf_vals = I2.inf
        sup_vals = I2.sup
        if len(inf_vals) >= 2:
            x_coords = [inf_vals[0], sup_vals[0], sup_vals[0], inf_vals[0], inf_vals[0]]
            y_coords = [inf_vals[1], inf_vals[1], sup_vals[1], sup_vals[1], inf_vals[1]]
            plt.plot(x_coords, y_coords, 'g-', linewidth=2, label='I2')
    
    # Format plot
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Interval Operations Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Show plot
    plt.show()
    
    # Example completed
    print("=== Example completed successfully! ===")
    return True


if __name__ == "__main__":
    example_interval() 