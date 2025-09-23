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

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.insert(0, project_root)

from cora_python.contSet.interval import Interval


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
    
    print(I1)
    print(I2)
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

    # Plot I1
    I1.plot([1, 2], FaceColor='red', alpha=0.5, label='I1')
    
    # Plot I2
    I2.plot([1, 2], FaceColor='green', alpha=0.5, label='I2')

    I3.plot([1, 2], FaceColor='blue', alpha=0.7, label='I3 (intersection)')

    # Format plot
    plt.xlabel('x_1')
    plt.ylabel('x_2')
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