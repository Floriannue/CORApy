"""
custom example_zonotope
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../..')
sys.path.insert(0, project_root)

from cora_python.contSet.zonotope import Zonotope


def example_zonotope():
    
    print("=== Zonotope Example ===\n")
    
    # Create intervals
    Z1 = Zonotope([0, 0], [[1, 0, .5], [0, 1, .5]])
    Z2 = Zonotope([-1, 1], [[-1, 0], [0, 1]])
    Z4 = Z1 + Z2 # Minkowski addition
    Z3 = Z1.convHull(Z2) # Convex hull
    
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.hold = True

    Z4.plot([1, 2], FaceColor=np.array([0.9290, 0.6940, 0.1250]), alpha=0.3, label='Z4')
    Z3.plot([1, 2], FaceColor=np.array([0.2706, 0.5882, 1.0000]), alpha=0.3, label='Z3')
    Z2.plot([1, 2], FaceColor=np.array([0.4660, 0.6740, 0.1880]), alpha=0.4, label='Z2')
    Z1.plot([1, 2], FaceColor=np.array([0.8500, 0.3250, 0.0980]), alpha=0.4, label='Z1')
        
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig("ZonoConvHull_figure_python.svg")


    # Show plot
    plt.show()
    
    # Example completed
    print("=== Example completed successfully! ===")
    return True


if __name__ == "__main__":
    example_zonotope() 