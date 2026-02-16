"""
example_linear_basic - Basic example of linear system usage

This example demonstrates how to create and work with linear systems
in the Python CORA implementation.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import numpy as np
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../..')
sys.path.insert(0, project_root)

from cora_python.contDynamics import LinearSys


def example_linear_basic():
    """
    Basic example demonstrating LinearSys functionality
    
    Returns:
        bool: True if example completed successfully
    """
    
    print("=== Linear System Example ===\n")
    
    # Example 1: Simple 2D system
    print("1. Creating a simple 2D linear system:")
    A = np.array([[-2, 0], 
                  [1, -3]])
    B = np.array([[1], 
                  [1]])
    C = np.array([[1, 0]])
    
    sys1 = LinearSys('simple_2D_system', A, B, None, C)
    sys1.display()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: System from MATLAB manual example
    print("2. Creating system from CORA manual example:")
    A_manual = np.array([[-2, 0], 
                         [1, -3]])
    B_manual = np.array([[1], 
                         [1]])
    C_manual = np.array([[1, 0]])
    
    sys2 = LinearSys(A_manual, B_manual, None, C_manual)
    sys2.display()
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: More complex system with all matrices
    print("3. Creating a more complex system with all matrices:")
    A_complex = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                          [0.1362, 0.2742, 0.5195, 0.8266],
                          [0.0502, -0.1051, -0.6572, 0.3874],
                          [1.0227, -0.4877, 0.8342, -0.2372]])
    
    B_complex = 0.25 * np.array([[-2, 0, 3],
                                 [2, 1, 0],
                                 [0, 0, 1],
                                 [0, -2, 1]])
    
    c_complex = 0.05 * np.array([[-4], [2], [3], [1]])
    
    C_complex = np.array([[1, 1, 0, 0],
                          [0, -0.5, 0.5, 0]])
    
    D_complex = np.array([[0, 0, 1],
                          [0, 0, 0]])
    
    k_complex = np.array([[0], [0.02]])
    
    sys3 = LinearSys('complex_system', A_complex, B_complex, c_complex, 
                     C_complex, D_complex, k_complex)
    sys3.display()
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Random system generation
    print("4. Generating random linear systems:")
    
    # Default random system
    sys_random1 = LinearSys.generateRandom()
    print("Random system (default parameters):")
    print(f"  States: {sys_random1.nr_of_dims}")
    print(f"  Inputs: {sys_random1.nr_of_inputs}")
    print(f"  Outputs: {sys_random1.nr_of_outputs}")
    
    # Random system with specified dimensions
    sys_random2 = LinearSys.generateRandom(state_dimension=3,
                                          input_dimension=2,
                                          output_dimension=1)
    print(f"\nRandom system (3 states, 2 inputs, 1 output):")
    print(f"  States: {sys_random2.nr_of_dims}")
    print(f"  Inputs: {sys_random2.nr_of_inputs}")
    print(f"  Outputs: {sys_random2.nr_of_outputs}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: System comparison
    print("5. Comparing linear systems:")
    
    # Create two identical systems
    sys_a = LinearSys(A, B)
    sys_b = LinearSys(A, B)
    sys_c = LinearSys(A, B + 0.1)
    
    print(f"sys_a == sys_b: {sys_a == sys_b}")
    print(f"sys_a == sys_c: {sys_a == sys_c}")
    print(f"sys_a != sys_c: {sys_a != sys_c}")
    
    # Test with tolerance
    sys_d = LinearSys(A, B + 1e-15)  # Very small difference
    print(f"sys_a.isequal(sys_d): {sys_a.isequal(sys_d)}")
    print(f"sys_a.isequal(sys_d, tol=1e-10): {sys_a.isequal(sys_d, tol=1e-10)}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 6: System properties
    print("6. Accessing system properties:")
    print(f"System name: {sys1.name}")
    print(f"Number of states: {sys1.nr_of_dims}")
    print(f"Number of inputs: {sys1.nr_of_inputs}")
    print(f"Number of outputs: {sys1.nr_of_outputs}")
    print(f"A matrix shape: {sys1.A.shape}")
    print(f"B matrix shape: {sys1.B.shape}")
    print(f"C matrix shape: {sys1.C.shape}")
    
    print("\nA matrix:")
    print(sys1.A)
    print("\nB matrix:")
    print(sys1.B)
    print("\nC matrix:")
    print(sys1.C)
    
    print("\n=== Example completed successfully! ===")
    return True


if __name__ == "__main__":
    example_linear_basic() 