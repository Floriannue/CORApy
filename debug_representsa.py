import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.representsa_ import representsa_
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def run_debug():
    print("--- Debugging representsa_('emptySet') ---")

    # Test case 1: Polytope.empty(3)
    print("\nTesting Polytope.empty(3)")
    P_empty = Polytope.empty(3)
    print(f"P_empty is_empty: {P_empty.isemptyobject()}")
    try:
        result = representsa_(P_empty, 'emptySet', 1e-12)
        print(f"representsa_('emptySet') result for P_empty: {result}")
    except CORAerror as e:
        print(f"Caught CORAerror: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

    # Test case 2: A simple infeasible polytope (0x <= -1)
    print("\nTesting infeasible Polytope (0x <= -1)")
    A_infeasible = np.zeros((1, 1))
    b_infeasible = np.array([[-1]])
    P_infeasible = Polytope(A_infeasible, b_infeasible)
    print(f"P_infeasible is_empty: {P_infeasible.isemptyobject()}")
    try:
        result = representsa_(P_infeasible, 'emptySet', 1e-12)
        print(f"representsa_('emptySet') result for P_infeasible: {result}")
    except CORAerror as e:
        print(f"Caught CORAerror: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

    # Test case 3: A simple non-empty polytope (x <= 1)
    print("\nTesting non-empty Polytope (x <= 1)")
    A_non_empty = np.array([[1]])
    b_non_empty = np.array([[1]])
    P_non_empty = Polytope(A_non_empty, b_non_empty)
    print(f"P_non_empty is_empty: {P_non_empty.isemptyobject()}")
    try:
        result = representsa_(P_non_empty, 'emptySet', 1e-12)
        print(f"representsa_('emptySet') result for P_non_empty: {result}")
    except CORAerror as e:
        print(f"Caught CORAerror: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

if __name__ == "__main__":
    run_debug()
