import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval

# Simple test case
def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

# Create jacobian
import sympy as sp
vars_sym = sp.symbols('x0:2')
f_sym = f(list(vars_sym))
jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
jac_func = sp.lambdify(vars_sym, jac, 'numpy')

def jacHan(x):
    if isinstance(x, Interval):
        # Evaluate at corners
        corners = []
        for dim in range(x.dim()):
            inf_val = x.inf[dim] if x.inf.size > dim else x.inf
            sup_val = x.sup[dim] if x.sup.size > dim else x.sup
            corners.append([inf_val, sup_val])
        
        import itertools
        jac_vals = []
        for combo in itertools.product(*corners):
            val = jac_func(*combo)
            jac_vals.append(val)
        
        jac_vals_arrays = [np.array(v) for v in jac_vals]
        if len(jac_vals_arrays) > 0:
            jac_vals_array = np.stack(jac_vals_arrays, axis=0)
            jac_inf = np.min(jac_vals_array, axis=0)
            jac_sup = np.max(jac_vals_array, axis=0)
            jac_center = 0.5 * (jac_inf + jac_sup)
            return jac_center
        else:
            return np.array([])
    else:
        x_flat = np.asarray(x).flatten()
        result = jac_func(*x_flat)
        result = np.asarray(result)
        if result.ndim == 0:
            return np.array([[float(result)]], dtype=float)
        elif result.ndim == 1:
            return result.reshape(1, -1)
        elif result.ndim == 2:
            if result.shape[0] != 1 and result.shape[1] == 1:
                return result.T
            return result
        else:
            result_flat = result.flatten()
            return result_flat.reshape(1, -1)

# Add debug to contractInterval
import sys
original_contractInterval = contractInterval

def debug_contractInterval(f, dom, jacHan, method=None):
    print(f"\n=== contractInterval Debug ===")
    print(f"dom: inf={dom.inf}, sup={dom.sup}")
    
    mi = dom.center()
    print(f"mi (center): {mi}")
    
    A = jacHan(mi)
    print(f"A shape: {np.asarray(A).shape}, A={A}")
    
    # Compute b
    f_mi = f(mi)
    print(f"f(mi): {f_mi}")
    
    A_mi = A @ mi
    print(f"A*mi: {A_mi}")
    
    # Get J
    J = jacHan(dom)
    print(f"J type: {type(J)}, shape: {np.asarray(J).shape if hasattr(J, 'shape') else 'N/A'}")
    
    dom_minus_mi = dom - mi
    print(f"dom - mi: inf={dom_minus_mi.inf}, sup={dom_minus_mi.sup}")
    
    if isinstance(J, Interval):
        J_center = J.center()
        J_minus_A = J_center - A
    else:
        J_minus_A = J - A
    print(f"J - A shape: {np.asarray(J_minus_A).shape}")
    
    J_minus_A_dom = J_minus_A @ dom_minus_mi
    print(f"J_minus_A_dom type: {type(J_minus_A_dom)}")
    if isinstance(J_minus_A_dom, Interval):
        print(f"  J_minus_A_dom: inf={J_minus_A_dom.inf}, sup={J_minus_A_dom.sup}, dim={J_minus_A_dom.dim()}")
    
    # Compute b
    f_mi_val = np.asarray(f_mi).flatten().reshape(-1, 1) if not isinstance(f_mi, Interval) else f_mi.center().reshape(-1, 1)
    A_mi_arr = np.asarray(A_mi).flatten().reshape(-1, 1)
    f_mi_minus_A_mi = f_mi_val - A_mi_arr
    
    if isinstance(J_minus_A_dom, Interval):
        b = Interval(f_mi_minus_A_mi.flatten(), f_mi_minus_A_mi.flatten()) + J_minus_A_dom
    else:
        J_arr = np.asarray(J_minus_A_dom).flatten().reshape(-1, 1)
        b = Interval(f_mi_minus_A_mi.flatten(), f_mi_minus_A_mi.flatten()) + Interval(J_arr.flatten(), J_arr.flatten())
    
    print(f"b type: {type(b)}, dim={b.dim()}")
    print(f"  b.inf: {b.inf}, b.sup: {b.sup}")
    
    # Now trace through the loop
    res = dom
    for i in range(dom.dim()):
        print(f"\n--- Variable {i} ---")
        for j in range(A.shape[0]):
            if abs(A[j, i]) > 1e-10:
                print(f"  Constraint {j}: A[{j},{i}]={A[j,i]}")
                a = A[j, :].copy()
                a[i] = 0
                print(f"    a: {a}")
                
                # Extract b(j)
                b_inf_flat = b.inf.flatten()
                b_sup_flat = b.sup.flatten()
                b_j = Interval(b_inf_flat[j] if j < len(b_inf_flat) else b_inf_flat[0],
                               b_sup_flat[j] if j < len(b_sup_flat) else b_sup_flat[0])
                print(f"    b_j: inf={b_j.inf}, sup={b_j.sup}")
                
                # a*res
                a_dom = a @ res
                print(f"    a_dom type: {type(a_dom)}, dim={a_dom.dim() if hasattr(a_dom, 'dim') else 'N/A'}")
                if isinstance(a_dom, Interval):
                    print(f"      a_dom: inf={a_dom.inf}, sup={a_dom.sup}")
                
                temp = -(b_j + a_dom) / A[j, i]
                print(f"    temp: inf={temp.inf}, sup={temp.sup}, dim={temp.dim()}")
                
                dom_i = Interval(res.inf[i], res.sup[i])
                print(f"    dom_i: inf={dom_i.inf}, sup={dom_i.sup}")
                
                dom_ = dom_i & temp
                print(f"    dom_ (intersection): inf={dom_.inf}, sup={dom_.sup}, dim={dom_.dim()}")
                print(f"    dom_ is empty: {dom_.representsa_('emptySet', np.finfo(float).eps)}")
                
                if dom_.representsa_('emptySet', np.finfo(float).eps):
                    print(f"    *** EMPTY SET - returning None ***")
                    return None
                
                # Update
                new_inf = res.inf.copy()
                new_sup = res.sup.copy()
                new_inf[i] = dom_.inf if np.isscalar(dom_.inf) else dom_.inf[0]
                new_sup[i] = dom_.sup if np.isscalar(dom_.sup) else dom_.sup[0]
                res = Interval(new_inf, new_sup)
                print(f"    Updated res: inf={res.inf}, sup={res.sup}")
    
    return res

try:
    result = debug_contractInterval(f, dom, jacHan, 'interval')
    print(f"\n=== Final Result ===")
    print(f"Result: {result}")
    if result is None:
        print("Result is None (empty set)")
    else:
        print(f"Result inf: {result.inf}")
        print(f"Result sup: {result.sup}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

