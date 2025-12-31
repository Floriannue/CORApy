"""Debug 'all' contractor with iter_val=2"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing 'all' with iter_val=2 ===")
print(f"Initial domain: {dom}")

# Simulate the iteration loop
dom_ = dom
for i in range(2):
    print(f"\n--- Iteration {i+1} ---")
    print(f"dom before: {dom}")
    print(f"dom_ before: {dom_}")
    
    # Call 'all' algorithm
    from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractForwardBackward import contractForwardBackward
    from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval
    from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractParallelLinearization import contractParallelLinearization
    from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPolyBoxRevise import contractPolyBoxRevise
    
    # Create jacobian
    import sympy as sp
    n = dom.dim()
    vars_sym = sp.symbols('x0:{}'.format(n))
    if n == 1:
        vars_sym = [vars_sym]
    else:
        vars_sym = list(vars_sym)
    vars_list = list(vars_sym)
    f_sym = f(vars_list)
    if isinstance(f_sym, (list, tuple, np.ndarray)):
        if len(f_sym) == 1:
            f_sym = f_sym[0] if hasattr(f_sym, '__getitem__') else f_sym
            jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
        else:
            jac_list = []
            for fi in f_sym:
                if hasattr(fi, '__iter__') and len(fi) == 1:
                    fi = fi[0]
                jac_i = [sp.diff(fi, var) for var in vars_sym]
                jac_list.append(jac_i)
            jac = sp.Matrix(jac_list)
    else:
        jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
    jac_func = sp.lambdify(vars_sym, jac, 'numpy')
    
    def jacHan(x):
        if isinstance(x, Interval):
            n = x.dim()
            corners = []
            for dim in range(n):
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
    
    # Step 1: contractForwardBackward
    print("  Step 1: contractForwardBackward")
    dom = contractForwardBackward(f, dom)
    print(f"    Result: {dom}")
    if dom is None:
        print("    contractForwardBackward returned None - returning None")
        break
    
    # Step 2: contractInterval
    if not dom.representsa_('emptySet', np.finfo(float).eps):
        print("  Step 2: contractInterval")
        dom = contractInterval(f, dom, jacHan)
        print(f"    Result: {dom}")
        if dom is None:
            print("    contractInterval returned None - returning None")
            break
    
    # Step 3: contractParallelLinearization
    if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
        print("  Step 3: contractParallelLinearization")
        dom = contractParallelLinearization(f, dom, jacHan)
        print(f"    Result: {dom}")
        if dom is None:
            print("    contractParallelLinearization returned None - returning None")
            break
    
    # Step 4: contractPolyBoxRevise
    if dom is not None and not dom.representsa_('emptySet', np.finfo(float).eps):
        print("  Step 4: contractPolyBoxRevise")
        dom = contractPolyBoxRevise(f, dom)
        print(f"    Result: {dom}")
        if dom is None:
            print("    contractPolyBoxRevise returned None - returning None")
            break
    
    # Check if empty
    if dom is None:
        print("  dom is None - returning None")
        break
    if dom.representsa_('emptySet', np.finfo(float).eps):
        print("  dom is empty set - returning None")
        break
    
    # Check convergence
    if np.all(np.abs(dom.infimum() - dom_.infimum()) < np.finfo(float).eps) and \
       np.all(np.abs(dom.supremum() - dom_.supremum()) < np.finfo(float).eps):
        print("  Converged - breaking")
        break
    else:
        dom_ = dom
        print(f"  Not converged, continuing. dom_ updated to: {dom_}")

print(f"\n=== Final Result ===")
print(f"dom: {dom}")

