"""
Deep investigation of contractInterval differences between MATLAB and Python
"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval

# Test case from testLong_contract
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

print("=== Step-by-step investigation ===")
print(f"Initial dom: inf={dom.inf}, sup={dom.sup}")

# Step 1: Compute mi, A, J
mi = dom.center()
print(f"\n1. mi (center): {mi}")

A = jacHan(mi)
print(f"2. A (jacobian at mi): {A}, shape={np.asarray(A).shape}")

# Get J using taylm method (default)
try:
    from cora_python.contSet.taylm.taylm import Taylm
    tay = Taylm(dom)
    J_interval = Interval(jacHan(tay))
    J = J_interval
    print(f"3. J (interval from taylm): inf={J.inf}, sup={J.sup}, dim={J.dim()}")
except Exception as e:
    print(f"3. J (fallback): {e}")
    J = jacHan(dom)
    if not isinstance(J, Interval):
        J = Interval(J)
    print(f"   J: inf={J.inf}, sup={J.sup}, dim={J.dim()}")

# Step 2: Compute b = f(mi) - A * mi + (J - A) * (dom - mi)
f_mi = f(mi)
print(f"\n4. f(mi): {f_mi}, type={type(f_mi)}")

A_mi = A @ mi
print(f"5. A * mi: {A_mi}, type={type(A_mi)}")

dom_minus_mi = dom - mi
print(f"6. dom - mi: inf={dom_minus_mi.inf}, sup={dom_minus_mi.sup}, dim={dom_minus_mi.dim()}")

J_minus_A = J - A
print(f"7. J - A: inf={J_minus_A.inf}, sup={J_minus_A.sup}, dim={J_minus_A.dim()}")

J_minus_A_dom = J_minus_A @ dom_minus_mi
print(f"8. (J - A) * (dom - mi): inf={J_minus_A_dom.inf}, sup={J_minus_A_dom.sup}, dim={J_minus_A_dom.dim()}")

# Extract numeric from f_mi
if isinstance(f_mi, Interval):
    f_mi = f_mi.center()
f_mi = np.asarray(f_mi).flatten()
A_mi = np.asarray(A_mi).flatten()
f_mi_minus_A_mi = f_mi - A_mi
print(f"9. f(mi) - A*mi: {f_mi_minus_A_mi}")

b = f_mi_minus_A_mi + J_minus_A_dom
print(f"10. b = f(mi) - A*mi + (J-A)*(dom-mi): inf={b.inf}, sup={b.sup}, dim={b.dim()}")

# Step 3: Loop through variables and constraints
print(f"\n=== Loop through variables and constraints ===")
res = dom.copy() if hasattr(dom, 'copy') else Interval(dom.inf.copy(), dom.sup.copy())

for i in range(dom.dim()):
    print(f"\n--- Variable {i} (dom[{i}] = [{res.inf[i]}, {res.sup[i]}]) ---")
    for j in range(A.shape[0]):
        if abs(A[j, i]) > 1e-10:
            print(f"  Constraint {j}: A[{j},{i}]={A[j,i]}")
            
            # MATLAB: a = A(j,:); a(i) = 0;
            a = A[j, :].copy()
            a[i] = 0
            print(f"    a (A[{j},:] with a[{i}]=0): {a}")
            
            # MATLAB: b(j) - extract j-th element
            b_j = Interval(b.inf[j], b.sup[j])
            print(f"    b[{j}]: inf={b_j.inf}, sup={b_j.sup}, dim={b_j.dim()}")
            
            # MATLAB: a*dom - matrix multiplication
            # In MATLAB, dom is updated in place, so we use res
            print(f"    res before a*res: inf={res.inf}, sup={res.sup}")
            a_dom = a @ res
            print(f"    a*res: inf={a_dom.inf}, sup={a_dom.sup}, dim={a_dom.dim()}")
            print(f"    a_dom.inf.ndim={a_dom.inf.ndim}, a_dom.sup.ndim={a_dom.sup.ndim}")
            
            # Ensure scalar
            if a_dom.dim() > 1 or (a_dom.inf.ndim > 1 or a_dom.sup.ndim > 1):
                print(f"      Flattening a_dom...")
                a_dom = Interval(a_dom.inf.flatten()[0] if a_dom.inf.size > 0 else 0.0,
                                 a_dom.sup.flatten()[0] if a_dom.sup.size > 0 else 0.0)
                print(f"      After flattening: inf={a_dom.inf}, sup={a_dom.sup}, dim={a_dom.dim()}")
            
            # MATLAB: temp = -(b(j) + a*dom)/A(j,i);
            temp = -(b_j + a_dom) / A[j, i]
            print(f"    temp = -(b[{j}] + a*res)/A[{j},{i}]: inf={temp.inf}, sup={temp.sup}, dim={temp.dim()}")
            
            # Ensure scalar
            if temp.dim() > 1 or (temp.inf.ndim > 1 or temp.sup.ndim > 1):
                print(f"      Flattening temp...")
                temp = Interval(temp.inf.flatten()[0] if temp.inf.size > 0 else 0.0,
                                temp.sup.flatten()[0] if temp.sup.size > 0 else 0.0)
                print(f"      After flattening: inf={temp.inf}, sup={temp.sup}, dim={temp.dim()}")
            
            # MATLAB: dom_ = dom(i) & temp;
            dom_i = Interval(res.inf[i], res.sup[i])
            print(f"    dom[{i}]: inf={dom_i.inf}, sup={dom_i.sup}, dim={dom_i.dim()}")
            print(f"    temp: inf={temp.inf}, sup={temp.sup}, dim={temp.dim()}")
            
            dom_ = dom_i & temp
            print(f"    dom_ = dom[{i}] & temp: inf={dom_.inf}, sup={dom_.sup}, dim={dom_.dim()}")
            print(f"    dom_ is empty: {dom_.representsa_('emptySet', np.finfo(float).eps)}")
            
            if not dom_.representsa_('emptySet', np.finfo(float).eps):
                # MATLAB: dom(i) = dom_;
                new_inf = res.inf.copy()
                new_sup = res.sup.copy()
                new_inf[i] = dom_.inf[0] if dom_.inf.size > 0 else dom_.inf
                new_sup[i] = dom_.sup[0] if dom_.sup.size > 0 else dom_.sup
                res = Interval(new_inf, new_sup)
                print(f"    Updated res[{i}]: [{res.inf[i]}, {res.sup[i]}]")
            else:
                print(f"    *** EMPTY SET - would return None ***")
                break

print(f"\n=== Final Result ===")
print(f"res: inf={res.inf}, sup={res.sup}")

