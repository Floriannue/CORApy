import sys
sys.path.insert(0, 'cora_python')
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

I = Interval([-2, -1], [3, 4])
point = [3, 4]
S = np.asarray(point, dtype=float)
tol = 1e-12

print(f"Interval: inf={I.inf}, sup={I.sup}")
print(f"Point: {S}")
print(f"Tolerance: {tol}")

# Check the exact conditions from the contains_ function
print("\n=== Containment conditions ===")

# Lower bound check: I.inf < S + tol
lower_strict = I.inf < S + tol
print(f"I.inf < S + tol: {I.inf} < {S + tol} = {lower_strict}")

# Lower bound tolerance check: withinTol(I.inf, S, tol)
lower_tol = withinTol(I.inf, S, tol)
print(f"withinTol(I.inf, S, tol): withinTol({I.inf}, {S}, {tol}) = {lower_tol}")

# Combined lower check
lower_check = lower_strict | lower_tol
print(f"Lower check: {lower_check}")

# Upper bound check: I.sup > S - tol
upper_strict = I.sup > S - tol
print(f"I.sup > S - tol: {I.sup} > {S - tol} = {upper_strict}")

# Upper bound tolerance check: withinTol(I.sup, S, tol)
upper_tol = withinTol(I.sup, S, tol)
print(f"withinTol(I.sup, S, tol): withinTol({I.sup}, {S}, {tol}) = {upper_tol}")

# Combined upper check
upper_check = upper_strict | upper_tol
print(f"Upper check: {upper_check}")

# Final containment check
containment_check = lower_check & upper_check
print(f"Containment check: {containment_check}")

# Final result
res = np.all(containment_check)
print(f"Final result: {res}")

print("\n=== MATLAB-style check ===")
# Try the exact MATLAB condition
matlab_style = np.all((I.inf < S + tol) | withinTol(I.inf, S, tol)) & np.all((I.sup > S - tol) | withinTol(I.sup, S, tol))
print(f"MATLAB style result: {matlab_style}")

# Check individual MATLAB components
matlab_lower = np.all((I.inf < S + tol) | withinTol(I.inf, S, tol))
matlab_upper = np.all((I.sup > S - tol) | withinTol(I.sup, S, tol))
print(f"MATLAB lower: {matlab_lower}")
print(f"MATLAB upper: {matlab_upper}") 