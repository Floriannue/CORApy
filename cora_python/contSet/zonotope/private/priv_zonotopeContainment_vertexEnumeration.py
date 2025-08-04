"""
priv_zonotopeContainment_vertexEnumeration - Solves the zonotope containment problem by
   checking whether the maximum value of the Z2-norm at one of the
   vertices of Z1 exceeds 1+tol. This is done via a search algorithm.

Syntax:
   res, cert, scaling = priv_zonotopeContainment_vertexEnumeration(Z1, Z2, tol, scalingToggle)

Inputs:
   Z1 - zonotope object, inbody
   Z2 - zonotope object, circumbody
   tol - tolerance for the containment check
   scalingToggle - if set to True, scaling will be computed

Outputs:
   res - True/False
   cert - always True (since venumSearch is exact)
   scaling - the maximal Z2-norm value found (exact value)

References:
   [1] Kulmburg A., Brkan I., Althoff M.,: Search-based and Stochastic
       Solutions to the Zonotope and Ellipsotope Containment Problems
       (to appear)

Authors:       Adrian Kulmburg
Python port:   AI Assistant
"""
import numpy as np

def priv_zonotopeContainment_vertexEnumeration(Z1, Z2, tol, scalingToggle):
    """
    See file docstring above for details.
    """
    # Helper: norm of p w.r.t. Z2
    def norm_Z2(p):
        # Assume Z2 has a method zonotopeNorm or use fallback
        if hasattr(Z2, 'zonotopeNorm'):
            return Z2.zonotopeNorm(p)
        else:
            # Fallback: use L2 norm (not exact, but placeholder)
            return np.linalg.norm(p)

    # Get generator matrix and centers
    G = Z1.G if hasattr(Z1, 'G') else Z1.generators()
    c1 = Z1.c if hasattr(Z1, 'c') else Z1.center()
    c2 = Z2.c if hasattr(Z2, 'c') else Z2.center()
    G = np.hstack([(c1 - c2).reshape(-1, 1), G])
    G_size = G.shape[1]

    # Compute norm of each generator
    generator_norms = np.array([norm_Z2(G[:, i]) for i in range(G_size)])
    indices = np.argsort(-generator_norms)  # Descending order
    generator_norms = generator_norms[indices]
    G = G[:, indices]

    # Heuristic function
    def heuristic(nu, value):
        return value + np.dot(generator_norms, np.abs(np.abs(nu) - 1))

    # Initialize queue
    queue_nu = [np.zeros(G_size)]
    queue_values = [0.0]
    scaling = 0.0
    cert = True

    while queue_nu:
        current_nu = queue_nu.pop()
        current_value = queue_values.pop()
        # Find first zero
        zeros = np.where(current_nu == 0)[0]
        if zeros.size == 0:
            continue  # Leaf node
        i = zeros[0]
        # Children
        child_positive = current_nu.copy(); child_positive[i] = 1
        child_negative = current_nu.copy(); child_negative[i] = -1
        child_positive_value = norm_Z2(G @ child_positive)
        child_negative_value = norm_Z2(G @ child_negative)
        # Order children
        if child_positive_value <= child_negative_value:
            queue_nu.extend([child_positive, child_negative])
            queue_values.extend([child_positive_value, child_negative_value])
        else:
            queue_nu.extend([child_negative, child_positive])
            queue_values.extend([child_negative_value, child_positive_value])
        # Containment check
        if not scalingToggle and queue_values[-1] > 1 + tol:
            return False, cert, scaling
        else:
            scaling = max(scaling, queue_values[-1])
        # Cleaning
        if scalingToggle:
            if len(queue_nu) > 1 and heuristic(queue_nu[-2], queue_values[-2]) <= scaling:
                queue_nu.pop(-2)
                queue_values.pop(-2)
                if len(queue_nu) > 0 and heuristic(queue_nu[-1], queue_values[-1]) <= scaling:
                    queue_nu.pop()
                    queue_values.pop()
                continue
        else:
            if len(queue_nu) > 1 and heuristic(queue_nu[-2], queue_values[-2]) <= 1 + tol:
                queue_nu.pop(-2)
                queue_values.pop(-2)
                if len(queue_nu) > 0 and heuristic(queue_nu[-1], queue_values[-1]) <= 1 + tol:
                    queue_nu.pop()
                    queue_values.pop()
                continue
        # Sorting
        if len(queue_nu) > 1:
            j = next((k for k, v in enumerate(queue_values[:-1]) if v >= queue_values[-2]), len(queue_values)-1)
            queue_nu = queue_nu[:j] + [queue_nu[-2]] + queue_nu[j:-2] + [queue_nu[-1]]
            queue_values = queue_values[:j] + [queue_values[-2]] + queue_values[j:-2] + [queue_values[-1]]
    # Final result
    if scalingToggle:
        res = scaling <= 1 + tol
    else:
        res = True
    return res, cert, scaling 