"""
Private function for vertex enumeration-based zonotope containment check.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid


def priv_venumZonotope(E: 'Ellipsoid', Z, tol: float, scalingToggle: bool) -> Tuple[bool, bool, float]:
    """
    Checks whether an ellipsoid contains a zonotope using vertex enumeration.
    
    This implements the search-based algorithm described in:
    [1] Kulmburg A, Brkan I, Althoff A, "Search-based and Stochastic
        Solutions to the Zonotope and Ellipsotope Containment Problems",
        ECC 2024
    
    Syntax:
        res, cert, scaling = priv_venumZonotope(E, Z, tol, scalingToggle)
    
    Args:
        E: ellipsoid object (circumbody)
        Z: zonotope object (inbody)
        tol: tolerance
        scalingToggle: boolean allowing to choose whether scaling should be computed
                      (which takes more time in general)
    
    Returns:
        res: true/false
        cert: returns true iff the result could be verified. For example, if res=false 
              and cert=true, Z is guaranteed to not be contained in E, whereas if 
              res=false and cert=false, nothing can be deduced (Z could still be
              contained in E). If res=true, then cert=true.
        scaling: returns the smallest number 'scaling', such that
                scaling*(E - E.center) + E.center contains Z.
    """
    
    scaling = 0.0
    cert = True  # Since venumSearch is exact, cert is always true
    
    # Let c1, c2 be the centers of Z, E. We prepare the norm-function,
    # returning the norm of v-c2 w.r.t. the E-norm, where v is a given vertex
    # of Z. Since v = c1 +- g_1 +- ... +- g_m, where g_i are the generators of
    # Z, the norm of v-c2 is the same as the norm of G*nu + c1-c2, where
    # G is the generator matrix of Z, nu = [+-1;...;+-1]. However,
    # here, we strategically do NOT include the center just yet.
    
    G = Z.G
    # Instead, we add the combined centers to the generators of the inbody:
    G = np.column_stack([Z.c - E.q, G])
    
    G_size = G.shape[1]
    
    # We now compute the norm of each vector in G, and sort the vectors in
    # descending order; We keep the computed values for later use
    generator_norms = np.zeros(G_size)
    for i in range(G_size):
        generator_norms[i] = E.ellipsoidNorm(G[:, i])
    
    # Sort generators by norm in descending order
    indices = np.argsort(generator_norms)[::-1]  # descending order
    generator_norms = generator_norms[indices]
    G = G[:, indices]
    
    # We also setup a heuristic based on the triangle inequality, that allows
    # us to reject certain nodes that cannot possibly lead to a good result
    def heuristic(nu, value):
        # Let me explain this line: it looks at all the coordinates of nu that are
        # zero, and for those it adds the corresponding values of generator_norms.
        # Then, it adds to that 'value', which is the current value of the node. By
        # the triangle inequality, all nodes in the subtree of the node in question
        # must have a value that is <= heuristic, which may be used to remove
        # certain nodes of whose subtree cannot possibly lead to a high enough
        # value to disprove containment.
        return value + np.sum(generator_norms * np.abs(np.abs(nu) - 1))
    
    # We now setup the queue of nodes to go through. We basically need to check
    # all points of the form nu_1*h_1 + ... + nu_m*h_m, where h_i are the
    # columns of H and nu_i are +1 or -1. These choices generate a binary tree.
    # We then have a queue to save the nu's we still have to check
    queue_nu = []
    # We also need to save the values of the nodes, which we put into a
    # separate queue (but the two queues should be seen as linked)
    queue_values = []
    
    # We start at the top:
    queue_nu.append(np.zeros(G_size))
    queue_values.append(0.0)
    
    # We can now go ahead with the iteration:
    # While we still have to check a point, keep the iteration going:
    while queue_nu:
        # pop the last values of the queue:
        current_nu = queue_nu.pop()
        current_value = queue_values.pop()
        
        # find first zero of current nu
        zero_indices = np.where(current_nu == 0)[0]
        if len(zero_indices) == 0:
            # If every component of current_nu is nonzero, we have hit a leaf
            # of the tree, and so we can stop there
            continue
        
        i = zero_indices[0]  # first zero index
        
        # We now compute the children of the current_nu node
        child_positive = current_nu.copy()
        child_negative = current_nu.copy()
        child_positive[i] = 1
        child_negative[i] = -1
        
        # We compute the values of both children...
        child_positive_value = E.ellipsoidNorm(G @ child_positive)
        child_negative_value = E.ellipsoidNorm(G @ child_negative)
        
        # ... and order them, before adding them to the queue
        if child_positive_value <= child_negative_value:
            queue_nu.extend([child_positive, child_negative])
            queue_values.extend([child_positive_value, child_negative_value])
        else:
            queue_nu.extend([child_negative, child_positive])
            queue_values.extend([child_negative_value, child_positive_value])
        
        # If any of the children has a value that is >1+tol, we have disproven
        # containment. By the ordering above, we just need to check the last
        # element of the queue
        if not scalingToggle and queue_values[-1] > 1 + tol:
            res = False
            return res, cert, scaling
        else:
            scaling = max(scaling, queue_values[-1])
        
        # We now need to perform some 'cleaning' of the queue. First of all, we
        # need to compute the heuristic value of the sub-trees of both children
        # This part changes a bit depending on whether we need the scaling
        # factor, or whether we are only interested in a containment check
        if scalingToggle:
            # If any of the two (or both) have a heuristic <= scaling (which is
            # the currently maximal value we have found so far), we don't need
            # to investigate the subtrees, they have no chance of beating the
            # maximum.
            if heuristic(queue_nu[-2], queue_values[-2]) <= scaling:
                # So if the 'weakest' of the two nodes can be deleted from the
                # queue, do that
                queue_nu.pop(-2)
                queue_values.pop(-2)
                # Now, the question is whether the 'strongest' of the two also
                # needs to be deleted
                if heuristic(queue_nu[-1], queue_values[-1]) <= scaling:
                    queue_nu.pop()
                    queue_values.pop()
                
                # If something has been deleted, we don't need to sort anything in
                # the next step
                continue
            # No other situation can happen; it cannot happen, that the 'strongest'
            # node can be deleted, while the 'weakest' one doesn't; this follows
            # from the definition of the heuristic
        else:
            # If any of the two (or both) have a heuristic <= 1+tol, there is no
            # need to keep them
            if heuristic(queue_nu[-2], queue_values[-2]) <= 1 + tol:
                # So if the 'weakest' of the two nodes can be deleted from the
                # queue, do that
                queue_nu.pop(-2)
                queue_values.pop(-2)
                # Now, the question is whether the 'strongest' of the two also
                # needs to be deleted
                if heuristic(queue_nu[-1], queue_values[-1]) <= 1 + tol:
                    queue_nu.pop()
                    queue_values.pop()
                
                # If something has been deleted, we don't need to sort anything in
                # the next step
                continue
            # No other situation can happen; it cannot happen, that the 'strongest'
            # node can be deleted, while the 'weakest' one doesn't; this follows
            # from the definition of the heuristic
        
        # Finally, we need to sort the 'weakest' of the two children, so that
        # the queue remains sorted (by the triangle inequality, the 'strongest'
        # child always has a node value that is at least as large as that of
        # the parent, so it may remain the last element of the queue)
        if len(queue_values) >= 2:
            # Find position to insert the weakest child to maintain sorted order
            weakest_value = queue_values[-2]
            weakest_nu = queue_nu[-2]
            
            # Remove the weakest child from the end
            queue_nu.pop(-2)
            queue_values.pop(-2)
            
            # Find insertion point
            j = 0
            for k in range(len(queue_values) - 1):  # -1 because we need to keep the strongest at the end
                if queue_values[k] >= weakest_value:
                    j = k
                    break
            else:
                j = len(queue_values) - 1
            
            # Insert at the correct position
            queue_nu.insert(j, weakest_nu)
            queue_values.insert(j, weakest_value)
    
    # If we could not find a node that disproves containment, there isn't any;
    # so, the zonotope is contained in the other one
    if scalingToggle:
        res = scaling <= 1 + tol
    else:
        res = True
    
    return res, cert, scaling 