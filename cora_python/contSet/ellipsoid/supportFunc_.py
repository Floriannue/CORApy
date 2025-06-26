import numpy as np
from typing import Union, Tuple

# Make sure to create this interval class if it does not exist.
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def supportFunc_(E: Ellipsoid,
                 dir: np.ndarray,
                 type: str) -> Union[float, Interval, Tuple[Union[float, Interval], np.ndarray]]:
    """
    supportFunc_ - Calculate the upper or lower bound of an ellipsoid along a
    certain direction (see Def. 2.1.2 in [1])

    Syntax:
        val = supportFunc_(E,dir)
        [val,x] = supportFunc_(E,dir,type)

    Inputs:
        E - ellipsoid object
        dir - direction for which the bounds are calculated (vector of size
              (n,1) )
        type - upper or lower bound ('lower',upper','range')

    Outputs:
        val - bound of the ellipsoid in the specified direction
        x - point for which holds: dir'*x=val

    References:
        [1] A. Kurzhanski et al. "Ellipsoidal Toolbox Manual", 2006
            https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf
    """

    if E.representsa_('point', E.TOL):
        val = dir.T @ E.q
        return val[0, 0], E.q

    # Pre-calculate common term
    dir_q_dir = dir.T @ E.Q @ dir
    
    # Handle case where direction has no extent in the ellipsoid space
    if dir_q_dir <= E.TOL:
        val = dir.T @ E.q
        return val[0, 0], E.q

    sqrt_val = np.sqrt(dir_q_dir)
    
    if type == 'upper':
        val = dir.T @ E.q + sqrt_val
        x = E.q + (E.Q @ dir) / sqrt_val
        return val[0, 0], x
    elif type == 'lower':
        val = dir.T @ E.q - sqrt_val
        x = E.q - (E.Q @ dir) / sqrt_val
        return val[0, 0], x
    elif type == 'range':
        lower_bound = dir.T @ E.q - sqrt_val
        upper_bound = dir.T @ E.q + sqrt_val
        val = Interval(lower_bound[0, 0], upper_bound[0, 0])
        
        x_lower = E.q - (E.Q @ dir) / sqrt_val
        x_upper = E.q + (E.Q @ dir) / sqrt_val
        x = np.hstack((x_lower, x_upper))
        return val, x
    else:
        raise ValueError(f"Unknown type '{type}'") 