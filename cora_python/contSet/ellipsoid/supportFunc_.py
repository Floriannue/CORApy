import numpy as np
from typing import Union, Tuple
from cora_python.contSet.interval.interval import Interval
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def supportFunc_(E: Ellipsoid,
                 direction: np.ndarray, 
                 type_: str = 'upper',
                 *args) -> Union[float, Interval, Tuple[Union[float, Interval], np.ndarray]]:
    """
    supportFunc_ - Calculate the upper or lower bound of an ellipsoid along a
    certain direction (see Def. 2.1.2 in [1])

    Syntax:
        val = supportFunc_(E,dir)
        [val,x] = supportFunc_(E,dir,type)

    Inputs:
        E - ellipsoid object
        direction - direction for which the bounds are calculated (vector of size
                  (n,1) )
        type_ - upper or lower bound ('lower',upper','range')

    Outputs:
        val - bound of the ellipsoid in the specified direction
        x - point for which holds: dir'*x=val

    Example: 
        E = ellipsoid([5 7;7 13],[1;2]);
        dir = [1;1];

        [val,x] = supportFunc(E,dir);
      
        figure; hold on; box on;
        plot(E,[1,2],'b');
        plot(polytope([],[],dir',val),[1,2],'g');
        plot(x(1),x(2),'.r','MarkerSize',20);

    References: 
      [1] A. Kurzhanski et al. "Ellipsoidal Toolbox Manual", 2006
          https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: contSet/supportFunc, zonotope/supportFunc_

    Authors:       Victor Gassmann
    Written:       20-November-2019
    Last update:   12-March-2021
                    27-July-2021 (fixed degenerate case)
                    04-July-2022 (VG, class array case)
                    29-March-2023 (VG, changed to explicit comp of x vector)
    Last revision: 27-March-2023 (MW, rename supportFunc_)
    Automatic python translation: Florian Nüssel BA 2025
    """
    
    # Ensure direction is a column vector
    direction = np.asarray(direction)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)
    
    # Check if ellipsoid represents an empty set
    if E.representsa_('emptySet', E.TOL):
        if type_ == 'upper':
            return -np.inf, np.full((E.dim(), 1), np.nan)
        elif type_ == 'lower':
            return np.inf, np.full((E.dim(), 1), np.nan)
        elif type_ == 'range':
            from cora_python.contSet.interval.interval import Interval
            return Interval(-np.inf, np.inf), np.full((E.dim(), 2), np.nan)
    
    # Check if ellipsoid represents a point
    if E.representsa_('point', np.finfo(float).eps):
        val = float(direction.T @ E.q)  # Convert to scalar
        x = E.q
        return val, x
    
    # Normalize handling to ensure correct sign in expected tests
    quad = float(direction.T @ E.Q @ direction)
    if quad < 0 and abs(quad) < 1e-14:
        quad = 0.0
    rad = np.sqrt(quad)
    if type_ == 'upper':
        val = float(direction.T @ E.q + rad)
        x = E.q + (E.Q @ direction) / (rad if rad != 0 else 1.0)
    elif type_ == 'lower':
        val = float(direction.T @ E.q - rad)
        x = E.q - (E.Q @ direction) / (rad if rad != 0 else 1.0)
    elif type_ == 'range':
        from cora_python.contSet.interval.interval import Interval
        lower_val = float(direction.T @ E.q - np.sqrt(direction.T @ E.Q @ direction))  # Convert to scalar
        upper_val = float(direction.T @ E.q + np.sqrt(direction.T @ E.Q @ direction))  # Convert to scalar
        val = Interval(lower_val, upper_val)
        # Return [x_upper, x_lower] to match test that stacks x_upper then x_lower
        x_upper = E.q + (E.Q @ direction) / (rad if rad != 0 else 1.0)
        x_lower = E.q - (E.Q @ direction) / (rad if rad != 0 else 1.0)
        x = np.column_stack([x_upper, x_lower])
    else:
        raise ValueError(f"Invalid type '{type_}'. Use 'lower', 'upper', or 'range'.")
    
    return val, x 