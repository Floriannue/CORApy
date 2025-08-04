
import numpy as np
import scipy.linalg
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.g.functions.matlab.validate.preprocessing.setDefaultValues import setDefaultValues
from cora_python.g.functions.matlab.validate.check.inputArgsCheck import inputArgsCheck

# Import necessary methods that might be called on Ellipsoid objects
# These should ideally be imported at the top level or passed as arguments
# to avoid circular dependencies if they themselves call randPoint_
# For now, assuming they are properly attached to Ellipsoid class
# from .representsa_ import representsa_
# from .rank import rank
# from .dim import dim
# from .project import project

# Removed debug logging utility

def randPoint_(E, N=1, type='standard', *varargin):
    """
    randPoint_ - samples a random point from within an ellipsoid

    Syntax:
       p = randPoint_(E)
       p = randPoint_(E,N)
       p = randPoint_(E,N,type)

    Inputs:
       E - ellipsoid object
       N - number of random points
       type - type of the random point ('standard' or 'extreme')

    Outputs:
       p - random point in R^n

    Example:
       E = Ellipsoid(np.array([[9.3, -0.6, 1.9], [-0.6, 4.7, 2.5], [1.9, 2.5, 4.2]]))
       p = randPoint_(E)

    Other m-files required: none
    Subfunctions: none
    MAT-files required: none

    See also: contSet/randPoint, interval/randPoint_

    Authors:       Victor Gassmann, Mark Wetzlinger, Maximilian Perschl, Adrian Kulmburg (MATLAB)
                   Automatic python translation: Florian Nüssel BA 2025
    Written:       18-March-2021 (MATLAB)
    Last update:   25-June-2021 (MP, add type gaussian)
                   19-August-2022 (MW, integrate standardized pre-processing)
                   12-March-2024 (AK, made the distribution truly uniform and random)
    Last revision: 28-March-2023 (MW, rename randPoint_)
    """

    # Default values for N and type (handled by setDefaultValues in contSet/randPoint.m mostly)
    # Here, we directly use N=1, type='standard' as defaults if not provided.
    
    # 'all' vertices not supported
    if isinstance(N, str) and N == 'all':
        raise CORAerror('CORA:notSupported', 'Number of vertices \'all\' is not supported for class ellipsoid.')

    # Handle empty ellipsoid: return empty array with correct dimensions (n x 0)
    if E.isemptyobject():
        return np.empty((E.dim(), 0)) # Return n x 0 empty array

    # ellipsoid is just a point -> replicate center N times
    if E.representsa_('point', np.finfo(float).eps):
        return np.tile(E.q, (1, N))

    # save original center
    c_orig = E.q
    # shift ellipsoid to be centered at the origin
    E_shifted = Ellipsoid(E.Q, E.q - E.q) # Center at origin

    # compute rank and dimension
    r = E_shifted.rank()
    n = E_shifted.dim()

    # determine degeneracy: if so, project on proper subspace (via SVD)
    n_rem = n - r

    # Initialize for sampling in r-dim space and final n-dim space
    p_sampled_relative_to_origin = np.empty((r, N))
    final_p = np.empty((n, N))

    # Apply transformations to E_shifted to make it a unit ball in r dimensions
    T_svd = np.eye(n) # Default for non-degenerate case
    E_to_unit_ball_Q = E_shifted.Q # Default for non-degenerate case, will be modified
    E_to_unit_ball_q = E_shifted.q # Default for non-degenerate case, will be modified

    if n_rem > 0: # If degenerate
        T_svd, _, _ = np.linalg.svd(E_shifted.Q)

        Q_rotated = T_svd.T @ E_shifted.Q @ T_svd

        try:
            q_rotated = T_svd.T @ E_shifted.q
        except ValueError as e:
            raise # Re-raise to ensure test fails

        E_to_unit_ball_Q = Q_rotated[:r, :r]
        E_to_unit_ball_q = q_rotated[:r] # Project q_rotated to r dimensions

    
    # Check for empty E_to_unit_ball_Q after projection for empty sets
    if E_to_unit_ball_Q.shape[0] == 0 or E_to_unit_ball_Q.size == 0: # Handle 0x0 matrix for empty sets
        # This should now be handled by the earlier E.isemptyobject() check
        pass # Removed log_debug and return

    # Ensure sampling_transform_matrix is calculated on a non-empty, non-singular matrix
    # The previous if condition for E_to_unit_ball_Q.shape[0] > 0 is now implicitly handled by the outer isemptyobject() check
    # and the E.representsa_('point') check. So we can simplify the conditional logic for det check.
    if np.linalg.det(E_to_unit_ball_Q) < np.finfo(float).eps * 1e-2:
        try:
            sampling_transform_matrix = scipy.linalg.sqrtm(E_to_unit_ball_Q)
        except np.linalg.LinAlgError:
            sampling_transform_matrix = scipy.linalg.pinv(E_to_unit_ball_Q)
    else:
        sampling_transform_matrix = scipy.linalg.sqrtm(E_to_unit_ball_Q)


    if type == 'standard' or type.startswith('uniform'):
        # Generate points uniformly distributed on the unit hypersphere in r dimensions
        X = np.random.randn(r, N)
        norm_X = np.linalg.norm(X, axis=0)
        # Avoid division by zero for points exactly at origin (unlikely but possible)
        pt = X / np.where(norm_X == 0, 1e-10, norm_X)
        
        # Uniform radius: MATLAB uses (1/dim(E)) power, where dim(E) is the original dimension (n)
        random_radii = np.random.rand(1, N)**(1.0 / n)
        pt = random_radii * pt
        
        p_sampled_relative_to_origin[:] = sampling_transform_matrix @ pt

    elif type == 'extreme':
        # Generate points uniformly distributed on the unit hypersphere in r dimensions
        X = np.random.randn(r, N)
        norm_X = np.linalg.norm(X, axis=0)
        pt = X / np.where(norm_X == 0, 1e-10, norm_X)
        
        p_sampled_relative_to_origin[:] = sampling_transform_matrix @ pt

    else:
        raise CORAerror('CORA:notSupported', f'Sampling type \'{type}\' is not yet implemented for ellipsoid/randPoint_.')

    # Stack again with zeros for n_rem dimensions, and back-transform by T_svd if degenerate
    if n_rem > 0:
        intermediate_points = np.vstack((p_sampled_relative_to_origin, np.zeros((n_rem, N))))
        final_p[:] = T_svd @ intermediate_points
    else:
        final_p[:] = p_sampled_relative_to_origin

    # Add original center back (applied once for both cases)
    final_p = final_p + c_orig
    
    return final_p 