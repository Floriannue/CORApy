"""
split - Splits a zonotope bundle into two zonotope bundles.

This is done for one or every generator resulting in n possible splits where n is
the system dimension; it is also possible to use a splitting hyperplane.

Syntax:
    Zsplit = split(zB, ...)

Inputs:
    zB - zonoBundle object
    N/hyperplane - splitting dimension or splitting hyperplane

Outputs:
    Zsplit - one or many zonotope bundle pairs

Example:
    Z1 = zonotope([1;1;-1], [1 1 -1; -1 1 0; -2 0 1]);
    Z2 = Z1 + [2;1;-1];
    zB = zonoBundle({Z1,Z2});
    Zsplit = split(zB,2);

    figure; hold on;
    plot(zB);
    plot(Zsplit{1},[1,2],'LineStyle','--');
    plot(Zsplit{2},[1,2],'LineStyle','--');

Other m-files required: reduce
Subfunctions: none
MAT-files required: none

See also: none

Authors:       Matthias Althoff (MATLAB)
               Automatic python translation: Florian Nüssel BA 2025
Written:       03-February-2011 (MATLAB)
Last update:   23-August-2013 (MATLAB)
               25-January-2016 (MATLAB)
               25-July-2016 (intervalhull replaced by interval, MATLAB)
Last revision: ---
"""

from typing import List, Union, Any
import numpy as np

from cora_python.contSet.interval import Interval
from cora_python.contSet.interval.infimum import infimum
from cora_python.contSet.interval.supremum import supremum
from cora_python.contSet.interval.zonotope import zonotope
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def split(zB, *args) -> Union[List[Any], List[List[Any]]]:
    """
    Splits a zonotope bundle into two zonotope bundles.
    """
    # split all dimensions
    if len(args) == 0:
        IH = zB.interval()
        left_limit = infimum(IH)
        right_limit = supremum(IH)

        Zsplit = []
        for dim in range(len(left_limit)):
            Zsplit.append(_aux_split_one_dim(zB, left_limit, right_limit, dim))
        return Zsplit

    elif len(args) == 1:
        arg = args[0]
        # split given dimension
        if isinstance(arg, (int, np.integer)):
            IH = zB.interval()
            left_limit = infimum(IH)
            right_limit = supremum(IH)
            return _aux_split_one_dim(zB, left_limit, right_limit, int(arg))

        # split using a halfspace
        if _is_halfspace_polytope(arg):
            h = arg
            rot_mat = _aux_rotation_matrix_halfspace(h)
            inv_rot = rot_mat.T

            zB_rot = _apply_matrix_to_bundle(inv_rot, zB)
            IH = zB_rot.interval()
            intervals = np.hstack([infimum(IH), supremum(IH)])

            h_rot = inv_rot @ h
            h_rot_normal = h_rot.A[0, :].reshape(-1, 1)
            h_rot_d = float(h_rot.b[0, 0])

            new_interval_1 = intervals.copy()
            new_interval_2 = intervals.copy()
            new_interval_1[0, 1] = h_rot_d
            new_interval_2[0, 0] = h_rot_d

            int_hull_1 = Interval(new_interval_1[:, 0:1], new_interval_1[:, 1:2])
            int_hull_2 = Interval(new_interval_2[:, 0:1], new_interval_2[:, 1:2])

            Znew_1 = rot_mat @ zonotope(int_hull_1)
            Znew_2 = rot_mat @ zonotope(int_hull_2)

            Zsplit = [
                zB.and_(Znew_1, 'exact'),
                zB.and_(Znew_2, 'exact')
            ]
            return Zsplit

        raise CORAerror('CORA:wrongInputInConstructor', 'Unsupported split argument.')

    elif len(args) == 2:
        if isinstance(args[1], str) and args[1] == 'bundle':
            dir_vec = np.asarray(args[0]).reshape(-1, 1)
            return _aux_direction_split_bundle(zB, dir_vec)

        raise CORAerror('CORA:wrongInputInConstructor', 'Unsupported split arguments.')

    raise CORAerror('CORA:wrongInputInConstructor', 'Too many input arguments.')


def _aux_split_one_dim(zbundle, lb, ub, dim: int):
    """
    Split limits for a given dimension.
    """
    left_limit_mod = lb.copy()
    right_limit_mod = ub.copy()
    left_limit_mod[dim, 0] = 0.5 * (lb[dim, 0] + ub[dim, 0])
    right_limit_mod[dim, 0] = 0.5 * (lb[dim, 0] + ub[dim, 0])

    Zleft = zonotope(Interval(lb, right_limit_mod))
    Zright = zonotope(Interval(left_limit_mod, ub))

    return [zbundle.and_(Zleft, 'exact'), zbundle.and_(Zright, 'exact')]


def _aux_direction_split_bundle(zB, dir_vec: np.ndarray):
    """
    Split halfway in a direction using a zonotope bundle.
    """
    n = dir_vec.shape[0]
    new_dir = np.zeros((n, 1))
    new_dir[0, 0] = 1.0

    rot_mat = _aux_rotation_matrix_dir(dir_vec, new_dir)
    zB_rot = _apply_matrix_to_bundle(rot_mat, zB)
    IH = zB_rot.interval()
    intervals = np.hstack([infimum(IH), supremum(IH)])

    intervals_1 = intervals.copy()
    intervals_2 = intervals.copy()
    intervals_1[0, 1] = 0.5 * (intervals_1[0, 0] + intervals_1[0, 1])
    intervals_2[0, 0] = 0.5 * (intervals_2[0, 0] + intervals_2[0, 1])

    IH1 = Interval(intervals_1[:, 0:1], intervals_1[:, 1:2])
    IH2 = Interval(intervals_2[:, 0:1], intervals_2[:, 1:2])

    Z1 = [zB.Z[0], rot_mat.T @ zonotope(IH1)]
    Z2 = [zB.Z[0], rot_mat.T @ zonotope(IH2)]

    from cora_python.contSet.zonoBundle import ZonoBundle
    return [ZonoBundle(Z1), ZonoBundle(Z2)]


def _aux_rotation_matrix_halfspace(h):
    """
    Rotation matrix for a halfspace polytope.
    """
    c = h.A[0, :].reshape(-1, 1)
    n = c.shape[0]
    new_dir = np.zeros((n, 1))
    new_dir[0, 0] = 1.0
    return _aux_rotation_matrix_dir(c, new_dir)


def _aux_rotation_matrix_dir(dir_vec: np.ndarray, new_dir: np.ndarray) -> np.ndarray:
    """
    Rotation matrix between two directions.
    """
    n = dir_vec.shape[0]
    if not np.isclose(float(dir_vec.T @ new_dir), 1.0) and not np.isclose(float(dir_vec.T @ new_dir), -1.0):
        # Normalize vectors
        n_vec = dir_vec / np.linalg.norm(dir_vec)
        new_dir = new_dir / np.linalg.norm(new_dir)

        B = np.zeros((n, n))
        B[:, 0:1] = n_vec

        ind_vec = new_dir - (new_dir.T @ n_vec) * n_vec
        B[:, 1:2] = ind_vec / np.linalg.norm(ind_vec)

        if n > 2:
            from scipy.linalg import null_space
            B[:, 2:] = null_space(B[:, :2].T)

        angle = np.arccos(float(new_dir.T @ n_vec))
        R = np.eye(n)
        R[0, 0] = np.cos(angle)
        R[0, 1] = -np.sin(angle)
        R[1, 0] = np.sin(angle)
        R[1, 1] = np.cos(angle)

        rot_mat = B @ R @ np.linalg.inv(B)
    else:
        if np.isclose(float(dir_vec.T @ new_dir), 1.0):
            rot_mat = np.eye(n)
        else:
            rot_mat = -np.eye(n)

    return rot_mat


def _apply_matrix_to_bundle(matrix: np.ndarray, zB):
    from cora_python.contSet.zonoBundle import ZonoBundle
    Z = [matrix @ Zi for Zi in zB.Z]
    return ZonoBundle(Z)


def _is_halfspace_polytope(obj) -> bool:
    if obj.__class__.__name__ != 'Polytope':
        return False
    try:
        return obj.representsa_('halfspace', 1e-12)
    except Exception:
        return False
