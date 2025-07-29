"""
zonotope - converts an ellipsoid to a zonotope

Syntax:
    Z = zonotope(E)
    Z = zonotope(E, mode)
    Z = zonotope(E, mode, nrGen)

Inputs:
    E - ellipsoid object
    mode - (optional) Specifies whether function uses a lower bound on the
           minimum zonotope norm or the exact value:
           * 'outer:box':      overapprox. parallelotope using
                               priv_encParallelotope
           * 'outer:norm':     uses priv_encZonotope with exact norm value
           * 'outer:norm_bnd': not implemented yet (throws error)
           * 'inner:box':      inner approx. parallelotope using
                               priv_inscParallelotope
           * 'inner:norm':     uses priv_inscZonotope with exact norm value
           * 'inner:norm_bnd': uses priv_inscZonotope with an bound on the
                               norm value
           * default:          same as 'outer:box'
    nrGen - (optional) number of generators

Outputs:
    Z - zonotope object

Example:
    E = Ellipsoid([[3, -1], [-1, 1]], [[1], [0]])
    Z_enc = zonotope(E, 'outer:norm', 10)
    Z_insc = zonotope(E, 'inner:norm', 10)
    Z_box = zonotope(E)

References:
    [1] V. Gaßmann, M. Althoff. "Scalable Zonotope-Ellipsoid Conversions
        using the Euclidean Zonotope Norm", 2020

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: priv_encZonotope, priv_encParallelotope, priv_inscZonotope,
    priv_inscParallelotope

Authors:       Victor Gassmann (MATLAB)
               Python translation by AI Assistant
Written:       11-October-2019 (MATLAB)
Last update:   08-June-2021 (moved handling of degenerate case here)
               04-July-2022 (VG, class array cases)
Python translation: 2025
"""

import numpy as np
from typing import TYPE_CHECKING
from .ellipsoid import Ellipsoid

if TYPE_CHECKING:
    from cora_python.contSet.zonotope.zonotope import Zonotope

from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from cora_python.contSet.ellipsoid.private.priv_encParallelotope import priv_encParallelotope
from cora_python.contSet.ellipsoid.private.priv_inscParallelotope import priv_inscParallelotope
from cora_python.contSet.ellipsoid.private.priv_encZonotope import priv_encZonotope
from cora_python.contSet.ellipsoid.private.priv_inscZonotope import priv_inscZonotope


def zonotope(E: 'Ellipsoid', mode: str = 'outer:box', nrGen: int = None) -> 'Zonotope':
    """
    Converts an ellipsoid to a zonotope

    Args:
        E: ellipsoid object
        mode: conversion mode (default: 'outer:box')
        nrGen: number of generators (default: dimension of E)

    Returns:
        Z: zonotope object
    """
    from cora_python.contSet.zonotope.zonotope import Zonotope
    # Set default values
    if nrGen is None:
        nrGen = E.dim()

    # Input validation - simplified to avoid class lookup issues
    if not isinstance(E, Ellipsoid):
        raise CORAerror('CORA:wrongValue', '1st', 'expected ellipsoid object')

    if mode not in ['outer:box', 'outer:norm', 'outer:norm_bnd',
                    'inner:box', 'inner:norm', 'inner:norm_bnd']:
        raise CORAerror('CORA:wrongValue', '2nd', 'invalid mode')

    if not isinstance(nrGen, int) or nrGen <= 0:
        raise CORAerror('CORA:wrongValue', '3rd', 'nrGen must be positive integer')

    # Handle empty case
    if E.representsa_('emptySet'):
        return Zonotope.empty(E.dim())

    # Compute rank and dimension of ellipsoid
    rankE = E.rank()
    c = E.center()
    isDeg = rankE != E.dim()

    # Handle degenerate case
    if isDeg:
        # Ellipsoid is just a point
        if rankE == 0:
            return Zonotope(c)

        # Compute SVD to find U matrix transforming the shape matrix to a
        # diagonal matrix (to isolate degenerate dimensions)
        U, S, Vt = np.linalg.svd(E.Q)
        Qt = np.diag(S[:rankE])

        # Construct non-degenerate ellipsoid
        E_nonDeg = Ellipsoid(Qt)

        # Construct revert transformation matrix
        T = U[:, :rankE]
    else:
        E_nonDeg = E
        T = None

    # Convert based on mode
    if mode == 'outer:box':
        Z = priv_encParallelotope(E_nonDeg)
    elif mode == 'outer:norm':
        Z = priv_encZonotope(E_nonDeg, nrGen)
    elif mode == 'outer:norm_bnd':
        raise CORAerror('CORA:notSupported', "mode = 'outer:norm_bnd'")
    elif mode == 'inner:box':
        Z = priv_inscParallelotope(E_nonDeg)
    elif mode == 'inner:norm':
        Z = priv_inscZonotope(E_nonDeg, nrGen, 'exact')
    elif mode == 'inner:norm_bnd':
        Z = priv_inscZonotope(E_nonDeg, nrGen, 'ub_convex')

    # In degenerate case, lift lower-dimensional non-degenerate ellipsoid
    if isDeg:
        Z = T @ Z + c

    return Z 