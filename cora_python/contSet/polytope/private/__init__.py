# This file is part of the CORA project.
# Copyright (c) 2024, System Control and Robotics Group, TU Graz.
# All rights reserved.

"""
private - private helper functions for polytope

This module contains internal helper functions for polytope operations
that should not be part of the public API.

Authors: Python translation by AI Assistant
Written: 2025
"""

from .priv_supportFunc import priv_supportFunc
from .priv_normalize_constraints import priv_normalize_constraints
from .priv_box_H import priv_box_H
from .priv_box_V import priv_box_V

from .priv_plus_minus_vector import priv_plus_minus_vector
from .priv_equality_to_inequality import priv_equality_to_inequality
from .priv_compact_all import priv_compact_all
from .priv_compact_zeros import priv_compact_zeros
from .priv_compact_toEquality import priv_compact_toEquality
from .priv_compact_alignedEq import priv_compact_alignedEq
from .priv_compact_alignedIneq import priv_compact_alignedIneq
from .priv_compact_1D import priv_compact_1D
from .priv_compact_2D import priv_compact_2D
from .priv_compact_nD import priv_compact_nD
from .priv_representsa_emptySet import priv_representsa_emptySet
from .priv_vertices_1D import priv_vertices_1D
from .priv_conZonotope_supportFunc import priv_conZonotope_supportFunc
from .priv_conZonotope_vertices import priv_conZonotope_vertices
from .priv_copyProperties import priv_copyProperties
from .priv_feasiblePoint import priv_feasiblePoint
from .priv_isFullDim_V import priv_isFullDim_V
from .priv_normalizeConstraints import priv_normalizeConstraints
from .priv_equalityToInequality import priv_equalityToInequality
from .priv_cartprod import priv_cartprod
from .priv_compactv import priv_compactv
from .priv_minkdiff import priv_minkdiff

__all__ = [
    "priv_supportFunc",
    "priv_normalize_constraints",
    "priv_box_H",
    "priv_box_V",
    "priv_V_to_H",
    "priv_plus_minus_vector",
    "priv_equality_to_inequality",
    "priv_compact_all",
    "priv_compact_zeros",
    "priv_compact_toEquality",
    "priv_compact_alignedEq",
    "priv_compact_alignedIneq",
    "priv_compact_1D",
    "priv_compact_2D",
    "priv_compact_nD",
    "priv_representsa_emptySet",
    "priv_vertices_1D",
    "priv_conZonotope_supportFunc",
    "priv_conZonotope_vertices",
    "priv_copyProperties",
    "priv_feasiblePoint",
    "priv_isFullDim_V",
    "priv_normalizeConstraints",
    "priv_equalityToInequality",
    "priv_cartprod",
    "priv_compactv",
    "priv_minkdiff",
] 