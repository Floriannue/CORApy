"""
contains_ - determines if a zonotope contains a set or a point

Syntax:
   [res,cert,scaling] = contains_(Z,S,method,tol,maxEval,certToggle,scalingToggle)

Inputs:
   Z - zonotope object
   S - contSet object or single point or matrix of points
   method - method used for the containment check.
      The available options are:
          - 'exact': Checks for containment by using either
              'exact:venum' or 'exact:polymax', depending on the number
              of generators of Z and the object S.
          - 'approx': Checks for containment using 'approx:st' (see  
              below) if S is a zonotope, or any approximative method 
              available otherwise.
          - 'exact:venum': Checks for containment by enumerating all 
              vertices of S (see Algorithm 1 in [2]).
          - 'exact:polymax': Checks for containment by maximizing the
              polyhedral norm w.r.t. Z over S (see Algorithm 2 in [2]).
          - 'approx:st': Solves the containment problem using the
              approximative method from [1]. If a solution using
              'approx:st' returns that Z1 is contained in Z2, then this
              is guaranteed to be the case. The runtime is polynomial
              w.r.t. all inputs.
          - 'approx:stDual': Solves the containment problem using the 
              dual approximative method from [3]. Returns the same values
              for res and scaling as 'approx:st', but cert can be more
              precise.
         For the next methods, note that if both certToggle and
         scalingToggle are set to 'false', then res will be set to
         'false' automatically, and the algorithms will not be executed.
         This is because stochastic/optimization-based algorithms can not
         confirm containment, so res = true can never happen. However, if
         maxEval is set high enough, and res = false but cert = false,
         one might conclude that with good probability, containment
         holds.
          - 'opt': Solves the containment problem via optimization
              (see [2]) using the subroutine ga. If a solution
              using 'opt' returns that Z1 is not contained in Z2, then
              this is guaranteed to be the case. The runtime is
              polynomial w.r.t. maxEval and the other inputs.
          - 'sampling:primal': Solves the containment stochastically,
              using the Shenmaier vertex sampling from [4].
          - 'sampling:dual': Solves the containment stochastically, using
              the Shenmaier halfspace sampling from [4].
      The methods 'exact:venum' and 'exact:polymax' are only available if
      S is a zonotope or a point/point cloud, and 'opt', 'approx:st', and
      'approx:stDual' are only available if S is a zonotope.
   tol - tolerance for the containment check; the higher the
      tolerance, the more likely it is that points near the boundary of Z
      will be detected as lying in Z, which can be useful to counteract
      errors originating from floating point errors.
   maxEval - only, if 'opt', 'sampling:primal', or 'sampling:dual' is
      used: Number of maximal function evaluations.
   certToggle - if set to 'true', cert will be computed (see below).
   scalingToggle - if set to 'true', scaling will be computed (see
      below).

Outputs:
   res - true/false
   cert - returns true iff the result of res could be
          verified. For example, if res=false and cert=true, S is
          guaranteed to not be contained in Z, whereas if res=false and
          cert=false, nothing can be deduced (S could still be
          contained in Z).
          If res=true, then cert=true.
          Note that computing this certification may marginally increase
          the runtime.
   scaling - returns the smallest number 'scaling', such that
          scaling*(Z - center(Z)) + center(Z) contains S.
          For the methods 'approx' and 'approx:st' this is an upper
          bound, for 'opt', 'sampling:primal' and 'sampling:dual', this
          number is a lower bound.
          Note that computing this scaling factor may significantly 
          increase the runtime.

Note: For low dimensions or number of generators, and if S is a point
cloud with a very large number of points, it may be beneficial to convert
the zonotope to a polytope and call its containment operation

Example: 
   Z1 = Zonotope(np.array([[0.5, 2, 3, 0], [0.5, 2, 0, 3]]))
   Z2 = Zonotope(np.array([[0, -1, 1, 0], [0, 1, 0, 1]]))
   Z3 = Z2 + np.array([[3], [0]])

   contains_(Z1, Z2)
   contains_(Z1, Z3)

References:
   [1] Sadraddini et. al: Linear Encodings for Polytope Containment
       Problems, CDC 2019
   [2] A. Kulmburg, M. Althoff.: On the co-NP-Completeness of the
       Zonotope Containment Problem, European Journal of Control 2021
   [3] A. Kulmburg, M. Althoff.: Hardness and Approximability of the
       Containment Problem for Zonotopes and Ellipsotopes
       (to appear)
   [4] Kulmburg A., Brkan I., Althoff M.,: Search-based and Stochastic
       Solutions to the Zonotope and Ellipsotope Containment Problems,
       ECC 2024

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: contSet/contains, interval/contains_, conZonotope/contains_

Authors: Matthias Althoff, Niklas Kochdumper, Adrian Kulmburg, Ivan Brkan (MATLAB)
         Python translation by AI Assistant
Written: 07-May-2007 (MATLAB)
Last update: 28-May-2025 (TL, quick check for representsa interval) (MATLAB)
         2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import numpy as np
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror
from .compact_ import compact_
from .representsa_ import representsa_
from .private.priv_zonotopeContainment_pointContainment import priv_zonotopeContainment_pointContainment
from .private.priv_zonotopeContainment_vertexEnumeration import priv_zonotopeContainment_vertexEnumeration
from .private.priv_zonotopeContainment_SadraddiniTedrake import priv_zonotopeContainment_SadraddiniTedrake
from .private.priv_zonotopeContainment_SadraddiniTedrakeDual import priv_zonotopeContainment_SadraddiniTedrakeDual
from .private.priv_zonotopeContainment_zonoSampling import priv_zonotopeContainment_zonoSampling
from .private.priv_zonotopeContainment_zonoSamplingDual import priv_zonotopeContainment_zonoSamplingDual
from .private.priv_zonotopeContainment_geneticMaximization import priv_zonotopeContainment_geneticMaximization
from .private.priv_zonotopeContainment_DIRECTMaximization import priv_zonotopeContainment_DIRECTMaximization
from .private.priv_zonotopeContainment_ellipsoidSampling import priv_zonotopeContainment_ellipsoidSampling
from .private.priv_zonotopeContainment_ellipsoidSamplingDual import priv_zonotopeContainment_ellipsoidSamplingDual

def contains_(Z, S, method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True, *varargin):
    def safe_representsa(obj, *args, **kwargs):
        """Helper function to safely handle representsa_ return values"""
        kwargs['return_set'] = True  # Always request the set to be returned
        result = representsa_(obj, *args, **kwargs)
        if isinstance(result, tuple):
            return result
        else:
            return result, None
    
    Z = compact_(Z)
    # Trivial case: Z is a point
    Z_isPoint, p = safe_representsa(Z, 'point', tol)
    if Z_isPoint:
        if isinstance(S, np.ndarray):
            if p is None or not isinstance(p, np.ndarray):
                raise ValueError("representsa_ did not return a valid numpy array for Z as point.")
            res = np.all(np.abs(S - p) <= tol)
            cert = True
            scaling = 0 if res else np.inf
            return res, cert, scaling
        else:
            S_isPoint, q = safe_representsa(S, 'point', tol)
            if S_isPoint:
                if p is None or q is None or not isinstance(p, np.ndarray) or not isinstance(q, np.ndarray):
                    raise ValueError("representsa_ did not return valid numpy arrays for Z or S as point.")
                res = np.all(np.abs(p - q) <= tol)
                cert = True
                scaling = 0 if res else np.inf
                return res, cert, scaling
            return False, True, np.inf
    # Trivial case: Z is interval
    Z_isInterval, I = safe_representsa(Z, 'interval', tol)
    if Z_isInterval and method.startswith(('exact', 'approx')):
        if I is None:
            raise ValueError("representsa_ did not return a valid interval object for Z.")
        if not hasattr(I, 'contains_'):
            raise ValueError("Invalid interval object returned - missing contains_ method.")
        # MATLAB: always delegate to interval.contains_ (matches MATLAB behavior exactly)
        # interval.contains_ converts to polytope, which now handles all ContSets including zonotope
        try:
            return I.contains_(S, method, tol, maxEval, certToggle, scalingToggle)
        except Exception as ME:
            # MATLAB: check if a specific method was used, try with base method
            if ':' in method:
                base_method = method.split(':')[0]
                try:
                    return I.contains_(S, base_method, tol, maxEval, certToggle, scalingToggle)
                except Exception:
                    raise ME
            else:
                raise ME
    # Buffer degenerate sets if not full-dimensional
    if hasattr(Z, 'isFullDim'):
        is_full_dim_result = Z.isFullDim(tol)
        # Handle tuple return (isFullDim, direction) or boolean
        if isinstance(is_full_dim_result, tuple):
            is_full_dim = is_full_dim_result[0]
        else:
            is_full_dim = is_full_dim_result
    else:
        is_full_dim = True
    if not is_full_dim:
        from ..interval import Interval
        # Robustly determine dimension
        if hasattr(Z, 'G') and isinstance(Z.G, np.ndarray):
            d = Z.G.shape[0]
        elif hasattr(Z, 'generators') and callable(Z.generators()):
            gens = Z.generators()
            if gens is not None and isinstance(gens, np.ndarray):
                d = gens.shape[0]
            else:
                raise ValueError("Z.generators() did not return a valid numpy array; cannot determine dimension.")
        else:
            raise ValueError("Z is missing required generator information (G or generators()).")
        I = tol * Interval(-np.ones((d, 1)), np.ones((d, 1)))
        Z = Z + I
    # Point or point cloud
    if isinstance(S, np.ndarray):
        if method not in ['exact', 'exact:venum', 'exact:polymax', 'approx']:
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
        # For point containment, 'approx' methods are not supported - use 'exact' instead
        if method == 'approx':
            method = 'exact'
        return priv_zonotopeContainment_pointContainment(Z, S, method, tol, scalingToggle)
    # S is a contSet
    S = compact_(S)
    # Trivial cases for S
    if hasattr(S, 'isBounded') and not S.isBounded():
        return False, True, np.inf
    if safe_representsa(S, 'emptySet', tol)[0]:
        return True, True, 0
    # For 'approx:st' and 'approx:stDual', these methods are only for zonotope-zonotope containment
    # So we should NOT convert point zonotopes to points - treat them as zonotopes
    if method not in ['approx:st', 'approx:stDual']:
        try:
            isPoint, p = safe_representsa(S, 'point', tol, return_point=True)
            if isPoint:
                if p is None or not isinstance(p, np.ndarray):
                    raise ValueError("representsa_ did not return a valid numpy array for S as point.")
                # Convert 'approx' to 'exact' for point containment
                point_method = 'exact' if method == 'approx' else method
                return priv_zonotopeContainment_pointContainment(Z, p, point_method, tol, scalingToggle)
        except Exception as ME:
            if hasattr(ME, 'identifier') and (getattr(ME, 'identifier', None) == 'CORA:notSupported' or getattr(ME, 'identifier', None) == 'MATLAB:maxlhs'):
                pass
            else:
                raise ME
    # Method dispatch
    if method in ['exact', 'exact:venum', 'exact:polymax']:
        return aux_exactParser(Z, S, method, tol, maxEval, certToggle, scalingToggle)
    elif method in ['approx', 'approx:st', 'approx:stDual']:
        return aux_approxParser(Z, S, method, tol, maxEval, certToggle, scalingToggle)
    elif method in ['sampling', 'sampling:primal', 'sampling:dual']:
        return aux_samplingParser(Z, S, method, tol, maxEval, certToggle, scalingToggle)
    elif method == 'opt':
        if not hasattr(S, '__class__') or S.__class__.__name__.lower() != 'zonotope':
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
        # Try genetic, fallback to DIRECT
        try:
            return priv_zonotopeContainment_geneticMaximization(S, Z, tol, maxEval, scalingToggle)
        except Exception as e:
            # Check if it's a NotImplementedError (genetic algorithm not available)
            if isinstance(e, NotImplementedError):
                # Show warning - genetic algorithm not available in Python implementation
                import warnings
                warnings.warn(
                    'The genetic algorithm optimization is not available in the Python '
                    'implementation. The DIRECT algorithm will be used instead for solving '
                    'the zonotope containment problem.',
                    category=UserWarning
                )
            return priv_zonotopeContainment_DIRECTMaximization(S, Z, tol, maxEval, scalingToggle)
    else:
        raise CORAerror('CORA:noSpecificAlg', method, Z, S)

def aux_exactParser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    class_name = S.__class__.__name__.lower() if hasattr(S, '__class__') else ''
    if class_name in ['interval', 'zonotope']:
        if class_name == 'interval' and hasattr(S, 'toZonotope'):
            S = S.toZonotope()
        if method == 'exact' and hasattr(S, 'dim') and S.dim() >= 4:
            method = 'exact:venum'
        else:
            method = 'exact:polymax'
    elif class_name in ['conhyperplane', 'emptyset', 'fullspace', 'halfspace', 'polytope', 'zonobundle']:
        if method == 'exact':
            method = 'exact:polymax'
    elif class_name in ['capsule', 'ellipsoid']:
        if method == 'exact':
            method = 'exact:polymax'
        elif method == 'exact:venum':
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    else:
        raise CORAerror('CORA:noExactAlg', Z, S)
    # Algorithm dispatch
    if method == 'exact:venum':
        if class_name == 'zonotope':
            return priv_zonotopeContainment_vertexEnumeration(S, Z, tol, scalingToggle)
        else:
            # Fallback to conZonotope - let it raise the error if not implemented
            try:
                from ..conZonotope.conZonotope import ConZonotope
                cZ = ConZonotope(Z)
                return cZ.contains_(S, 'exact:venum', tol, maxEval, certToggle, scalingToggle)
            except (ImportError, AttributeError, NotImplementedError):
                # conZonotope not available or doesn't support this - let it raise the error
                raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    elif method == 'exact:polymax':
        # MATLAB: convert Z to polytope and call polytope.contains_
        # polytope.contains_ now handles all ContSets including zonotope via supportFunc_
        from ..polytope.polytope import Polytope
        P = Polytope(Z)
        return P.contains_(S, 'exact:polymax', tol, maxEval, certToggle, scalingToggle)
    else:
        raise CORAerror('CORA:noSpecificAlg', method, Z, S)

def aux_approxParser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    class_name = S.__class__.__name__.lower() if hasattr(S, '__class__') else ''
    if class_name in ['interval', 'zonotope']:
        if class_name == 'interval' and hasattr(S, 'toZonotope'):
            S = S.toZonotope()
        if certToggle and method == 'approx':
            method = 'approx:stDual'
        elif method == 'approx':
            method = 'approx:st'
    elif class_name in ['conhyperplane', 'emptyset', 'fullspace', 'halfspace', 'polytope', 'zonobundle']:
        if method != 'approx':
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    elif class_name in ['capsule', 'ellipsoid']:
        if method != 'approx':
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    else:
        if method != 'approx':
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    # Algorithm dispatch
    if method == 'approx':
        # Fallback to conZonotope - let it raise the error if not implemented
        try:
            from ..conZonotope.conZonotope import ConZonotope
            cZ = ConZonotope(Z)
            return cZ.contains_(S, method, tol, maxEval, certToggle, scalingToggle)
        except (ImportError, AttributeError, NotImplementedError):
            # conZonotope not available or doesn't support this - let it raise the error
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    elif method == 'approx:st':
        return priv_zonotopeContainment_SadraddiniTedrake(S, Z, tol, scalingToggle)
    elif method == 'approx:stDual':
        return priv_zonotopeContainment_SadraddiniTedrakeDual(S, Z, tol, scalingToggle)
    else:
        raise CORAerror('CORA:noSpecificAlg', method, Z, S)

def aux_samplingParser(Z, S, method, tol, maxEval, certToggle, scalingToggle):
    class_name = S.__class__.__name__.lower() if hasattr(S, '__class__') else ''
    if class_name == 'conzonotope':
        # Fallback to conZonotope - let it raise the error if not implemented
        try:
            from ..conZonotope.conZonotope import ConZonotope
            cZ = ConZonotope(Z)
            return cZ.contains_(S, method, tol, maxEval, certToggle, scalingToggle)
        except (ImportError, AttributeError, NotImplementedError):
            # conZonotope not available or doesn't support this - let it raise the error
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    elif class_name == 'zonotope':
        if method in ['sampling', 'sampling:primal']:
            return priv_zonotopeContainment_zonoSampling(S, Z, tol, maxEval, scalingToggle)
        elif method == 'sampling:dual':
            return priv_zonotopeContainment_zonoSamplingDual(S, Z, tol, maxEval, scalingToggle)
        else:
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    elif class_name == 'ellipsoid':
        if method in ['sampling', 'sampling:primal']:
            return priv_zonotopeContainment_ellipsoidSampling(S, Z, tol, maxEval, scalingToggle)
        elif method == 'sampling:dual':
            return priv_zonotopeContainment_ellipsoidSamplingDual(S, Z, tol, maxEval, scalingToggle)
        else:
            raise CORAerror('CORA:noSpecificAlg', method, Z, S)
    else:
        raise CORAerror('CORA:noSpecificAlg', method, Z, S) 