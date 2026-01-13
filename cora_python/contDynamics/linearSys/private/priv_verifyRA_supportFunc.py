"""
priv_verifyRA_supportFunc - verification for linear systems
   via reach-avoid with support functions:
   quicker verification algorithm based on non-trivial
   exploitations of the standard propagation-based wrapping-free
   reachability algorithm to the extent that reachable sets only computed
   implicitly with respect to their distance to an unsafe set
   caution: specification needs to be given as a safe set!

Syntax:
    res = priv_verifyRA_supportFunc(linsys,params,options,spec)

Inputs:
    linsys - linearSys object
    params - model parameters
    options - algorithm parameters
    spec - safe set as object of class specification

Outputs:
    res - boolean (true if specification verified, false if not)
    fals - dict containing falsifying trajectory
           .x0 ... point from initial set
           .u  ... piecewise-constant input values
           .tu ... switching times of .u
    savedata - distances for plotting

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: linearSys/verify

Authors:       Mark Wetzlinger
Written:       04-April-2022
Last update:   22-April-2022
Last revision: ---
               Automatic python translation: Florian Nüssel BA 2025
"""

import numpy as np
import time
import scipy.linalg
from typing import Any, Dict, List, Optional, Tuple, Union
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope
# Note: All methods are used on objects, not as standalone functions
from cora_python.g.functions.matlab.struct.rmiffield import rmiffield
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


# Auxiliary functions -----------------------------------------------------

def aux_getSetsFromSpec(spec: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract safe sets G and unsafe sets F from the specifications
    (to be deleted (once integration in verify is complete))
    
    Args:
        spec: list of specification objects
        
    Returns:
        G: list of safe set dicts with keys 'set' and 'time'
        F: list of unsafe set dicts with keys 'set', 'time', 'int', 'isBounded'
    """
    # Normalize spec to always be a list (MATLAB handles both scalar and array)
    if not isinstance(spec, list):
        spec = [spec]
    
    # MATLAB: G = {}; F = {};
    G = []
    F = []
    
    # MATLAB: for i = 1:length(spec)
    for i in range(len(spec)):
        # MATLAB: if strcmp(spec(i).type,'safeSet')
        if spec[i].type == 'safeSet':
            # MATLAB: G{end+1}.set = normalizeConstraints(polytope(spec(i).set),'A');
            P = Polytope(spec[i].set)
            P_norm = P.normalizeConstraints('A')
            # MATLAB: G{end}.time = spec(i).time;
            G.append({'set': P_norm, 'time': spec[i].time})
        
        # MATLAB: elseif strcmp(spec(i).type,'unsafeSet')
        elif spec[i].type == 'unsafeSet':
            # MATLAB: tmp = normalizeConstraints(polytope(spec(i).set),'A');
            P = Polytope(spec[i].set)
            tmp = P.normalizeConstraints('A')
            # MATLAB: if size(tmp.A,1) > 1
            if tmp.A.shape[0] > 1:
                # MATLAB: F{end+1}.set = tmp;
                # MATLAB: F{end}.time = spec(i).time;
                F_entry = {'set': tmp, 'time': spec[i].time}
                # MATLAB: if size(F{end}.set.A,1) > size(F{end}.set.A,2)
                if tmp.A.shape[0] > tmp.A.shape[1]:
                    # MATLAB: F{end}.int = interval(F{end}.set);
                    F_entry['int'] = tmp.interval()
                    # MATLAB: F{end}.isBounded = ~(any(isinf(infimum(F{end}.int))) ...
                    #                            | any(isinf(supremum(F{end}.int))));
                    int_inf = F_entry['int'].infimum()
                    int_sup = F_entry['int'].supremum()
                    F_entry['isBounded'] = not (np.any(np.isinf(int_inf)) or np.any(np.isinf(int_sup)))
                else:
                    # MATLAB: F{end}.isBounded = false;
                    F_entry['isBounded'] = False
                F.append(F_entry)
            else:
                # MATLAB: G{end+1}.set = polytope(-tmp.A,-tmp.b);
                # MATLAB: G{end}.time = spec(i).time;
                G.append({'set': Polytope(-tmp.A, -tmp.b), 'time': spec[i].time})
        
        else:
            # MATLAB: throw(CORAerror('CORA:notSupported',...))
            raise CORAerror('CORA:notSupported',
                           'This type of specification is not supported!')
    
    return G, F


def aux_canonicalForm(linsys: Any, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Put inhomogeneity to canonical forms:
       Ax + Bu + c + w  ->  Ax + u, where u \in U + uTransVec
       Cx + Du + k + v  ->  Cx + v, where v \in V + vTransVec
    the sets params.U and params.V return being centered at the origin, all
    (potentially piecewise-constant) offsets are comprised in the vectors
    params.uTransVec and params.vTransVec
    
    Args:
        linsys: linearSys object
        params: model parameters dict
        
    Returns:
        params: modified params dict (in canonical form)
        origInput: dict with original input (keys: 'U', 'u')
    """
    # MATLAB: if isa(params.W,'interval')
    if isinstance(params.get('W'), Interval):
        # MATLAB: params.W = linsys.E*zonotope(params.W);
        params['W'] = linsys.E @ params['W'].zonotope()
    # MATLAB: if isa(params.V,'interval')
    if isinstance(params.get('V'), Interval):
        # MATLAB: params.V = linsys.F*zonotope(params.V);
        # F is (outputs, noises), V should be in noise space
        # Convert Interval to zonotope first, then multiply by F
        v_zonotope = params['V'].zonotope()
        # Check dimension: v_zonotope should have dimension matching F.shape[1] (noise dimension)
        v_dim = v_zonotope.dim()
        f_noise_dim = linsys.F.shape[1] if linsys.F.ndim == 2 else 1
        if v_dim != f_noise_dim:
            raise CORAerror('CORA:dimensionMismatch',
                          f'V dimension ({v_dim}) does not match F noise dimension ({f_noise_dim}). '
                          f'V should be in noise space (dimension {f_noise_dim}).')
        params['V'] = linsys.F @ v_zonotope
    
    # read out disturbance
    # MATLAB: centerW = center(params.W);
    centerW = params['W'].center()
    # MATLAB: W = params.W + (-centerW);
    W = params['W'] + (-centerW)
    # read out sensor noise, combine with feedthrough if given
    # MATLAB: if any(any(linsys.D))
    # IMPORTANT: Save U in input space BEFORE transforming it to state space
    # We need U in input space for D @ U operation (D is (outputs, inputs))
    U_input_space = params['U']  # U is still in input space at this point
    if linsys.D is not None and np.any(linsys.D):
        # MATLAB: params.V = linsys.D * params.U + params.V;
        # D is (outputs, inputs), U should be in input space
        params['V'] = linsys.D @ U_input_space + params['V']
    # MATLAB: centerV = center(params.V);
    centerV = params['V'].center()
    # MATLAB: params.V = params.V + (-centerV);
    params['V'] = params['V'] + (-centerV)
    
    # initialize input vector for state and output equation (if sequence given)
    # MATLAB: if isfield(params,'uTransVec')
    if 'uTransVec' in params:
        # time-varying input vector
        # MATLAB: uVec = params.uTransVec;
        uVec = params['uTransVec']
    else:
        # no time-varying input vector, but uTrans given
        # MATLAB: uVec = params.uTrans;
        uVec = params.get('uTrans', np.zeros((linsys.nr_of_inputs, 1)))
    
    # save original input for falsifying trajectory
    # MATLAB: origInput.U = params.U;
    # MATLAB: origInput.u = uVec;
    origInput = {'U': params['U'], 'u': uVec}
    
    # put output equation in canonical form
    # MATLAB: if any(any(linsys.D))
    if linsys.D is not None and np.any(linsys.D):
        # MATLAB: params.vTransVec = linsys.D * uVec + linsys.k + centerV;
        params['vTransVec'] = linsys.D @ uVec + (linsys.k if linsys.k is not None else 0) + centerV
    else:
        # MATLAB: params.vTransVec = linsys.k + centerV;
        params['vTransVec'] = (linsys.k if linsys.k is not None else 0) + centerV
    
    # put state equation in canonical form
    # MATLAB: params.U = linsys.B * params.U + W;
    # B is (states, inputs), U should be in input space (dimension = inputs)
    u_dim = params['U'].dim()
    b_input_dim = linsys.B.shape[1] if linsys.B.ndim == 2 else 1
    if u_dim != b_input_dim:
        raise CORAerror('CORA:dimensionMismatch',
                      f'U dimension ({u_dim}) does not match B input dimension ({b_input_dim}). '
                      f'U should be in input space (dimension {b_input_dim}) before B transformation.')
    params['U'] = linsys.B @ params['U'] + W
    # MATLAB: params.uTransVec = linsys.B * uVec + linsys.c + centerW;
    params['uTransVec'] = linsys.B @ uVec + (linsys.c if linsys.c is not None else 0) + centerW
    
    # remove fields for safety
    # MATLAB: params = rmfield(params,'W');
    params = rmiffield(params, 'W')
    # note: U and V now overwritten!
    
    return params, origInput


def aux_initExpmat(obj: Any) -> Dict[str, Any]:
    """
    Initialize exponential matrix structures
    
    Args:
        obj: linearSys object
        
    Returns:
        expmat: dict with exponential matrix structures
    """
    # MATLAB: expmat.conv = true;
    expmat = {'conv': True}
    # MATLAB: expmat.Apower{1} = obj.A;
    expmat['Apower'] = [obj.A]
    # MATLAB: expmat.Apower_abs{1} = abs(obj.A);
    expmat['Apower_abs'] = [np.abs(obj.A)]
    # MATLAB: expmat.Apos = cell(0);
    expmat['Apos'] = []
    # MATLAB: expmat.Aneg = cell(0);
    expmat['Aneg'] = []
    # MATLAB: expmat.Deltatk = [];
    expmat['Deltatk'] = None
    
    # precompute inverse of A matrix
    # MATLAB: expmat.isAinv = rank(full(obj.A)) == obj.nrOfDims;
    A_full = obj.A if not hasattr(obj.A, 'toarray') else obj.A.toarray()
    expmat['isAinv'] = np.linalg.matrix_rank(A_full) == obj.nr_of_dims
    # MATLAB: expmat.Ainv = [];
    expmat['Ainv'] = None
    # MATLAB: if expmat.isAinv
    if expmat['isAinv']:
        # MATLAB: expmat.Ainv = inv(obj.A);
        expmat['Ainv'] = np.linalg.inv(obj.A)
    
    return expmat


def aux_initStructsFlags(obj: Any, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool, np.ndarray, bool, bool]:
    """
    Initialize all auxiliary structs and flags
    
    Args:
        obj: linearSys object
        params: model parameters dict
        
    Returns:
        dist: empty dict for distances
        fals: dict for falsifying trajectory
        isU: boolean (true if input set U is non-empty)
        G_U: generator matrix of U
        isu: boolean (true if input vector is non-zero)
        isuconst: boolean (true if input vector is constant)
    """
    # struct for distances
    # MATLAB: dist = [];
    dist = {}
    
    # saving of operations for affine systems (u = const. over entire time
    # horizon) vs. system with varying u or even uncertainty U
    # MATLAB: G_U = generators(params.U);
    G_U = params['U'].generators()
    # MATLAB: isU = ~isempty(G_U);
    isU = G_U.shape[1] > 0 if G_U.ndim == 2 else len(G_U) > 0
    # MATLAB: isu = any(any(params.uTransVec));
    isu = np.any(params['uTransVec'])
    # MATLAB: isuconst = size(params.uTransVec,2) == 1;
    isuconst = params['uTransVec'].shape[1] == 1 if params['uTransVec'].ndim == 2 else True
    # sparsity for speed up (acc. to numeric tests only for very sparse
    # matrices actually effective)
    # MATLAB: if nnz(obj.A) / numel(obj.A) < 0.1
    A_full = obj.A if not hasattr(obj.A, 'toarray') else obj.A.toarray()
    if hasattr(obj.A, 'nnz'):
        nnz_A = obj.A.nnz
    else:
        nnz_A = np.count_nonzero(A_full)
    numel_A = A_full.size
    if nnz_A / numel_A < 0.1:
        # MATLAB: obj.A = sparse(obj.A);
        import scipy.sparse
        obj.A = scipy.sparse.csr_matrix(obj.A)
    
    # struct for falsifying trajectory
    # MATLAB: fals.tFinal = []; fals.x0 = []; fals.u = []; fals.tu = [];
    fals = {'tFinal': None, 'x0': None, 'u': None, 'tu': None}
    
    return dist, fals, isU, G_U, isu, isuconst


def aux_uTrans_vTrans(t: float, timeStep: float, uTransVec: np.ndarray, vTransVec: np.ndarray,
                      tu: np.ndarray, tnextSwitch: float, uTrans: np.ndarray, 
                      vTransNext: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    (note: it is already ensured that the input vector changes over time)
    reads the value of the input vector (uTrans, vTrans) from the matrix
    storing all input vectors (uTransVec, vTransVec) for the current step
    
    in each step, we compute the output set for
       R([t_k,t_k+1])   using   vTransVec(k) = vTrans
    and
       R(t_k+1)   using   vTransVec(k+1) = vTransNext,
    so we have return two values for v
    
    Args:
        t: current time
        timeStep: time step size
        uTransVec: matrix of input vectors (each column is a time step)
        vTransVec: matrix of output vectors (each column is a time step)
        tu: switching times vector
        tnextSwitch: next switching time
        uTrans: current input vector (will be updated)
        vTransNext: next output vector (will be updated)
        
    Returns:
        uTrans: updated input vector
        vTransNext: updated next output vector
        tnextSwitch: updated next switching time
    """
    # MATLAB: if withinTol(t,tnextSwitch)
    if withinTol(t, tnextSwitch):
        # take correct matrix index depending on current time (only compute
        # this if the next switching time has actually come)
        # MATLAB: idx = find(t >= tu,1,'last');
        idx = np.where(t >= tu)[0]
        if len(idx) > 0:
            idx = idx[-1]  # last index
        else:
            idx = 0
        # MATLAB: uTrans = uTransVec(:,idx);
        uTrans = uTransVec[:, idx:idx+1] if uTransVec.ndim == 2 else uTransVec
        
        # update time of next input vector switch
        # MATLAB: tnextSwitch = Inf;
        tnextSwitch = np.inf
        # MATLAB: if idx < length(tu)
        if idx < len(tu) - 1:
            # MATLAB: tnextSwitch = tu(idx+1);
            tnextSwitch = tu[idx + 1]
    
    # MATLAB: if t == 0 || (size(vTransVec,2) > 1 && withinTol(t+timeStep,tnextSwitch))
    if t == 0 or (vTransVec.shape[1] > 1 and withinTol(t + timeStep, tnextSwitch)):
        # also entered in initial step (to assign vnext), otherwise this
        # if-condition and the above one should not be entered in the same step
        
        # value for the end of the step
        # MATLAB: idx = find(t + timeStep >= tu,1,'last');
        idx = np.where(t + timeStep >= tu)[0]
        if len(idx) > 0:
            idx = idx[-1]  # last index
        else:
            idx = 0
        # MATLAB: vTransNext = vTransVec(:,idx);
        vTransNext = vTransVec[:, idx:idx+1] if vTransVec.ndim == 2 else vTransVec
        
        # no update for next input vector switch here
    
    return uTrans, vTransNext, tnextSwitch


def aux_timeStep(timeStep: float, tFinal: float, isuconst: bool, tu: np.ndarray) -> float:
    """
    We compute the largest time step size so that we can use a constant time
    step size which hits all switching times exactly; the input argument
    value for the time step size is an upper bound of the desired value
    
    Args:
        timeStep: initial time step size (upper bound)
        tFinal: final time
        isuconst: boolean (true if input vector is constant)
        tu: switching times vector
        
    Returns:
        timeStep: computed time step size
    """
    # MATLAB: if isuconst
    if isuconst:
        # no switches in constant input vector
        # MATLAB: timeStep = tFinal / ceil(tFinal / timeStep);
        timeStep = tFinal / np.ceil(tFinal / timeStep)
        return timeStep
    
    # find largest time step size considering switches in piecewise-constant
    # input vector
    
    # duration of each piecewise-constant input vector
    # MATLAB: if withinTol(tFinal-tu(end),0)
    if withinTol(tFinal - tu[-1], 0):
        # systems with inhomogeneity in the output equation
        # MATLAB: constInt = diff(tu)';
        constInt = np.diff(tu).reshape(1, -1)
    else:
        # systems without inhomogeneity in the output equation
        # MATLAB: constInt = [diff(tu)',tFinal - tu(end)];
        constInt = np.concatenate([np.diff(tu).reshape(1, -1), 
                                   np.array([[tFinal - tu[-1]]])], axis=1)
    # minimum duration and corresponding number of total steps
    # MATLAB: timeStep = min([min(constInt),timeStep]);
    timeStep = min(np.min(constInt), timeStep)
    # MATLAB: steps = ceil(tFinal / timeStep);
    steps = int(np.ceil(tFinal / timeStep))
    
    # max number of steps
    # MATLAB: maxSteps = 1000000000;
    maxSteps = 1000000000
    
    # loop over increasing number of steps
    # MATLAB: while true
    while True:
        # resulting time step size
        # MATLAB: timeStep = tFinal/steps;
        timeStep = tFinal / steps
        # check if that time step size divides all durations of
        # piecewise-constant input vectors into integers
        # MATLAB: temp = constInt ./ timeStep;
        temp = constInt / timeStep
        # MATLAB: if all(withinTol(temp,round(temp)))
        if np.all(withinTol(temp, np.round(temp))):
            # time step found
            # MATLAB: break
            break
        
        # increment number of steps
        # MATLAB: steps = steps + 1;
        steps = steps + 1
        
        # stopping condition: no time step found until arbitrary value
        # MATLAB: if steps > maxSteps
        if steps > maxSteps:
            # MATLAB: throw(CORAerror("CORA:specialError",...))
            raise CORAerror("CORA:specialError",
                           f"No duration from {steps} to {maxSteps} "
                           f"time steps can divide all individual piecewise-constant "
                           f"input vector durations into integers.")
    
    return timeStep


def aux_removeFromUnsat(FGunsat: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Adapt FGunsat so that timeInterval is not part of time intervals covered
    
    Args:
        FGunsat: array of unverified time intervals (N x 2, each row is [start, end])
        t: time interval to remove [t_start, t_end]
        
    Returns:
        FGunsat: updated array with time interval removed
    """
    # MATLAB: FGunsat_col = reshape(FGunsat',numel(FGunsat),1);
    FGunsat_col = FGunsat.T.flatten()
    # MATLAB: if mod(sum(t(1) >= FGunsat_col),2) ~= 0
    if np.mod(np.sum(t[0] >= FGunsat_col), 2) != 0:
        # lower bound starts inside unverified time interval
        # t(1) \in [ timeInterval )
        # MATLAB: idx = find(t(1) >= FGunsat(:,1) & t(1) <= FGunsat(:,2));
        idx_mask = (t[0] >= FGunsat[:, 0]) & (t[0] <= FGunsat[:, 1])
        idx = np.where(idx_mask)[0]
        if len(idx) > 0:
            idx = idx[0]  # first match
        
        # MATLAB: if t(2) <= FGunsat(idx,2)
        if t[1] <= FGunsat[idx, 1]:
            # MATLAB: if t(1) > FGunsat(idx,1)
            if t[0] > FGunsat[idx, 0]:
                # MATLAB: FGunsat = [FGunsat(1:idx-1,:); ...
                #     [FGunsat(idx,1), t(1)]; FGunsat(idx:end,:)];
                FGunsat = np.vstack([
                    FGunsat[:idx, :],
                    np.array([[FGunsat[idx, 0], t[0]]]),
                    FGunsat[idx:, :]
                ])
                # MATLAB: idx = idx + 1;
                idx = idx + 1
            # split, potential merge later
            # MATLAB: FGunsat(idx,1) = t(2);
            FGunsat[idx, 0] = t[1]
            # MATLAB: t = [];
            t = np.array([])
        else:
            # remove interval, potential merge later
            # MATLAB: FGunsat(idx,2) = t(1);
            FGunsat[idx, 1] = t[0]
            # MATLAB: if idx < size(FGunsat,1)
            if idx < FGunsat.shape[0] - 1:
                # MATLAB: t(1) = FGunsat(idx+1,1);
                t[0] = FGunsat[idx + 1, 0]
            # MATLAB: if t(2) <= t(1)
            if len(t) > 0 and t[1] <= t[0]:
                # MATLAB: t = [];
                t = np.array([])
    
    # now: lower bound starts in between unverified time intervals or at
    # maximum at the start point of an unverified set
    # t(1) \in [ notTimeInterval )
    # MATLAB: while ~isempty(t)
    while len(t) > 0:
        # MATLAB: idx = find(t(1) <= FGunsat(:,1),1,'first');
        idx_mask = t[0] <= FGunsat[:, 0]
        idx = np.where(idx_mask)[0]
        if len(idx) > 0:
            idx = idx[0]  # first match
        else:
            break  # No matching interval found
        
        # upper bound is at least at the beginning of the next time interval
        # MATLAB: if t(2) <= FGunsat(idx,2)
        if t[1] <= FGunsat[idx, 1]:
            # split, potential merge later
            # MATLAB: FGunsat(idx,1) = t(2);
            FGunsat[idx, 0] = t[1]
            # MATLAB: t = [];
            t = np.array([])
        else:
            # remove entire thing (full time interval verified)
            # MATLAB: if idx < size(FGunsat,1)
            if idx < FGunsat.shape[0] - 1:
                # MATLAB: t(1) = FGunsat(idx,2);
                t[0] = FGunsat[idx, 1]
                # MATLAB: if t(2) < FGunsat(idx+1,1)
                if t[1] < FGunsat[idx + 1, 0]:
                    # MATLAB: t = [];
                    t = np.array([])
            else:
                # MATLAB: t = [];
                t = np.array([])
            # MATLAB: FGunsat(idx,:) = [];
            FGunsat = np.delete(FGunsat, idx, axis=0)
    
    # remove
    # MATLAB: idxRemove = abs(FGunsat(:,2) - FGunsat(:,1)) < 1e-14;
    idxRemove = np.abs(FGunsat[:, 1] - FGunsat[:, 0]) < 1e-14
    # MATLAB: FGunsat(idxRemove,:) = [];
    FGunsat = FGunsat[~idxRemove, :]
    
    return FGunsat


def aux_getApower(eta: int, A: np.ndarray, Apower: List[np.ndarray], 
                  A_abs: Optional[np.ndarray] = None, 
                  Apower_abs: Optional[List[np.ndarray]] = None) -> Union[Tuple[List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    This function ensures that the eta-th power of A and |A| is computed
    (this is necessary, since we do not know the largest power in advance,
    and we want to save computations as much as possible)
    we do not compute A^eta but A^eta / eta! instead to increase the stability
    -> this has to be taken into account in all use cases!
    this is currently not enacted for |A|^eta
    
    Args:
        eta: power index
        A: matrix A
        Apower: list of computed powers A^i/i! (1-indexed)
        A_abs: (optional) |A| matrix
        Apower_abs: (optional) list of computed powers |A|^i
        
    Returns:
        Apower: updated list of powers
        Apower_abs: (optional) updated list of absolute powers
    """
    import scipy.sparse
    
    # check A^eta
    # MATLAB: if length(Apower) >= eta
    if len(Apower) >= eta:
        # read from memory
        pass
    else:
        # compute all terms A^i/i! until eta
        # MATLAB: maxeta = length(Apower);
        # Note: MATLAB uses 1-based indexing: Apower{1} = A^1/1!, Apower{2} = A^2/2!, etc.
        # Python uses 0-based: Apower[0] = A^1/1!, Apower[1] = A^2/2!, etc.
        # So MATLAB's Apower{i} = Python's Apower[i-1]
        maxeta = len(Apower)
        # MATLAB: for i=maxeta:eta-1
        # This computes Apower{maxeta+1} through Apower{eta}
        # In Python, we need to compute Apower[maxeta] through Apower[eta-1]
        for i in range(maxeta, eta):
            # MATLAB: Apower{i+1} = Apower{i} * A / (i+1);
            # MATLAB's Apower{i} is Python's Apower[i-1], MATLAB's Apower{i+1} is Python's Apower[i]
            # So: Apower[i] = Apower[i-1] * A / (i+1)
            Apower_next = Apower[i - 1] @ A / (i + 1)
            # sparse/full storage for more efficiency
            # MATLAB: if nnz(Apower{i+1}) / (size(Apower{i+1},1)^2) < 0.1
            if hasattr(Apower_next, 'nnz'):
                nnz_val = Apower_next.nnz
            else:
                nnz_val = np.count_nonzero(Apower_next)
            if nnz_val / (Apower_next.shape[0] ** 2) < 0.1:
                # MATLAB: Apower{i+1} = sparse(Apower{i+1});
                Apower_next = scipy.sparse.csr_matrix(Apower_next)
            else:
                # MATLAB: Apower{i+1} = full(Apower{i+1});
                if hasattr(Apower_next, 'toarray'):
                    Apower_next = Apower_next.toarray()
            Apower.append(Apower_next)
    
    if Apower_abs is not None and A_abs is not None:
        # check |A|^eta
        # MATLAB: if length(Apower_abs) >= eta
        if len(Apower_abs) >= eta:
            # read from memory
            pass
        else:
            # compute all powers |A|^i until eta
            # MATLAB: maxeta = length(Apower_abs);
            # Note: MATLAB uses 1-based indexing, Python uses 0-based
            maxeta = len(Apower_abs)
            # MATLAB: for i=maxeta:eta-1
            for i in range(maxeta, eta):
                # MATLAB: Apower_abs{i+1} = Apower_abs{i}*A_abs;
                # MATLAB's Apower_abs{i} is Python's Apower_abs[i-1]
                Apower_abs_next = Apower_abs[i - 1] @ A_abs
                # sparse/full storage for more efficiency
                # MATLAB: if nnz(Apower_abs{i+1}) / (size(Apower_abs{i+1},1)^2) < 0.1
                if hasattr(Apower_abs_next, 'nnz'):
                    nnz_val = Apower_abs_next.nnz
                else:
                    nnz_val = np.count_nonzero(Apower_abs_next)
                if nnz_val / (Apower_abs_next.shape[0] ** 2) < 0.1:
                    # MATLAB: Apower_abs{i+1} = sparse(Apower_abs{i+1});
                    Apower_abs_next = scipy.sparse.csr_matrix(Apower_abs_next)
                else:
                    # MATLAB: Apower_abs{i+1} = full(Apower_abs{i+1});
                    if hasattr(Apower_abs_next, 'toarray'):
                        Apower_abs_next = Apower_abs_next.toarray()
                Apower_abs.append(Apower_abs_next)
        return Apower, Apower_abs
    
    return Apower


def aux_getAposneg(eta: int, Apos: List[np.ndarray], Aneg: List[np.ndarray], 
                   Apower_eta: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    The separation of A^eta into positive and negative indices can be
    precomputed and saved; Apower_eta has to match eta correctly!
    
    Args:
        eta: power index
        Apos: list of positive parts of A^i
        Aneg: list of negative parts of A^i
        Apower_eta: A^eta / eta! matrix
        
    Returns:
        Apos: updated list with positive part of A^eta
        Aneg: updated list with negative part of A^eta
    """
    # MATLAB: if length(Apos) >= eta && ~isempty(Apos{eta})
    if len(Apos) >= eta and (len(Apos) > 0 and Apos[eta - 1] is not None):
        # ... then also length(Aneg) >= eta
        # read from memory
        pass
    else:
        # new method (slightly faster)
        # MATLAB: Aneg{eta} = Apower_eta;
        Aneg_eta = Apower_eta.copy()
        # MATLAB: Apos{eta} = Apower_eta;
        Apos_eta = Apower_eta.copy()
        # MATLAB: Aneg{eta}(Aneg{eta} > 0) = 0;
        Aneg_eta[Aneg_eta > 0] = 0
        # MATLAB: Apos{eta}(Apos{eta} < 0) = 0;
        Apos_eta[Apos_eta < 0] = 0
        
        # Extend lists if needed
        while len(Aneg) < eta:
            Aneg.append(None)
        while len(Apos) < eta:
            Apos.append(None)
        
        Aneg[eta - 1] = Aneg_eta
        Apos[eta - 1] = Apos_eta
    
    return Apos, Aneg


def aux_intmat(obj: Any, isu: bool, expmat: Dict[str, Any], timeStep: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Computation of F and G for a given time step size; currently, these
    variables are computed by a Taylor series until floating-point precision,
    i.e., we increase the truncation order until the additional values are so
    small that the stored number (finite precision!) does not change anymore
    
    Args:
        obj: linearSys object
        isu: boolean (true if input vector is non-zero)
        expmat: dict with exponential matrix structures
        timeStep: time step size
        
    Returns:
        expmat: updated exponential matrix structures
        intmatF: dict with interval matrix F (keys: 'set', 'rad', 'center')
        intmatG: dict with interval matrix G (keys: 'set', 'rad', 'center')
    """
    # load data from object/options structure
    # MATLAB: A = obj.A;
    A = obj.A
    # MATLAB: n = obj.nrOfDims;
    n = obj.nr_of_dims
    
    # initialize auxiliary variables and flags for loop
    # MATLAB: Asum_pos_F = zeros(n);
    Asum_pos_F = np.zeros((n, n))
    # MATLAB: Asum_neg_F = zeros(n);
    Asum_neg_F = np.zeros((n, n))
    # MATLAB: stoploop_F = false;
    stoploop_F = False
    # MATLAB: if isu
    if isu:
        # MATLAB: Asum_pos_G = zeros(n);
        Asum_pos_G = np.zeros((n, n))
        # MATLAB: Asum_neg_G = zeros(n);
        Asum_neg_G = np.zeros((n, n))
        # MATLAB: stoploop_G = false;
        stoploop_G = False
    else:
        # MATLAB: stoploop_G = true;
        stoploop_G = True
        # MATLAB: intmatG.set = interval(zeros(n),zeros(n));
        intmatG = {
            'set': Interval(np.zeros((n, n)), np.zeros((n, n))),
            'rad': np.zeros((n, n)),
            'center': np.zeros((n, n))
        }
    
    # MATLAB: eta = 1;
    eta = 1
    # MATLAB: while true
    while True:
        # exponential: 1:eta
        
        # compute powers
        # MATLAB: expmat.Apower = aux_getApower(eta,A,expmat.Apower);
        expmat['Apower'] = aux_getApower(eta, A, expmat['Apower'])
        # MATLAB: Apower_eta = expmat.Apower{eta};
        Apower_eta = expmat['Apower'][eta - 1]  # 0-based indexing
        
        # F starts at eta = 2, so skip for eta = 1
        # MATLAB: if eta==1; eta = eta + 1; continue; end
        if eta == 1:
            eta = eta + 1
            continue
        
        # tie/inputTie: 2:eta
        # note: usually, inputTie goes to eta+1 (with eta from F), but since we
        # compute terms until floating-point precision, this does not need to
        # be respected (only if we were to use a remainder term E, which then
        # would necessarily need to be adapted to a fixed eta)
        
        # compute factor (factorial already included in powers of A)
        # MATLAB: exp1 = -(eta)/(eta-1); exp2 = -1/(eta-1);
        exp1 = -(eta) / (eta - 1)
        exp2 = -1 / (eta - 1)
        # MATLAB: factor = ((eta)^exp1-(eta)^exp2) * timeStep^eta;
        factor = ((eta ** exp1) - (eta ** exp2)) * (timeStep ** eta)
        
        # MATLAB: if ~stoploop_F
        if not stoploop_F:
            # MATLAB: [expmat.Apos,expmat.Aneg] = ...
            #     aux_getAposneg(eta,expmat.Apos,expmat.Aneg,Apower_eta);
            expmat['Apos'], expmat['Aneg'] = aux_getAposneg(eta, expmat['Apos'], expmat['Aneg'], Apower_eta)
            
            # if new term does not change result anymore, loop to be finished
            # MATLAB: Asum_add_pos_F = factor*expmat.Aneg{eta};
            Asum_add_pos_F = factor * expmat['Aneg'][eta - 1]  # 0-based indexing
            # MATLAB: Asum_add_neg_F = factor*expmat.Apos{eta};
            Asum_add_neg_F = factor * expmat['Apos'][eta - 1]  # 0-based indexing
            
            # safety check (if time step size too large, then the sum converges
            # too late so we already have Inf values)
            # MATLAB: if eta == 75
            if eta == 75:
                # MATLAB: intmatF.set = []; intmatG.set = [];
                intmatF = {'set': None}
                if isu:
                    intmatG = {'set': None}
                # MATLAB: expmat.conv = false; return
                expmat['conv'] = False
                return expmat, intmatF, intmatG
            
            # compute ratio for floating-point precision
            # MATLAB: if all(all(Asum_add_pos_F <= eps * Asum_pos_F)) && ...
            #         all(all(Asum_add_neg_F >= eps * Asum_neg_F))
            eps_val = np.finfo(float).eps
            if np.all(Asum_add_pos_F <= eps_val * Asum_pos_F) and \
               np.all(Asum_add_neg_F >= eps_val * Asum_neg_F):
                # MATLAB: stoploop_F = true;
                stoploop_F = True
                # MATLAB: intmatF.rad = 0.5*(Asum_pos_F - Asum_neg_F);
                intmatF = {
                    'rad': 0.5 * (Asum_pos_F - Asum_neg_F),
                    'center': Asum_neg_F + 0.5 * (Asum_pos_F - Asum_neg_F)
                }
            
            # compute powers; factor is always negative
            # MATLAB: Asum_pos_F = Asum_pos_F + Asum_add_pos_F;
            Asum_pos_F = Asum_pos_F + Asum_add_pos_F
            # MATLAB: Asum_neg_F = Asum_neg_F + Asum_add_neg_F;
            Asum_neg_F = Asum_neg_F + Asum_add_neg_F
        
        # MATLAB: if ~stoploop_G
        if not stoploop_G:
            # MATLAB: [expmat.Apos,expmat.Aneg] = ...
            #     aux_getAposneg(eta-1,expmat.Apos,expmat.Aneg,expmat.Apower{eta-1});
            expmat['Apos'], expmat['Aneg'] = aux_getAposneg(eta - 1, expmat['Apos'], expmat['Aneg'], 
                                                             expmat['Apower'][eta - 2])  # 0-based indexing
            
            # if new term does not change result anymore, loop to be finished
            # we require one additional division by eta as the terms in expmat
            # are divided by (eta-1)! instead of eta! as required
            # MATLAB: Asum_add_pos_G = factor*expmat.Aneg{eta-1} / eta;
            Asum_add_pos_G = factor * expmat['Aneg'][eta - 2] / eta  # 0-based indexing
            # MATLAB: Asum_add_neg_G = factor*expmat.Apos{eta-1} / eta;
            Asum_add_neg_G = factor * expmat['Apos'][eta - 2] / eta  # 0-based indexing
            
            # safety check (if time step size too large, then the sum converges
            # too late so we already have Inf values)
            # MATLAB: if eta == 75
            if eta == 75:
                # MATLAB: intmatF.set = []; intmatG.set = [];
                intmatF = {'set': None}
                intmatG = {'set': None}
                # MATLAB: expmat.conv = false; return
                expmat['conv'] = False
                return expmat, intmatF, intmatG
            
            # compute ratio for floating-point precision
            # MATLAB: if all(all(Asum_add_pos_G <= eps * Asum_pos_G)) && ...
            #         all(all(Asum_add_neg_G >= eps * Asum_neg_G))
            if np.all(Asum_add_pos_G <= eps_val * Asum_pos_G) and \
               np.all(Asum_add_neg_G >= eps_val * Asum_neg_G):
                # MATLAB: stoploop_G = true;
                stoploop_G = True
                # MATLAB: intmatG.rad = 0.5*(Asum_pos_G - Asum_neg_G);
                intmatG = {
                    'rad': 0.5 * (Asum_pos_G - Asum_neg_G),
                    'center': Asum_neg_G + 0.5 * (Asum_pos_G - Asum_neg_G)
                }
            
            # compute powers; factor is always negative
            # MATLAB: Asum_pos_G = Asum_pos_G + Asum_add_pos_G;
            Asum_pos_G = Asum_pos_G + Asum_add_pos_G
            # MATLAB: Asum_neg_G = Asum_neg_G + Asum_add_neg_G;
            Asum_neg_G = Asum_neg_G + Asum_add_neg_G
        
        # instantiate interval matrices if converged
        # MATLAB: if stoploop_F
        if stoploop_F:
            # MATLAB: intmatF.set = interval(Asum_neg_F,Asum_pos_F);
            intmatF['set'] = Interval(Asum_neg_F, Asum_pos_F)
        # MATLAB: if stoploop_G && isu
        if stoploop_G and isu:
            # MATLAB: intmatG.set = interval(Asum_neg_G,Asum_pos_G);
            intmatG['set'] = Interval(Asum_neg_G, Asum_pos_G)
        
        # exit loop if both converged
        # MATLAB: if stoploop_F && stoploop_G
        if stoploop_F and stoploop_G:
            # MATLAB: break;
            break
        
        # increment eta
        # MATLAB: eta = eta + 1;
        eta = eta + 1
    
    return expmat, intmatF, intmatG


def aux_Pu(obj: Any, u: np.ndarray, expmat: Dict[str, Any], timeStep: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computation of the particular solution due to the input vector u using
    a Taylor series where the truncation order is increased until the
    additional values are so small that the stored number (finite precision!)
    does not change anymore; in case the inverse of A exists, we directly
    compute the analytical solution (where the exponential matrix is also
    only computed until finite precision)
    
    Args:
        obj: linearSys object
        u: input vector
        expmat: dict with exponential matrix structures
        timeStep: time step size
        
    Returns:
        Pu: particular solution vector
        expmat: updated exponential matrix structures
    """
    # MATLAB: if ~any(u)
    if not np.any(u):
        # MATLAB: Pu = zeros(obj.nrOfDims,1);
        Pu = np.zeros((obj.nr_of_dims, 1))
    
    # MATLAB: elseif expmat.isAinv
    elif expmat['isAinv']:
        # MATLAB: Pu = expmat.Ainv * (expmat.Deltatk - eye(obj.nrOfDims)) * u;
        Pu = expmat['Ainv'] @ (expmat['Deltatk'] - np.eye(obj.nr_of_dims)) @ u
    
    else:
        # compute by sum until floating-point precision (same as for PU)
        # formula: \sum_{j=1}^\infty \frac{A^{j-1}}{j!} timeStep^{j}
        
        # initialize truncation order
        # MATLAB: eta = 1;
        eta = 1
        
        # first term
        # MATLAB: Asum = timeStep * eye(obj.nrOfDims);
        Asum = timeStep * np.eye(obj.nr_of_dims)
        
        # loop until Asum no longer changes (additional values too small)
        # MATLAB: while true
        while True:
            # increment truncation order
            # MATLAB: eta = eta + 1;
            eta = eta + 1
            
            # get A^eta-1
            # MATLAB: expmat.Apower = aux_getApower(eta-1,obj.A,expmat.Apower);
            expmat['Apower'] = aux_getApower(eta - 1, obj.A, expmat['Apower'])
            # MATLAB: Apower_etaminus1 = expmat.Apower{eta-1};
            Apower_etaminus1 = expmat['Apower'][eta - 2]  # 0-based indexing
            
            # compute additional term (division by (eta-1)! already included in
            # Apower_etaminus1, so one additional /eta required)
            # MATLAB: addTerm = Apower_etaminus1 / eta * timeStep^eta;
            addTerm = Apower_etaminus1 / eta * (timeStep ** eta)
            
            # safety check (if time step size too large, then the sum converges
            # too late so we already have Inf values)
            # MATLAB: if any(any(isinf(addTerm)))
            if np.any(np.isinf(addTerm)):
                # MATLAB: expmat.conv = false; return
                expmat['conv'] = False
                return np.zeros((obj.nr_of_dims, 1)), expmat
            
            # if new term does not change stored values in Asum, i.e., all
            # entries are below floating-point accuracy -> stop loop
            # MATLAB: if all(all(abs(addTerm) <= eps * abs(Asum)))
            eps_val = np.finfo(float).eps
            if np.all(np.abs(addTerm) <= eps_val * np.abs(Asum)):
                # MATLAB: break;
                break
            
            # add term to current Asum
            # MATLAB: Asum = Asum + addTerm;
            Asum = Asum + addTerm
        
        # compute particular solution due to input vector
        # MATLAB: Pu = Asum * u;
        Pu = Asum @ u
    
    return Pu, expmat


def aux_underPU(obj: Any, G_U: np.ndarray, expmat: Dict[str, Any], timeStep: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computation of the particular solution due to the input vector u using
    a Taylor series where the truncation order is increased until the
    additional values are so small that the stored number (finite precision!)
    does not change anymore; in case the inverse of A exists, we directly
    compute the analytical solution (where the exponential matrix is also
    only computed until finite precision)
    note: G_U is non-zero (checked outside)
    
    Args:
        obj: linearSys object
        G_U: generator matrix of input set U
        expmat: dict with exponential matrix structures
        timeStep: time step size
        
    Returns:
        G_underPU: generator matrix for particular solution
        expmat: updated exponential matrix structures
    """
    # MATLAB: if expmat.isAinv
    if expmat['isAinv']:
        # MATLAB: G_underPU = expmat.Ainv * (expmat.Deltatk - eye(obj.nrOfDims)) * G_U;
        G_underPU = expmat['Ainv'] @ (expmat['Deltatk'] - np.eye(obj.nr_of_dims)) @ G_U
    
    else:
        # compute by sum until floating-point precision
        # formula: \sum_{j=1}^\infty \frac{A^{j-1}}{j!} timeStep^{j}
        
        # initialize truncation order
        # MATLAB: eta = 1;
        eta = 1
        
        # first term
        # MATLAB: Asum = timeStep * eye(obj.nrOfDims);
        Asum = timeStep * np.eye(obj.nr_of_dims)
        
        # loop until Asum no longer changes (additional values too small)
        # MATLAB: while true
        while True:
            # increment truncation order
            # MATLAB: eta = eta + 1;
            eta = eta + 1
            
            # get A^eta-1
            # MATLAB: expmat.Apower = aux_getApower(eta-1,obj.A,expmat.Apower);
            expmat['Apower'] = aux_getApower(eta - 1, obj.A, expmat['Apower'])
            # MATLAB: Apower_etaminus1 = expmat.Apower{eta-1};
            Apower_etaminus1 = expmat['Apower'][eta - 2]  # 0-based indexing
            
            # compute additional term (division by (eta-1)! already included in
            # Apower_etaminus1, so one additional /eta required)
            # MATLAB: addTerm = Apower_etaminus1 / eta * timeStep^eta;
            addTerm = Apower_etaminus1 / eta * (timeStep ** eta)
            
            # safety check (if time step size too large, then the sum converges
            # too late so we already have Inf values)
            # MATLAB: if any(any(isinf(addTerm)))
            if np.any(np.isinf(addTerm)):
                # MATLAB: expmat.conv = false; return
                expmat['conv'] = False
                return np.zeros((obj.nr_of_dims, G_U.shape[1])), expmat
            
            # if new term does not change stored values in Asum, i.e., all
            # entries are below floating-point accuracy -> stop loop
            # MATLAB: if all(all(abs(addTerm) <= eps * abs(Asum)))
            eps_val = np.finfo(float).eps
            if np.all(np.abs(addTerm) <= eps_val * np.abs(Asum)):
                # MATLAB: break;
                break
            
            # add term to current Asum
            # MATLAB: Asum = Asum + addTerm;
            Asum = Asum + addTerm
        
        # compute particular solution due to input vector
        # MATLAB: G_underPU = Asum * G_U;
        G_underPU = Asum @ G_U
    
    return G_underPU, expmat


def aux_overPU(obj: Any, G_U: np.ndarray, expmat: Dict[str, Any], timeStep: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Computation of the particular solution due to the uncertain input set U
    using a Taylor series where the truncation order is increased until
    the additional values are so small that the stored number (finite
    precision!) does not change anymore;
    we use only the generator matrix to spare some zonotope function calls
    
    Args:
        obj: linearSys object
        G_U: generator matrix of input set U
        expmat: dict with exponential matrix structures
        timeStep: time step size
        
    Returns:
        G_overPU: generator matrix for particular solution
        expmat: updated exponential matrix structures
    """
    # maximum truncation order
    # MATLAB: maxeta = 75;
    maxeta = 75
    
    # initialize particular solution
    # MATLAB: G_U_size = size(G_U,2);
    G_U_size = G_U.shape[1]
    # MATLAB: G_overPU = zeros(obj.nrOfDims,G_U_size*maxeta);
    G_overPU = np.zeros((obj.nr_of_dims, G_U_size * maxeta))
    # MATLAB: G_overPU(:,1:G_U_size) = timeStep * G_U;
    G_overPU[:, :G_U_size] = timeStep * G_U
    # MATLAB: PU_diag = sum(abs(G_overPU),2);
    PU_diag = np.sum(np.abs(G_overPU), axis=1)
    
    # MATLAB: A = obj.A;
    A = obj.A
    
    # loop until floating-point precision
    # MATLAB: stoploop = false;
    stoploop = False
    # MATLAB: eta = 1;
    eta = 1
    # MATLAB: while true
    while True:
        # compute powers of A
        # MATLAB: expmat.Apower = aux_getApower(eta,A,expmat.Apower);
        expmat['Apower'] = aux_getApower(eta, A, expmat['Apower'])
        # MATLAB: Apower_eta = expmat.Apower{eta};
        Apower_eta = expmat['Apower'][eta - 1]  # 0-based indexing
        
        # additional term (Apower_eta already contains division by (eta)!,
        # thus we require one more /(eta+1) to get correct denominator)
        # MATLAB: addG_PU = Apower_eta / (eta+1) * timeStep^(eta+1) * G_U;
        # Note: In MATLAB, * is matrix multiplication. In Python, we need @ for matrix multiplication
        addG_PU = (Apower_eta / (eta + 1) * (timeStep ** (eta + 1))) @ G_U
        # MATLAB: addG_PU_diag = sum(abs(addG_PU),2);
        addG_PU_diag = np.sum(np.abs(addG_PU), axis=1)
        
        # safety check (if time step size too large, then the sum converges
        # too late so we already have Inf values)
        # previous criterion: any(any(isinf(addPU_diag))) || any(any(isnan(addPU_diag)))
        # ... but very costly for large A!
        
        # check if floating-point precision reached
        # MATLAB: if all( abs(addG_PU_diag) <= eps * abs(PU_diag) )
        eps_val = np.finfo(float).eps
        if np.all(np.abs(addG_PU_diag) <= eps_val * np.abs(PU_diag)):
            # MATLAB: stoploop = true;
            stoploop = True
        
        # add term to simplified value for convergence
        # MATLAB: PU_diag = PU_diag + addG_PU_diag;
        PU_diag = PU_diag + addG_PU_diag
        
        # append to generator matrix
        # MATLAB: G_overPU(:,eta*G_U_size+1:(eta+1)*G_U_size) = addG_PU;
        start_idx = eta * G_U_size
        end_idx = (eta + 1) * G_U_size
        G_overPU[:, start_idx:end_idx] = addG_PU
        
        # break loop, remove remaining pre-allocated zeros
        # MATLAB: if stoploop
        if stoploop:
            # MATLAB: G_overPU(:,(eta+1)*G_U_size+1:end) = [];
            G_overPU = G_overPU[:, :(eta + 1) * G_U_size]
            # MATLAB: break;
            break
        
        # increment eta
        # MATLAB: eta = eta + 1;
        eta = eta + 1
        
        # exit and try again with smaller time step size if truncation order
        # becomes too large
        # MATLAB: if eta == maxeta
        if eta == maxeta:
            # MATLAB: expmat.conv = false; return
            expmat['conv'] = False
            return np.zeros((obj.nr_of_dims, G_U_size)), expmat
    
    return G_overPU, expmat


def aux_affineTimeStep(obj: Any, dist: Dict[str, Any], tFinal: float, timeStep: float,
                       timeStepFactor: float, isuconst: bool, tu: np.ndarray, 
                       normC: float) -> float:
    """
    Adaptation of the time step size for the next iteration based on the
    distance of the affine dynamics solution and the quadratic dependency of
    said distance on the time step size
    
    Args:
        obj: linearSys object
        dist: dict with distance information
        tFinal: final time
        timeStep: current time step size
        timeStepFactor: factor for minimum time step
        isuconst: boolean (true if input vector is constant)
        tu: switching times vector
        normC: norm of output matrix C
        
    Returns:
        timeStep: adapted time step size
    """
    # for sanity check below...
    # MATLAB: timeStep_prev = timeStep;
    timeStep_prev = timeStep
    
    # one time step proposition for each spec
    # MATLAB: nrSpecs = length(dist.affine_ti);
    nrSpecs = len(dist['affine_ti'])
    # MATLAB: timeStep_prop = zeros(1,nrSpecs);
    timeStep_prop = np.zeros(nrSpecs)
    
    # MATLAB: for s=1:nrSpecs
    for s in range(nrSpecs):
        # get time index + start/end of maximum infringement
        # MATLAB: [tempDist,tempIdx] = max(dist.affine_ti{s},[],2);
        tempDist = np.max(dist['affine_ti'][s], axis=1)
        tempIdx = np.argmax(dist['affine_ti'][s], axis=1)
        # MATLAB: [~,tIdx] = max(tempDist);
        tIdx = np.argmax(tempDist)
        
        # compute coefficient for equation: sizeC = (det(eAt)factor)*a*timeStep^2
        # since error set Cbloat shrinks quadratically over timeStep
        # MATLAB: sizeFactor = normC * exp(trace(obj.A * timeStep*(tIdx-1)));
        sizeFactor = normC * np.exp(np.trace(obj.A * timeStep * (tIdx)))
        # MATLAB: if sizeFactor > 0
        if sizeFactor > 0:
            # MATLAB: coeff_a = dist.Cbloat(tIdx,s) / (sizeFactor * timeStep^2);
            coeff_a = dist['Cbloat'][tIdx, s] / (sizeFactor * (timeStep ** 2))
        else:
            # MATLAB: coeff_a = dist.Cbloat(tIdx,s) / timeStep^2;
            coeff_a = dist['Cbloat'][tIdx, s] / (timeStep ** 2)
        
        # compute time step size which would achieve the maximum
        # admissible value for dist.Cbloat(tIdx)
        # MATLAB: tIdx = tIdx + (tempIdx(tIdx) - 1);
        tIdx = tIdx + (tempIdx[tIdx] - 1)
        # MATLAB: timeStep = sqrt(-dist.affine_tp(tIdx,s) / coeff_a);
        timeStep = np.sqrt(-dist['affine_tp'][tIdx, s] / coeff_a)
        # sanity check
        # MATLAB: assert(timeStep < timeStep_prev,'Error in adaptation of time step size');
        assert timeStep < timeStep_prev, 'Error in adaptation of time step size'
        
        # adjust time step size to yield an integer number of steps and is
        # compatible with switches in the input vector
        # MATLAB: timeStep_prop(s) = aux_timeStep(timeStep,tFinal,isuconst,tu);
        timeStep_prop[s] = aux_timeStep(timeStep, tFinal, isuconst, tu)
    
    # ensure that too small values are avoided (which mostly occurs when the
    # quadratic approximation does not provide a good estimate, which in turn
    # often occurs when the previous time step size was too large)
    # MATLAB: timeStep_min = aux_timeStep(timeStep_prev*timeStepFactor,tFinal,isuconst,tu);
    timeStep_min = aux_timeStep(timeStep_prev * timeStepFactor, tFinal, isuconst, tu)
    
    # take maximum of both time step sizes
    # MATLAB: timeStep = max([timeStep_prop, timeStep_min]);
    timeStep = max(np.max(timeStep_prop), timeStep_min)
    
    return timeStep


def aux_affinePUTimeStep(obj: Any, dist: Dict[str, Any], tFinal: float, timeStep: float,
                         timeStepFactor: float, isuconst: bool, tu: np.ndarray, 
                         normC: float, k: int) -> float:
    """
    Adaptation of the time step size for the next iteration based on the
    distance of the full solution and the quadratic dependency of both parts
    of said distance (affine + particular solutions) on the time step size;
    both contributions are affected simultaneously by the same time step size
    
    Args:
        obj: linearSys object
        dist: dict with distance information
        tFinal: final time
        timeStep: current time step size
        timeStepFactor: factor for minimum time step
        isuconst: boolean (true if input vector is constant)
        tu: switching times vector
        normC: norm of output matrix C
        k: time step index
        
    Returns:
        timeStep: adapted time step size
    """
    # for sanity check below...
    # MATLAB: timeStep_prev = timeStep;
    timeStep_prev = timeStep
    
    # case: dist.affine_ti(k) + dist.overPU(k+1) >= 0
    # MATLAB: nrSpecs = length(dist.affine_ti);
    nrSpecs = len(dist['affine_ti'])
    # MATLAB: timeStep_prop = zeros(1,nrSpecs);
    timeStep_prop = np.zeros(nrSpecs)
    
    # MATLAB: for s=1:nrSpecs
    for s in range(nrSpecs):
        # compute margin
        # MATLAB: surplus = max(dist.affine_tp(k:k+1,s)) + dist.underPU(k+1,s);
        surplus = np.max(dist['affine_tp'][k:k+2, s]) + dist['underPU'][k+1, s]
        
        # scaling factor by propagation matrix
        # MATLAB: sizeFactor = normC * exp(trace(obj.A * timeStep*(k-1)));
        sizeFactor = normC * np.exp(np.trace(obj.A * timeStep * (k - 1)))
        
        # compute coefficient for regression equations:
        # 1. sizeC = (det(eAt)factor)*a*timeStep^2
        # 2. sizePU = (det(eAt)factor)*a*timeStep^2
        # since Cbloat and PU shrink quadratically with the time step size
        # MATLAB: if withinTol(sizeFactor,0)
        if withinTol(sizeFactor, 0):
            # MATLAB: coeff_a_affine = dist.Cbloat(k,s) / timeStep^2;
            coeff_a_affine = dist['Cbloat'][k, s] / (timeStep ** 2)
            # MATLAB: coeff_a_overPU = dist.overPU(k+1,s) / timeStep^2;
            coeff_a_overPU = dist['overPU'][k+1, s] / (timeStep ** 2)
        else:
            # MATLAB: coeff_a_affine = dist.Cbloat(k,s) / (sizeFactor * timeStep^2);
            coeff_a_affine = dist['Cbloat'][k, s] / (sizeFactor * (timeStep ** 2))
            # MATLAB: coeff_a_overPU = dist.overPU(k+1,s) / (sizeFactor * timeStep^2);
            coeff_a_overPU = dist['overPU'][k+1, s] / (sizeFactor * (timeStep ** 2))
        
        # compute time step size which would achieve the maximum
        # admissible value for dist.Cbloat(tIdx,s)
        # MATLAB: timeStep_prop(s) = sqrt(-surplus / (coeff_a_affine + coeff_a_overPU));
        timeStep_prop[s] = np.sqrt(-surplus / (coeff_a_affine + coeff_a_overPU))
        
        # adjust time step size to yield an integer number of steps
        # MATLAB: timeStep_prop(s) = aux_timeStep(timeStep_prop(s),tFinal,isuconst,tu);
        timeStep_prop[s] = aux_timeStep(timeStep_prop[s], tFinal, isuconst, tu)
    
    # ensure that too small values are avoided (which mostly occurs when the
    # quadratic approximation does not provide a good estimate, which in turn
    # often occurs when the previous time step size was too large)
    # MATLAB: timeStep_min = aux_timeStep(timeStep_prev*timeStepFactor,tFinal,isuconst,tu);
    timeStep_min = aux_timeStep(timeStep_prev * timeStepFactor, tFinal, isuconst, tu)
    
    # take maximum of both time step sizes
    # MATLAB: timeStep = max([timeStep_prop, timeStep_min]);
    timeStep = max(np.max(timeStep_prop), timeStep_min)
    
    # sanity check
    # MATLAB: assert(timeStep < timeStep_prev,'Error in adaptation of time step size');
    assert timeStep < timeStep_prev, 'Error in adaptation of time step size'
    
    return timeStep


# Main function -------------------------------------------------------------

def priv_verifyRA_supportFunc(linsys: Any, params: Dict[str, Any], 
                              options: Dict[str, Any], spec: Any) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """
    Verification for linear systems via reach-avoid with support functions
    
    Args:
        linsys: linearSys object
        params: model parameters
        options: algorithm parameters
        spec: specification object
        
    Returns:
        res: boolean (true if verified, false if falsified, -1 if undecided)
        fals: dict with falsifying trajectory
        savedata: dict with distances and computation time
    """
    # integrate in verify?
    # MATLAB: [params,options] = validateOptions(linsys,params,options);
    # Use _validateOptions from reach.py (internal validation)
    from cora_python.contDynamics.linearSys.reach import _validateOptions
    params, options = _validateOptions(linsys, params, options)
    
    # start stopwatch
    # MATLAB: onlyLoop = tic;
    onlyLoop = time.time()
    # initializations ---------------------------------------------------------
    # MATLAB: if options.verbose
    if options.get('verbose', False):
        print("...initializations")
    
    # init time shift
    # MATLAB: params.tFinal = params.tFinal - params.tStart;
    params['tFinal'] = params['tFinal'] - params.get('tStart', 0.0)
    
    # initialize satisfaction of specification
    # MATLAB: res = false;
    res = False
    
    # get safe sets and unsafe sets G from specification
    # TODO: remove once integration in verify has been done
    # MATLAB: [safeSet,unsafeSet] = aux_getSetsFromSpec(spec);
    safeSet, unsafeSet = aux_getSetsFromSpec(spec)
    # get halfspaces from specifications (transform all to safe sets)
    # MATLAB: nrSpecs = length(safeSet) + length(unsafeSet);
    nrSpecs = len(safeSet) + len(unsafeSet)
    # MATLAB: Cs = zeros(nrSpecs,linsys.nrOfOutputs);
    Cs = np.zeros((nrSpecs, linsys.nr_of_outputs))
    # MATLAB: ds = zeros(nrSpecs,1);
    ds = np.zeros((nrSpecs, 1))
    # MATLAB: for i=1:length(safeSet)
    for i in range(len(safeSet)):
        # if all values satisfy C*x <= d, system is safe
        # MATLAB: Cs(i,:) = safeSet{i}.set.A;
        Cs[i, :] = safeSet[i]['set'].A
        # MATLAB: ds(i) = safeSet{i}.set.b;
        ds[i] = safeSet[i]['set'].b
    # MATLAB: for j=1:length(unsafeSet)
    for j in range(len(unsafeSet)):
        # if any value satisfies C*x <= d, system is unsafe
        # -> rewrite to safe set: C*x >= d  <=>  -C*x <= -d
        # MATLAB: Cs(i+j,:) = -unsafeSet{j}.set.A;
        Cs[len(safeSet) + j, :] = -unsafeSet[j]['set'].A
        # MATLAB: ds(i+j) = -unsafeSet{j}.set.b;
        ds[len(safeSet) + j] = -unsafeSet[j]['set'].b
    
    # time intervals where specifications are not yet verified
    # TODO: integrate in validateOptions later...
    # MATLAB: specUnsat = cell(nrSpecs,1);
    specUnsat = []
    # MATLAB: for s=1:nrSpecs
    for s in range(nrSpecs):
        # MATLAB: specUnsat{s} = [0,params.tFinal];
        specUnsat.append(np.array([[0, params['tFinal']]]))
    
    # initialize all variables for exponential (propagation) matrix
    # MATLAB: expmat = aux_initExpmat(linsys);
    expmat = aux_initExpmat(linsys)
    
    # rewrite equations in canonical form
    # MATLAB: [params,origInput] = aux_canonicalForm(linsys,params);
    params, origInput = aux_canonicalForm(linsys, params)
    
    # compute directions for support function evaluation: the specifications
    #    Cs*y <= ds
    # are defined in the output space, thus we insert y=C*x to obtain
    #    Cs*y <= d  ->  Cs*(C*x) <= d  ->  (Cs * C)*x <= d
    # and map all Cs to l(i) = Cs(i)*C for further use
    # MATLAB: l = (Cs * full(linsys.C))';
    C_full = linsys.C if not hasattr(linsys.C, 'toarray') else linsys.C.toarray()
    l = (Cs @ C_full).T
    
    # scaling factor due to output matrix (only used for heuristic to adapt the
    # time step size)
    # MATLAB: options.normC = norm(full(linsys.C));
    options['normC'] = np.linalg.norm(C_full)
    
    # init all auxiliary structs and flags
    # MATLAB: [dist,fals,isU,G_U,isu,isuconst] = aux_initStructsFlags(linsys,params);
    dist, fals, isU, G_U, isu, isuconst = aux_initStructsFlags(linsys, params)
    
    # initialize time step size as largest whole number divisor over all
    # individual piecewise-constant inputs
    # MATLAB: nrStepsStart = 100;
    nrStepsStart = 100
    # MATLAB: timeStep = aux_timeStep(params.tFinal/nrStepsStart,params.tFinal,isuconst,params.tu);
    timeStep = aux_timeStep(params['tFinal'] / nrStepsStart, params['tFinal'], isuconst, params.get('tu', np.array([0])))
    # maximum factor by which time step size may decrease in one iteration
    # MATLAB: timeStepFactor = 1/50;
    timeStepFactor = 1.0 / 50.0
    # factor by which time step size decreases if a sum does not converge
    # MATLAB: timeStepFactor_nonconverged = 0.2;
    timeStepFactor_nonconverged = 0.2
    # method from paper: fixed factor
    # MATLAB: timeStepFactor_fixed = 0.2;
    timeStepFactor_fixed = 0.2
    
    # init savedata
    # MATLAB: savedata.tFinal = params.tFinal;
    savedata = {
        'tFinal': params['tFinal'],
        'isU': isU,
        'nrSpecs': nrSpecs,
        'iterations': 0
    }
    
    # plots on (only debugging) or off
    # MATLAB: isplot = false; blobsize = 3;
    isplot = False
    blobsize = 3
    # -------------------------------------------------------------------------
    
    # quick check: does start set already intersect the unsafe set?
    
    # debug ---
    # (debug plotting code skipped - not essential for functionality)
    # debug ---
    
    # shift distance to halfspace by influence from additive output set V
    # (from y = C*x + v, v \in V... only accounting for constant V!),
    # so that now we have already dealt with V for all sets;
    # note: we use only Cs instead of l = Cs*C as V is defined in output space!
    # MATLAB: ds = ds - Cs*center(params.V) - sum(abs(Cs*generators(params.V)),2);
    # Convert to array to avoid matrix type issues with keepdims
    Cs_V_gen = np.asarray(Cs @ params['V'].generators())
    ds = ds - Cs @ params['V'].center() - np.sum(np.abs(Cs_V_gen), axis=1, keepdims=True)
    
    # read data from initial set (these remain constant for the algorithm!)
    # MATLAB: c_X0 = center(params.R0);
    c_X0 = params['R0'].center()
    # MATLAB: G_X0 = generators(params.R0);
    G_X0 = params['R0'].generators()
    
    # compute distance of initial output set to each specification
    # MATLAB: dist_affine_tp_0 = (-ds + l'*c_X0 + sum(abs(l'*G_X0),2) + Cs*params.vTransVec(:,1))';
    # Convert to array to avoid matrix type issues with keepdims
    lT_G_X0 = np.asarray(l.T @ G_X0)
    dist_affine_tp_0 = (-ds + l.T @ c_X0 + np.sum(np.abs(lT_G_X0), axis=1, keepdims=True) + Cs @ params['vTransVec'][:, [0]]).T
    
    # check initial output set (outside of verification loop below)
    # MATLAB: if any(dist_affine_tp_0 > 0)
    if np.any(dist_affine_tp_0 > 0):
        # check which spec causes falsification
        # MATLAB: falsIdx = find(dist_affine_tp_0 > 0,1,'first');
        falsIdx = np.where(dist_affine_tp_0.flatten() > 0)[0][0]
        # time where falsification occurs: initial time
        # MATLAB: fals.tFinal = 0;
        fals['tFinal'] = 0
        # most critical initial state (via support function evaluation)
        # MATLAB: fals.x0 = c_X0 + G_X0*sign(l(:,falsIdx)'*G_X0)';
        fals['x0'] = c_X0 + G_X0 @ np.sign((l[:, falsIdx].T @ G_X0).T)
        # no input trajectory needed: same as input trajectory = 0 at all times
        # MATLAB: fals.u = zeros(dim(origInput.U),1);
        fals['u'] = np.zeros((origInput['U'].dim(), 1))
        # MATLAB: fals.tu = 0;
        fals['tu'] = 0
        # measure elapsed time
        # MATLAB: savedata.tComp = toc(onlyLoop);
        savedata['tComp'] = time.time() - onlyLoop
        # provably falsified -> exit!
        # MATLAB: return;
        return False, fals, savedata
    
    # iteration counter
    # MATLAB: savedata.iterations = 0;
    savedata['iterations'] = 0
    # main loop: continues until specification is verified/falsified
    # MATLAB: while true
    while True:
        # increment counter
        # MATLAB: savedata.iterations = savedata.iterations + 1;
        savedata['iterations'] = savedata['iterations'] + 1
        
        # compute number of steps (we use fixed time step sizes)
        # MATLAB: nrSteps = round(params.tFinal/timeStep);
        nrSteps = int(np.round(params['tFinal'] / timeStep))
        
        # save data
        # MATLAB: savedata.timeStep = timeStep;
        savedata['timeStep'] = timeStep
        # MATLAB: savedata.nrSteps = nrSteps;
        savedata['nrSteps'] = nrSteps
        
        # log
        # MATLAB: if options.verbose
        if options.get('verbose', False):
            # MATLAB: disp("Iteration " + savedata.iterations + ": no. of steps = " + nrSteps + ...
            #     " (time step size = " + timeStep + ", horizon = " + params.tFinal + ")");
            print(f"Iteration {savedata['iterations']}: no. of steps = {nrSteps} "
                  f"(time step size = {timeStep}, horizon = {params['tFinal']})")
        
        # initialize contributions to distance to unsafe set from piecewise
        # constant input solution, affine solution, full curvature
        # MATLAB: dist.Pu = zeros(nrSteps+1,nrSpecs);
        dist['Pu'] = np.zeros((nrSteps + 1, nrSpecs))
        # MATLAB: dist.affine_tp = [dist_affine_tp_0; NaN(nrSteps,nrSpecs)];
        dist['affine_tp'] = np.vstack([dist_affine_tp_0, np.full((nrSteps, nrSpecs), np.nan)])
        # MATLAB: dist.Cbloat = zeros(nrSteps,nrSpecs);
        dist['Cbloat'] = np.zeros((nrSteps, nrSpecs))
        # initialize contribution to distance to unsafe set from input vector
        # in output equation (assuming constant input vector for now,
        # recomputed below if necessary); note: included in dist.affine_tp!
        # MATLAB: dist.vTrans = repmat((Cs*params.vTransVec(:,1))',nrSteps+1,1);
        dist['vTrans'] = np.tile((Cs @ params['vTransVec'][:, [0]]).T, (nrSteps + 1, 1))
        
        # compute exponential matrix for given time step size (+ transposed)
        # MATLAB: if options.verbose
        if options.get('verbose', False):
            # MATLAB: disp("...compute propagation matrix");
            print("...compute propagation matrix")
        # MATLAB: expmat.Deltatk = expm(linsys.A * timeStep);
        A_full = linsys.A if not hasattr(linsys.A, 'toarray') else linsys.A.toarray()
        expmat['Deltatk'] = scipy.linalg.expm(A_full * timeStep)
        # MATLAB: expmat.Deltatk_T = expmat.Deltatk';
        expmat['Deltatk_T'] = expmat['Deltatk'].T
        
        # compute constant input solution
        # MATLAB: if isu
        if isu:
            # assign constant shifts
            # MATLAB: [u,vnext,tnextSwitch] = aux_uTrans_vTrans(0,timeStep,...
            #     params.uTransVec,params.vTransVec,params.tu,0,...
            #     params.uTransVec(:,1),params.vTransVec(:,1));
            u, vnext, tnextSwitch = aux_uTrans_vTrans(0, timeStep,
                                                      params['uTransVec'], params['vTransVec'],
                                                      params.get('tu', np.array([0])), 0,
                                                      params['uTransVec'][:, [0]], params['vTransVec'][:, [0]])
            # MATLAB: Pu = aux_Pu(linsys,u,expmat,timeStep);
            Pu, expmat = aux_Pu(linsys, u, expmat, timeStep)
            # initialize accumulated solution
            # MATLAB: Pu_total = zeros(linsys.nrOfDims,1);
            Pu_total = np.zeros((linsys.nr_of_dims, 1))
            # reduce time step size if computation has not converged
            # MATLAB: if ~expmat.conv
            if not expmat['conv']:
                # MATLAB: timeStep = timeStep * timeStepFactor_nonconverged;
                timeStep = timeStep * timeStepFactor_nonconverged
                # MATLAB: expmat.conv = true;
                expmat['conv'] = True
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: disp("...reduce time step size");
                    print("...reduce time step size")
                # MATLAB: continue;
                continue
        
        # compute interval matrices required for curvature error sets
        # MATLAB: if options.verbose
        if options.get('verbose', False):
            # MATLAB: disp("...compute interval matrices for curvature errors");
            print("...compute interval matrices for curvature errors")
        # MATLAB: [expmat,intmatF,intmatG] = aux_intmat(linsys,isu,expmat,timeStep);
        expmat, intmatF, intmatG = aux_intmat(linsys, isu, expmat, timeStep)
        # reduce time step size if computation has not converged
        # MATLAB: if ~expmat.conv
        if not expmat['conv']:
            # MATLAB: timeStep = timeStep * timeStepFactor_nonconverged;
            timeStep = timeStep * timeStepFactor_nonconverged
            # MATLAB: expmat.conv = true;
            expmat['conv'] = True
            # MATLAB: if options.verbose
            if options.get('verbose', False):
                # MATLAB: disp("...reduce time step size");
                print("...reduce time step size")
            # MATLAB: continue;
            continue
        
        # original computation (not taking time-varying u into account)
        # speed-up
        # MATLAB: cx_Cbloat = intmatF.center * c_X0;
        cx_Cbloat = intmatF['center'] @ c_X0
        # MATLAB: G1x_Cbloat = intmatF.center * G_X0;
        G1x_Cbloat = intmatF['center'] @ G_X0
        # MATLAB: G2x_Cbloat = intmatF.rad * sum(abs([c_X0 G_X0]),2);
        # Convert to array to avoid matrix type issues with keepdims
        cX0_GX0 = np.asarray(np.hstack([c_X0, G_X0]))
        G2x_Cbloat = intmatF['rad'] @ np.sum(np.abs(cX0_GX0), axis=1, keepdims=True)
        # MATLAB: if isu
        if isu:
            # MATLAB: cu_Cbloat = intmatG.center * u;
            cu_Cbloat = intmatG['center'] @ u
            # MATLAB: Gu_Cbloat = intmatG.rad * abs(u);
            Gu_Cbloat = intmatG['rad'] @ np.abs(u)
        # note: G2x and Gu require diag(.), but we can omit it here in order to
        # speed up the computation of dist.Cbloat!
        
        # first curvature distance out of the loop (better indexing)
        # MATLAB: dist.Cbloat(1,:) = l'*cx_Cbloat + sum(abs(l'*G1x_Cbloat),2) ...
        #     + abs(l'*G2x_Cbloat);
        # Convert to array to avoid matrix type issues with keepdims
        lT_G1x = np.asarray(l.T @ G1x_Cbloat)
        dist['Cbloat'][0, :] = (l.T @ cx_Cbloat + np.sum(np.abs(lT_G1x), axis=1, keepdims=True) 
                                + np.abs(l.T @ G2x_Cbloat)).flatten()
        # MATLAB: if isu
        if isu:
            # MATLAB: dist.Cbloat(1,:) = dist.Cbloat(1,:)' ...
            #     + l'*cu_Cbloat + abs(l'*Gu_Cbloat);
            dist['Cbloat'][0, :] = (dist['Cbloat'][0, :].reshape(-1, 1) + l.T @ cu_Cbloat 
                                   + np.abs(l.T @ Gu_Cbloat)).flatten()
        
        # initialize variables for back-propagated direction of support
        # function (including transpose)
        # MATLAB: l_prop = cell(nrSpecs,1);
        l_prop = []
        # MATLAB: l_prop_T = cell(nrSpecs,1);
        l_prop_T = []
        # MATLAB: for s=1:nrSpecs
        for s in range(nrSpecs):
            # MATLAB: l_prop{s} = [l(:,s), zeros(linsys.nrOfDims,nrSteps)];
            l_prop_s = np.hstack([l[:, [s]], np.zeros((linsys.nr_of_dims, nrSteps))])
            l_prop.append(l_prop_s)
            # MATLAB: l_prop_T{s} = l_prop{s}';
            l_prop_T.append(l_prop_s.T)
        
        # log
        # MATLAB: if options.verbose
        if options.get('verbose', False):
            # MATLAB: disp("...compute affine solutions");
            print("...compute affine solutions")
        # MATLAB: if nrSteps > 10
        logVar = None
        if nrSteps > 10:
            # MATLAB: logVar = 0.1;
            logVar = 0.1
        
        # compute distance of affine time-point solutions H(tk) to unsafe set
        # MATLAB: for k=1:nrSteps
        for k in range(nrSteps):
            # log
            # MATLAB: if options.verbose && nrSteps > 10 && k > round(nrSteps * logVar)
            if options.get('verbose', False) and nrSteps > 10 and logVar is not None and k > round(nrSteps * logVar):
                # MATLAB: if logVar == 0.1
                if logVar == 0.1:
                    # MATLAB: fprintf('...');
                    print('...', end='')
                # MATLAB: fprintf([num2str(100*logVar), '%%, ']);
                print(f'{int(100*logVar)}%, ', end='')
                # MATLAB: logVar = logVar + 0.1;
                logVar = logVar + 0.1
            
            # back-propagate direction of support function (incl. transpose)
            # MATLAB: for s=1:nrSpecs
            for s in range(nrSpecs):
                # MATLAB: l_prop{s}(:,k+1) = expmat.Deltatk_T * l_prop{s}(:,k);
                l_prop[s][:, k+1] = expmat['Deltatk_T'] @ l_prop[s][:, k]
                # MATLAB: l_prop_T{s}(k+1,:) = l_prop{s}(:,k+1)';
                l_prop_T[s][k+1, :] = l_prop[s][:, k+1].T
            
            # propagate constant input solution
            # MATLAB: if isu
            if isu:
                # MATLAB: if ~isuconst
                if not isuconst:
                    # compute next(!) Pu and contribution to bloating term
                    # MATLAB: [u,vnext,tnextSwitch] = aux_uTrans_vTrans((k-1)*timeStep,...
                    #     timeStep,params.uTransVec,params.vTransVec,...
                    #     params.tu,tnextSwitch,u,vnext);
                    u, vnext, tnextSwitch = aux_uTrans_vTrans(k * timeStep, timeStep,
                                                              params['uTransVec'], params['vTransVec'],
                                                              params.get('tu', np.array([0])), tnextSwitch, u, vnext)
                    
                    # MATLAB: if k > 1 && ~all(withinTol(u,u_prev))
                    if k > 0 and not np.all(withinTol(u, u_prev)):
                        # only update Pu if necessary
                        # MATLAB: Pu = aux_Pu(linsys,u,expmat,timeStep);
                        Pu, expmat = aux_Pu(linsys, u, expmat, timeStep)
                        # update values for Cbloat
                        # MATLAB: cu_Cbloat = intmatG.center * u;
                        cu_Cbloat = intmatG['center'] @ u
                        # note: Gu requires diag(.), but we can omit it here in
                        # order to speed up the computation of dist.Cbloat!
                        # MATLAB: Gu_Cbloat = intmatG.rad * abs(u);
                        Gu_Cbloat = intmatG['rad'] @ np.abs(u)
                    
                    # store u for comparison to avoid re-computations
                    # MATLAB: u_prev = u;
                    u_prev = u
                
                # accumulate particular solution
                # MATLAB: Pu_total = expmat.Deltatk * Pu_total + Pu;
                Pu_total = expmat['Deltatk'] @ Pu_total + Pu
                
                # actual distance computation (accumulates over time as the
                # particular constant input solution does too)
                # MATLAB: dist.Pu(k+1,:) = (l' * Pu_total)';
                dist['Pu'][k+1, :] = (l.T @ Pu_total).T
                
                # distance contribution from input vector in output equation
                # MATLAB: dist.vTrans(k+1,:) = (Cs*vnext)';
                dist['vTrans'][k+1, :] = (Cs @ vnext).T
            
            # propagate time-point solution H(tk), compute distance of output
            # time-point solution Y(tk) to unsafe set
            # MATLAB: for s=1:nrSpecs
            for s in range(nrSpecs):
                # MATLAB: dist.affine_tp(k+1,s) = -ds(s) ...
                #     + l_prop_T{s}(k+1,:) * c_X0 ...
                #     + sum(abs(l_prop_T{s}(k+1,:)*G_X0),2) ...
                #     + dist.Pu(k+1,s) ...
                #     + dist.vTrans(k+1,s);
                dist['affine_tp'][k+1, s] = (-ds[s] 
                                             + l_prop_T[s][k+1, :] @ c_X0 
                                             + np.sum(np.abs(l_prop_T[s][k+1, :] @ G_X0)) 
                                             + dist['Pu'][k+1, s] 
                                             + dist['vTrans'][k+1, s])
            
            # in case of an intersection, already falsified
            # MATLAB: if any(dist.affine_tp(k+1) > 0)
            if np.any(dist['affine_tp'][k+1, :] > 0):
                # provably falsified!
                # MATLAB: savedata.tComp = toc(onlyLoop);
                savedata['tComp'] = time.time() - onlyLoop
                
                # log
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: fprintf('...falsification detected!\n');
                    print('...falsification detected!')
                
                # check which spec causes falsification
                # MATLAB: falsIdx = find(dist.affine_tp(k+1) > 0,1,'first');
                falsIdx = np.where(dist['affine_tp'][k+1, :] > 0)[0][0]
                # time where falsification occurs
                # MATLAB: fals.tFinal = timeStep * k;
                # In MATLAB, k is 1-based (1:nrSteps), so k=1 means timeStep*1
                # In Python, k is 0-based (0:nrSteps-1), so k=0 should mean timeStep*1
                # Therefore: timeStep * (k + 1)
                fals['tFinal'] = timeStep * (k + 1)
                # starting point x0 which yields the falsifying trajectory
                # MATLAB: fals.x0 = c_X0 + G_X0*sign(l_prop_T{falsIdx}(k+1,:)*G_X0)';
                fals['x0'] = c_X0 + G_X0 @ np.sign((l_prop_T[falsIdx][k+1, :] @ G_X0).T)
                # falsifying piecewise-constant input is equal to uTransVec
                # until current point in time...
                # MATLAB: if ~isu || isuconst
                if not isu or isuconst:
                    # MATLAB: fals.u = origInput.u;
                    fals['u'] = origInput['u']
                    # MATLAB: fals.tu = 0;
                    fals['tu'] = 0
                else:
                    # MATLAB: idx = find(k*timeStep >= params.tu,1,'last');
                    # In MATLAB, k is 1-based, so k*timeStep is checked
                    # In Python, k is 0-based, so (k+1)*timeStep matches MATLAB's k*timeStep
                    idx_arr = np.where((k + 1) * timeStep >= params.get('tu', np.array([0])))[0]
                    if len(idx_arr) > 0:
                        # MATLAB find(...,'last') returns 1-based index
                        # Python np.where returns 0-based indices, so idx_arr[-1] is 0-based
                        # To match MATLAB's 1-based indexing for slicing, we need idx_arr[-1] + 1
                        idx = idx_arr[-1] + 1
                    else:
                        idx = 1
                    # MATLAB: fals.u = origInput.u(:,1:idx);
                    # MATLAB uses 1-based indexing, so (1:idx) means columns 1 to idx (inclusive)
                    # Python uses 0-based indexing, so [:idx] means columns 0 to idx-1 (exclusive)
                    # To get columns 0 to idx-1 (inclusive), we use [:idx]
                    fals['u'] = origInput['u'][:, :idx]
                    # MATLAB: fals.tu = params.tu(1:idx);
                    # MATLAB (1:idx) means elements 1 to idx (inclusive) in 1-based indexing
                    # Python [:idx] means elements 0 to idx-1 (exclusive) in 0-based indexing
                    # To get elements 0 to idx-1 (inclusive), we use [:idx]
                    fals['tu'] = params.get('tu', np.array([0]))[:idx]
                # save data
                # MATLAB: savedata.fals_tFinal = fals.tFinal;
                savedata['fals_tFinal'] = fals['tFinal']
                # MATLAB: savedata.dist_affinetp = dist.affine_tp;
                savedata['dist_affinetp'] = dist['affine_tp']
                # MATLAB: return;
                return False, fals, savedata
            
            # computation of distance contribution from curvature term and
            # offset in output equation required for the time-interval affine
            # solution H(tauk)
            # MATLAB: if k < nrSteps
            if k < nrSteps - 1:  # k is 0-based
                # MATLAB: for s=1:nrSpecs
                for s in range(nrSpecs):
                    # MATLAB: dist.Cbloat(k+1,s) = l_prop_T{s}(k+1,:)*cx_Cbloat ...
                    #     + sum(abs(l_prop_T{s}(k+1,:)*G1x_Cbloat)) ...
                    #     + abs(l_prop_T{s}(k+1,:)*G2x_Cbloat);
                    dist['Cbloat'][k+1, s] = (l_prop_T[s][k+1, :] @ cx_Cbloat 
                                             + np.sum(np.abs(l_prop_T[s][k+1, :] @ G1x_Cbloat)) 
                                             + np.abs(l_prop_T[s][k+1, :] @ G2x_Cbloat))
                    # MATLAB: if isu
                    if isu:
                        # MATLAB: dist.Cbloat(k+1,s) = dist.Cbloat(k+1,s) ...
                        #     + l_prop_T{s}(k+1,:)*cu_Cbloat ...
                        #     + abs(l_prop_T{s}(k+1,:)*Gu_Cbloat);
                        dist['Cbloat'][k+1, s] = (dist['Cbloat'][k+1, s] 
                                                  + l_prop_T[s][k+1, :] @ cu_Cbloat 
                                                  + np.abs(l_prop_T[s][k+1, :] @ Gu_Cbloat))
        
        # end log
        # MATLAB: if options.verbose && nrSteps > 10
        if options.get('verbose', False) and nrSteps > 10:
            # MATLAB: fprintf('100%%\n');
            print('100%')
        
        # compute distance for time-interval solution H(tauk): add curvature
        # to both start and end set of each H(tk) -> 2 columns (note: the
        # piecewise-constant input vector vTransVec is included in
        # dist.affine_tp)
        # MATLAB: dist.affine_ti = cell(nrSpecs,1);
        dist['affine_ti'] = []
        # MATLAB: for s=1:nrSpecs
        for s in range(nrSpecs):
            # MATLAB: dist.affine_ti{s} = [dist.affine_tp(1:end-1,s),...
            #     dist.affine_tp(2:end,s)] + dist.Cbloat(:,s);
            dist['affine_ti'].append(np.hstack([dist['affine_tp'][:-1, [s]], 
                                                dist['affine_tp'][1:, [s]]]) + dist['Cbloat'][:, [s]])
        # TODO: inner-approx of affine_ti with -dist.Cbloat!
        
        # gather already verified time intervals (unless input set provided)
        # MATLAB: if ~isU
        if not isU:
            # MATLAB: if options.verbose
            if options.get('verbose', False):
                # MATLAB: disp("...check distances");
                print("...check distances")
            
            # quick version
            # MATLAB: if all(cellfun(@(x) all(all(x<0)),dist.affine_ti,'UniformOutput',true))
            if all(np.all(x < 0) for x in dist['affine_ti']):
                # MATLAB: specUnsat = cell(nrSpecs,1);
                specUnsat = [[] for _ in range(nrSpecs)]
            # MATLAB: elseif any(cellfun(@(x) any(any(x<0)),dist.affine_ti,'UniformOutput',true))
            elif any(np.any(x < 0) for x in dist['affine_ti']):
                # MATLAB: for s=1:nrSpecs
                for s in range(nrSpecs):
                    # MATLAB: t = [0,timeStep];
                    t = np.array([[0, timeStep]])
                    # MATLAB: for k=1:nrSteps
                    for k in range(nrSteps):
                        # MATLAB: if all(dist.affine_ti{s}(k,:) < 0)
                        if np.all(dist['affine_ti'][s][k, :] < 0):
                            # MATLAB: specUnsat{s} = aux_removeFromUnsat(specUnsat{s},t);
                            specUnsat[s] = aux_removeFromUnsat(specUnsat[s], t.flatten())
                        # MATLAB: t = t + timeStep;
                        t = t + timeStep
            
            # suggest refinement based on H(tauk) (important if no input set)
            # MATLAB: if ~all(cellfun(@isempty,specUnsat,'UniformOutput',true))
            # Check if all arrays are empty (size == 0, not len == 0)
            if not all(x.size == 0 for x in specUnsat):
                # shorten time horizon if possible
                # MATLAB: params.tFinal = max(cellfun(@(x) x(end,2),specUnsat,'UniformOutput',true));
                params['tFinal'] = max(x[-1, 1] if len(x) > 0 else 0 for x in specUnsat)
                # method from paper: fixed factor
                # MATLAB: timeStep = timeStep * timeStepFactor_fixed;
                timeStep = timeStep * timeStepFactor_fixed
        
        # remainder only required if there is an input to the system
        # MATLAB: else
        else:
            # MATLAB: if options.verbose
            if options.get('verbose', False):
                # MATLAB: disp("...compute inner/outer-approximation of particular solution");
                print("...compute inner/outer-approximation of particular solution")
            # compute inner-approximation underPU(Delta t)
            # MATLAB: underPU_G = aux_underPU(linsys,G_U,expmat,timeStep);
            underPU_G, expmat = aux_underPU(linsys, G_U, expmat, timeStep)
            # reduce time step size if computation has not converged
            # MATLAB: if ~expmat.conv
            if not expmat['conv']:
                # MATLAB: timeStep = timeStep * timeStepFactor_nonconverged;
                timeStep = timeStep * timeStepFactor_nonconverged
                # MATLAB: expmat.conv = true;
                expmat['conv'] = True
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: disp("...reduce time step size");
                    print("...reduce time step size")
                # MATLAB: continue;
                continue
            # compute outer-approximation overPU(Delta t)
            # MATLAB: overPU_G = aux_overPU(linsys,G_U,expmat,timeStep);
            overPU_G, expmat = aux_overPU(linsys, G_U, expmat, timeStep)
            # reduce time step size if computation has not converged
            # MATLAB: if ~expmat.conv
            if not expmat['conv']:
                # MATLAB: timeStep = timeStep * timeStepFactor_nonconverged;
                timeStep = timeStep * timeStepFactor_nonconverged
                # MATLAB: expmat.conv = true;
                expmat['conv'] = True
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: disp("...reduce time step size");
                    print("...reduce time step size")
                # MATLAB: continue;
                continue
            
            # propagate distance shifts (index k for solution at tk since both
            # inner- and outer-approximation are 0 at time t0)
            # MATLAB: dist.underPU = zeros(nrSteps+1,nrSpecs);
            dist['underPU'] = np.zeros((nrSteps + 1, nrSpecs))
            # MATLAB: dist.overPU = zeros(nrSteps+1,nrSpecs);
            dist['overPU'] = np.zeros((nrSteps + 1, nrSpecs))
            # MATLAB: for s=1:nrSpecs
            for s in range(nrSpecs):
                # MATLAB: dist.underPU(:,s) = [0;cumsum(sum(abs(l_prop_T{s}(1:end-1,:)*underPU_G),2))];
                dist['underPU'][:, s] = np.concatenate([[0], 
                                                         np.cumsum(np.sum(np.abs(l_prop_T[s][:-1, :] @ underPU_G), axis=1))])
                # MATLAB: dist.overPU(:,s) = [0;cumsum(sum(abs(l_prop_T{s}(1:end-1,:)*overPU_G),2))];
                dist['overPU'][:, s] = np.concatenate([[0], 
                                                        np.cumsum(np.sum(np.abs(l_prop_T[s][:-1, :] @ overPU_G), axis=1))])
            
            # MATLAB: if options.verbose
            if options.get('verbose', False):
                # MATLAB: disp("...check distances");
                print("...check distances")
            # quick check: distance of H(tk) + underPU(tk) is not ok
            # MATLAB: if any(any(dist.affine_tp + dist.underPU > 0))
            if np.any(dist['affine_tp'] + dist['underPU'] > 0):
                # provably falsified!
                # MATLAB: savedata.tComp = toc(onlyLoop);
                savedata['tComp'] = time.time() - onlyLoop
                
                # MATLAB: if options.verbose
                if options.get('verbose', False):
                    # MATLAB: fprintf('...falsification detected!\n');
                    print('...falsification detected!')
                
                # check which spec causes falsification
                # MATLAB: for s=1:nrSpecs
                falsIdx = None
                for s in range(nrSpecs):
                    # MATLAB: if any(dist.affine_tp(:,s) + dist.underPU(:,s) > 0)
                    if np.any(dist['affine_tp'][:, s] + dist['underPU'][:, s] > 0):
                        # MATLAB: falsIdx = s; break
                        falsIdx = s
                        break
                # time where falsification occurs
                # MATLAB: tIdx = find( dist.affine_tp(:,falsIdx) ...
                #     + dist.underPU(:,falsIdx) > 0,1,'first');
                tIdx = np.where(dist['affine_tp'][:, falsIdx] + dist['underPU'][:, falsIdx] > 0)[0][0]
                # MATLAB: fals.tFinal = timeStep * (tIdx-1);
                fals['tFinal'] = timeStep * tIdx
                
                # starting point x0 which yields the falsifying trajectory
                # MATLAB: fals.x0 = c_X0 + G_X0*sign(l_prop_T{falsIdx}(tIdx,:)*G_X0)';
                fals['x0'] = c_X0 + G_X0 @ np.sign((l_prop_T[falsIdx][tIdx, :] @ G_X0).T)
                
                # input contribution: different length depending on whether we
                # have feedthrough or not
                # MATLAB: u_tIdx = tIdx - 1;
                # In MATLAB, tIdx is 1-based, so tIdx-1 gives 0-based index
                # In Python, tIdx is already 0-based, so we use tIdx directly
                u_tIdx = tIdx
                # MATLAB: if any(any(linsys.D))
                if linsys.D is not None and np.any(linsys.D):
                    # MATLAB: u_tIdx = tIdx;
                    # In MATLAB, tIdx is 1-based, so u_tIdx = tIdx (1-based)
                    # In Python, tIdx is 0-based, so u_tIdx = tIdx + 1 (to match MATLAB's 1-based)
                    u_tIdx = tIdx + 1
                # contribution from input set
                # MATLAB: fals.u = fliplr(generators(origInput.U) * ...
                #     sign(l_prop_T{falsIdx}(1:u_tIdx,:)*underPU_G)');
                fals['u'] = np.fliplr(origInput['U'].generators() @ 
                                      np.sign((l_prop_T[falsIdx][:u_tIdx, :] @ underPU_G).T))
                # MATLAB: if isuconst
                if isuconst:
                    # contribution from constant input vector
                    # MATLAB: fals.u = fals.u + origInput.u;
                    fals['u'] = fals['u'] + origInput['u']
                else:
                    # contribution from piecewise-constant input vectors
                    # MATLAB: for j=1:u_tIdx
                    for j in range(u_tIdx):
                        # MATLAB: idx = find((j-1)*timeStep >= params.tu,1,'last');
                        idx = np.where((j) * timeStep >= params.get('tu', np.array([0])))[0]
                        if len(idx) > 0:
                            idx = idx[-1]
                        else:
                            idx = 0
                        # MATLAB: fals.u(:,j) = fals.u(:,j) + origInput.u(:,idx);
                        fals['u'][:, j] = fals['u'][:, j] + origInput['u'][:, idx]
                # MATLAB: if tIdx == 1
                if tIdx == 0:  # 0-based indexing
                    # MATLAB: fals.tu = 0;
                    fals['tu'] = 0
                else:
                    # MATLAB: fals.tu = (0:timeStep:(tIdx-2)*timeStep)';
                    fals['tu'] = np.arange(0, tIdx * timeStep, timeStep).reshape(-1, 1)
                # save data (plots)
                # MATLAB: savedata.fals_tFinal = fals.tFinal;
                savedata['fals_tFinal'] = fals['tFinal']
                # MATLAB: savedata.dist_affinetp = dist.affine_tp;
                savedata['dist_affinetp'] = dist['affine_tp']
                # MATLAB: savedata.dist_affinetp_underPU = dist.affine_tp + dist.underPU;
                savedata['dist_affinetp_underPU'] = dist['affine_tp'] + dist['underPU']
                # MATLAB: return;
                return False, fals, savedata
            
            else:
                # loop over specs...
                # MATLAB: timeStep_prop = Inf(nrSteps,nrSpecs);
                timeStep_prop = np.full((nrSteps, nrSpecs), np.inf)
                # MATLAB: for s=1:nrSpecs
                for s in range(nrSpecs):
                    # quick check: all distances H(tauk) + overPU(tk+1) are ok
                    # MATLAB: if all(dist.affine_ti{s} + dist.overPU(2:end,s) < 0)
                    if np.all(dist['affine_ti'][s] + dist['overPU'][1:, s].reshape(-1, 1) < 0):
                        # MATLAB: specUnsat{s} = double.empty(2,0); continue
                        specUnsat[s] = np.array([]).reshape(2, 0)
                        continue
                    
                    # check distances one-by-one
                    # MATLAB: t = [0,timeStep];
                    t = np.array([[0, timeStep]])
                    # MATLAB: for k=1:nrSteps
                    for k in range(nrSteps):
                        # 1. distance of H(tauk) + overPU(tk+1) is ok
                        # MATLAB: if all(dist.affine_ti{s}(k) + dist.overPU(k+1,s) < 0)
                        if np.all(dist['affine_ti'][s][k, :] + dist['overPU'][k+1, s] < 0):
                            # add to already verified time intervals
                            # MATLAB: specUnsat{s} = aux_removeFromUnsat(specUnsat{s},t);
                            specUnsat[s] = aux_removeFromUnsat(specUnsat[s], t.flatten())
                        
                        # 2. distance of H(tauk) + overPU(tk+1) is not ok
                        else:
                            # case: dist.affine_ti(k) + dist.overPU(k+1) >= 0
                            # suggest refinement for Delta t by using quadratic
                            # dependence of the size of overPU (and Cbloat...)
                            # (commented out in MATLAB - using fixed factor instead)
                            pass
                        
                        # shift time interval
                        # MATLAB: t = t + timeStep;
                        t = t + timeStep
                
                # method from paper: fixed factor
                # MATLAB: timeStep = timeStep * timeStepFactor_fixed;
                timeStep = timeStep * timeStepFactor_fixed
                # refine time step size
                # (commented out in MATLAB)
        
        # specification is satisfied over entire time horizon
        # MATLAB: if all(cellfun(@isempty,specUnsat,'UniformOutput',true))
        # Check if all arrays are empty (size == 0, not len == 0, since reshape(2,0) has len=2 but size=0)
        if all(x.size == 0 for x in specUnsat):
            # MATLAB: break
            break
        # for safety reasons, stop in case time step size becomes too small
        # -> return that no decision (-1) could be obtained!
        # MATLAB: elseif timeStep < 1e-12
        elif timeStep < 1e-12:
            # MATLAB: res = -1; return
            res = -1
            savedata['tComp'] = time.time() - onlyLoop
            return res, fals, savedata
    
    # verification successful
    # MATLAB: res = true;
    res = True
    
    # savedata for visualization
    # MATLAB: savedata.dist_affinetp = dist.affine_tp;
    savedata['dist_affinetp'] = dist['affine_tp']
    # MATLAB: savedata.dist_affineti = dist.affine_ti;
    savedata['dist_affineti'] = dist['affine_ti']
    # MATLAB: if isU
    if isU:
        # MATLAB: savedata.dist_affinetp_underPU = zeros(nrSteps+1,nrSpecs);
        savedata['dist_affinetp_underPU'] = np.zeros((nrSteps + 1, nrSpecs))
        # MATLAB: savedata.dist_affineti_overPU = cell(nrSpecs,1);
        savedata['dist_affineti_overPU'] = []
        # MATLAB: for s=1:nrSpecs
        for s in range(nrSpecs):
            # MATLAB: savedata.dist_affinetp_underPU(:,s) = dist.affine_tp(:,s) + dist.underPU(:,s);
            savedata['dist_affinetp_underPU'][:, s] = dist['affine_tp'][:, s] + dist['underPU'][:, s]
            # MATLAB: savedata.dist_affineti_overPU{s,1} = dist.affine_ti{s} + dist.overPU(2:end,s);
            savedata['dist_affineti_overPU'].append(dist['affine_ti'][s] + dist['overPU'][1:, s].reshape(-1, 1))
    
    # measure elapsed time
    # MATLAB: savedata.tComp = toc(onlyLoop);
    savedata['tComp'] = time.time() - onlyLoop
    
    return res, fals, savedata

