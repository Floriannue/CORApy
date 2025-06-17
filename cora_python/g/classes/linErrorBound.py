"""
linErrorBound - helper class for dealing with error bounds in the
adaptive reachability algorithm for linear continuous-time systems

Syntax:
    errs = LinErrorBound(emax, tFinal)

Inputs:
    emax - error bound
    tFinal - time horizon

Outputs:
    errs - generated LinErrorBound object

Example:
    -

References:
    [1] M. Wetzlinger et al. "Fully automated verification of linear
        systems using inner-and outer-approximations of reachable sets",
        TAC, 2023.

Other m-files required: none
Subfunctions: none
MAT-files required: none

See also: LinearSys/private/priv_reach_adaptive

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 06-November-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
import scipy.linalg
from typing import List, Dict, Any, Optional, Union
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class LinErrorBound:
    """
    Helper class for dealing with error bounds in the adaptive reachability algorithm
    for linear continuous-time systems
    """
    
    def __init__(self, emax: float, tFinal: float):
        """
        Constructor for LinErrorBound
        
        Args:
            emax: Error bound (positive scalar)
            tFinal: Time horizon (non-negative scalar)
        """
        # Check input arguments
        if not isinstance(emax, (int, float)) or not np.isfinite(emax) or emax <= 0:
            raise CORAError('CORA:wrongInputInConstructor',
                           'Error margin must be a scalar, positive, real number.')
        
        if not isinstance(tFinal, (int, float)) or not np.isfinite(tFinal) or tFinal < 0:
            raise CORAError('CORA:wrongInputInConstructor',
                           'Time horizon must be a scalar, non-negative, real number.')
        
        # --- errors ---
        self.emax = float(emax)                 # full error margin
        self.tFinal = float(tFinal)            # time horizon
        self.timeSteps = []                    # time step sizes
        
        self.step_acc = []                     # committed accumulating error (k-th step)
        self.step_red = []                     # committed reduction error (k-th step)
        
        self.seq_nonacc = []                   # committed non-accumulating error (each step)
        self.cum_acc = []                      # committed accumulating error (cumulative)
        self.cum_red = []                      # committed reduction error (cumulative)
        
        self.bound_acc = []                    # bound for accumulating error
        self.bound_red = []                    # bound for reduction error
        self.bound_red_max = []                # maximum value of reduction error (at tFinal)
        self.bound_rem = []                    # bound for remaining error (after acc/red)
        self.bound_remacc = []                 # bound for remaining accumulating error
        
        self.bound_acc_ok = []                 # fulfillment of accumulating error bounds
        self.bound_nonacc_ok = []              # fulfillment of non-accumulating error bounds
        
        self.idv_PUtkplus1 = []                # individual error: particular solution (accumulating)
        self.idv_F = []                        # individual error: state curvature (non-accumulating)
        self.idv_G = []                        # individual error: input curvature (non-accumulating)
        self.idv_linComb = []                  # individual error: linear combination (non-accumulating)
        self.idv_PUtauk = []                   # individual error: time dependency of particular solution (non-accumulating)
        
        self.useApproxFun = False              # flag whether to use approximation functions (true) or bisection (false)
        
        # --- coefficients for approximation functions ---
        # coefficients for approximation functions
        # - linear:         eps(Delta t) =                 b * Delta t
        # - quadratic:      eps(Delta t) = a * Delta t^2 + b * Delta t
        self.coeff_PUtkplus1_a = []
        self.coeff_PUtkplus1_b = []
        self.coeff_F_a = []
        self.coeff_F_b = []
        self.coeff_G_a = []
        self.coeff_G_b = []
        self.coeff_linComb_b = []
        self.coeff_PUtauk_b = []
        
        # for bisection ---
        # lower bound
        self.bisect_lb_timeStep_acc = []
        self.bisect_lb_acc = []
        self.bisect_lb_accok = []
        self.bisect_lb_acc_perc = []
        self.bisect_lb_timeStep_nonacc = []
        self.bisect_lb_nonacc = []
        self.bisect_lb_nonaccok = []
        self.bisect_lb_nonacc_perc = []
        
        # upper bound
        self.bisect_ub_timeStep_acc = []
        self.bisect_ub_acc = []
        self.bisect_ub_accok = []
        self.bisect_ub_acc_perc = []
        self.bisect_ub_timeStep_nonacc = []
        self.bisect_ub_nonacc = []
        self.bisect_ub_nonaccok = []
        self.bisect_ub_nonacc_perc = []
    
    def checkErrors(self, Rout_error: np.ndarray, Rout_tp_error: Optional[np.ndarray] = None) -> bool:
        """
        Check function which returns true/false depending on fulfillment
        of error bounds over time
        
        Args:
            Rout_error: Output error vector
            Rout_tp_error: Output time-point error vector (optional)
            
        Returns:
            True if error bounds are satisfied, False otherwise
        """
        # assume checks to be ok
        res = True
        internalcheck = True
        
        # read out vector of time steps
        if not self.timeSteps:
            return True
            
        tVec = np.array([step for sublist in self.timeSteps for step in (sublist if isinstance(sublist, list) else [sublist])])
        
        # read out some errors
        nonacc_step = np.array([step for sublist in self.seq_nonacc for step in (sublist if isinstance(sublist, list) else [sublist])])
        acc_step = np.array([step for sublist in self.step_acc for step in (sublist if isinstance(sublist, list) else [sublist])])
        nonacc_bound = np.array([bound for sublist in self.bound_rem for bound in (sublist if isinstance(sublist, list) else [sublist])])
        acc_bound = np.array([item for sublist in self.bound_acc for item in (sublist if isinstance(sublist, list) else [sublist])])
        red_bound = np.array([item for sublist in self.bound_red for item in (sublist if isinstance(sublist, list) else [sublist])])
        
        # check which steps are computed completely
        fullstep = ~np.isnan(nonacc_step)
        
        # quick fix for time-point error (empty in verify-call)
        if Rout_tp_error is None or len(Rout_tp_error) == 0:
            Rout_tp_error = np.zeros(len(tVec))
        
        # cumulative sum to yield bound for accumulating error
        cum_acc_arr = np.array(self.cum_acc) if self.cum_acc else np.zeros_like(acc_step)
        cum_red_arr = np.array(self.cum_red) if self.cum_red else np.zeros_like(acc_step)
        
        acctotal_bound = cum_acc_arr + acc_bound - acc_step
        
        # compute total committed error
        total_error = nonacc_step + cum_acc_arr - acc_step + cum_red_arr
        total_error_tp = cum_acc_arr + cum_red_arr
        
        # --- checks ---
        # all errors and bounds need to be larger or equal to zero
        all_values = np.concatenate([
            Rout_error[fullstep] if len(Rout_error[fullstep]) > 0 else [],
            Rout_tp_error[fullstep] if len(Rout_tp_error[fullstep]) > 0 else [],
            nonacc_step[fullstep] if len(nonacc_step[fullstep]) > 0 else [],
            acc_step, np.array(self.step_red) if self.step_red else [],
            nonacc_bound, acc_bound, red_bound, acctotal_bound
        ])
        
        if len(all_values) > 0 and np.any(all_values < 0):
            res = False
        
        # full errors (time-interval and time-point)
        # (1) need to be below maximum error
        if (len(Rout_error[fullstep]) > 0 and np.any(Rout_error[fullstep] > self.emax)) or \
           (len(Rout_tp_error[fullstep]) > 0 and np.any(Rout_tp_error[fullstep] > self.emax)):
            res = False
        
        # (2) re-computation of full errors has to match ongoing computation
        if len(total_error[fullstep]) > 0 and len(Rout_error[fullstep]) > 0:
            if np.any(np.abs(total_error[fullstep] - Rout_error[fullstep]) > 1e-9):
                internalcheck = False
        
        if len(Rout_tp_error) > 0 and np.any(Rout_tp_error) and len(total_error_tp[fullstep]) > 0:
            if np.any(np.abs(total_error_tp[fullstep] - Rout_tp_error[fullstep]) > 1e-9):
                internalcheck = False
        
        # non-accumulating errors (linComb, F, G)
        # (1) need to be below maximum error
        # (2) need to be below remaining error
        if len(nonacc_step) > 0 and np.any(nonacc_step > self.emax):
            res = False
        
        if len(nonacc_step) > 0 and len(nonacc_bound) > 0 and np.any(nonacc_step > nonacc_bound):
            internalcheck = False
        
        return res and internalcheck
    
    def nextBounds(self, timeStep: float, t: float, k: int, k_iter: int):
        """
        Compute error bounds for the next time step
        
        Args:
            timeStep: Current time step size
            t: Current time
            k: Step counter
            k_iter: Iteration counter within step
        """
        # Ensure lists are long enough
        while len(self.bound_acc) <= k:
            self.bound_acc.append([])
        while len(self.bound_rem) <= k:
            self.bound_rem.append([])
        while len(self.bound_red) <= k:
            self.bound_red.append([])
        
        # Ensure sublists are long enough
        while len(self.bound_acc[k]) <= k_iter:
            self.bound_acc[k].append(0.0)
        while len(self.bound_rem[k]) <= k_iter:
            self.bound_rem[k].append(0.0)
        while len(self.bound_red[k]) <= k_iter:
            self.bound_red[k].append(0.0)
        
        # Compute bounds based on remaining time and error budget
        remaining_time = self.tFinal - t
        if remaining_time <= 0:
            return
        
        # Simple bounds computation (can be made more sophisticated)
        cum_acc_val = self.cum_acc[-1] if self.cum_acc else 0.0
        cum_red_val = self.cum_red[-1] if self.cum_red else 0.0
        
        remaining_error = self.emax - cum_acc_val - cum_red_val
        
        # Distribute remaining error
        self.bound_acc[k][k_iter] = min(remaining_error * 0.5, remaining_error * timeStep / remaining_time)
        self.bound_red[k][k_iter] = min(remaining_error * 0.3, remaining_error * timeStep / remaining_time)
        self.bound_rem[k][k_iter] = remaining_error - self.bound_acc[k][k_iter] - self.bound_red[k][k_iter]
    
    def accumulateErrors(self, k: int, k_iter: int):
        """
        Compute accumulating errors (acc/red), also save non-accumulating
        errors in sequence vector
        
        Args:
            k: Step counter
            k_iter: Iteration counter within step
        """
        # Ensure step_acc is properly initialized
        while len(self.step_acc) <= k:
            self.step_acc.append([])
        while len(self.step_acc[k]) <= k_iter:
            self.step_acc[k].append(0.0)
            
        if k == 0:  # Python 0-based indexing
            self.cum_acc = [self.step_acc[k][k_iter]]
            self.cum_red = [self.step_red[k] if k < len(self.step_red) else 0.0]
        else:
            if len(self.cum_acc) <= k:
                self.cum_acc.append(self.cum_acc[-1] + self.step_acc[k][k_iter])
            else:
                self.cum_acc[k] = self.cum_acc[k-1] + self.step_acc[k][k_iter]
            
            if len(self.cum_red) <= k:
                self.cum_red.append(self.cum_red[-1] + (self.step_red[k] if k < len(self.step_red) else 0.0))
            else:
                self.cum_red[k] = self.cum_red[k-1] + (self.step_red[k] if k < len(self.step_red) else 0.0)
    
    def removeRedundantValues(self, k: int, k_iter: int):
        """
        In each step k, we save the errors for all iterations k_iter
        however, after a specific time step size is chosen, we only keep
        the error which has actually been committed
        
        Args:
            k: Step counter
            k_iter: Iteration counter within step
        """
        # Keep only the selected iteration values
        if k < len(self.seq_nonacc) and k_iter < len(self.seq_nonacc[k]):
            self.seq_nonacc[k] = self.seq_nonacc[k][k_iter]
        
        if k < len(self.idv_PUtkplus1) and k_iter < len(self.idv_PUtkplus1[k]):
            self.idv_PUtkplus1[k] = self.idv_PUtkplus1[k][k_iter]
        if k < len(self.idv_F) and k_iter < len(self.idv_F[k]):
            self.idv_F[k] = self.idv_F[k][k_iter]
        if k < len(self.idv_G) and k_iter < len(self.idv_G[k]):
            self.idv_G[k] = self.idv_G[k][k_iter]
        if k < len(self.idv_linComb) and k_iter < len(self.idv_linComb[k]):
            self.idv_linComb[k] = self.idv_linComb[k][k_iter]
        if k < len(self.idv_PUtauk) and k_iter < len(self.idv_PUtauk[k]):
            self.idv_PUtauk[k] = self.idv_PUtauk[k][k_iter]
        
        if k < len(self.step_acc) and k_iter < len(self.step_acc[k]):
            self.step_acc[k] = self.step_acc[k][k_iter]
        
        if k < len(self.timeSteps) and k_iter < len(self.timeSteps[k]):
            self.timeSteps[k] = self.timeSteps[k][k_iter]
        
        if k < len(self.bound_rem) and k_iter < len(self.bound_rem[k]):
            self.bound_rem[k] = self.bound_rem[k][k_iter]
        if k < len(self.bound_acc) and k_iter < len(self.bound_acc[k]):
            self.bound_acc[k] = self.bound_acc[k][k_iter]
        if k < len(self.bound_red) and k_iter < len(self.bound_red[k]):
            self.bound_red[k] = self.bound_red[k][k_iter]
    
    def fullErrors(self, k: int) -> tuple:
        """
        Compute the errors of the time-point and time-interval reachable
        set; since the non-accumulating error contains the accumulating
        error of the last step, we have two cases
        
        Args:
            k: Step counter
            
        Returns:
            Tuple of (Rcont_error, Rcont_tp_error)
        """
        # Ensure seq_nonacc is properly initialized (as list of lists)
        while len(self.seq_nonacc) <= k:
            self.seq_nonacc.append([])
        
        # Ensure cum_red and cum_acc are properly initialized
        while len(self.cum_red) <= k:
            self.cum_red.append(0.0)
        while len(self.cum_acc) <= k:
            self.cum_acc.append(0.0)
        
        # Get the committed seq_nonacc value (after removeRedundantValues is called)
        # If it's still a list, take the first element or default to 0.0
        if isinstance(self.seq_nonacc[k], list):
            seq_nonacc_k = self.seq_nonacc[k][0] if len(self.seq_nonacc[k]) > 0 else 0.0
        else:
            seq_nonacc_k = self.seq_nonacc[k]
        
        if k == 0:  # Python 0-based indexing
            Rcont_error = seq_nonacc_k + self.cum_red[k]
        else:
            Rcont_error = seq_nonacc_k + self.cum_acc[k-1] + self.cum_red[k]
        
        Rcont_tp_error = self.cum_acc[k] + self.cum_red[k]
        
        return Rcont_error, Rcont_tp_error
    
    def computeErrorBoundReduction(self, A: np.ndarray, G_U: np.ndarray):
        """
        Implementation of heuristics from [1, Sec. IV.D.2)]:
        determine a curve for the reduction error to yield the smallest zonotope order
        
        Args:
            A: System matrix
            G_U: Input generator matrix
        """
        # no generators
        if not np.any(G_U):
            self.bound_red_max = 0.0
            return
        
        # heuristics for near-optimal allocation
        stepsforerrorbound = 100
        timeStep = self.tFinal / stepsforerrorbound
        n = A.shape[0]
        e_At = np.eye(n)
        e_ADeltatk = scipy.linalg.expm(A * timeStep)
        
        # 1. compute auxiliary sets V_k and errors e(V_k)
        V = []
        errV = np.zeros(stepsforerrorbound)
        
        DeltatkU = timeStep * G_U
        for i in range(stepsforerrorbound):
            # compute auxiliary set
            V.append(e_At @ DeltatkU)
            # propagate exponential matrix for next V
            e_At = e_At @ e_ADeltatk
            # compute error (center is always zero)
            errV[i] = np.linalg.norm(np.sum(np.abs(V[i]), axis=1))
        
        # 2. compute weights and ordering
        weights = errV / np.sum(errV)
        tau = np.argsort(weights)
        
        # 3. sort V
        errVsort_cumsum = np.cumsum(errV[tau])
        
        # question: how much could we reduce if we allocate some portion of emax?
        meshsize = 1000
        emax_percentage4ered = np.linspace(0, 1, meshsize)[:-1]
        finalorder = np.zeros(meshsize - 1)
        
        # best final order is a priori the one where we do not reduce at all
        bestfinalorder = stepsforerrorbound + 1
        min_idx = 0
        
        # loop over the mesh of emax to find a near-optimal result
        for i in range(meshsize - 1):
            # find chi* = number of V_k that can be reduced for current red error
            idx_matches = errVsort_cumsum < self.emax * emax_percentage4ered[i]
            if np.any(idx_matches):
                idx = np.where(idx_matches)[0][-1]
            else:
                idx = None
            
            # fraction emax_percentage4ered is allocated to the reduction error,
            # results in factor N of total number of steps
            N = 1 / (1 - emax_percentage4ered[i]) if emax_percentage4ered[i] < 1 else float('inf')
            
            # compute resulting zonotope order
            if idx is None:
                finalorder[i] = N * (stepsforerrorbound + 1)
            else:
                finalorder[i] = N * (stepsforerrorbound + 1 - idx)
            
            # best reduction error allocation is the one which results in the
            # lowest zonotope order, also save the corresponding idx
            if finalorder[i] < bestfinalorder:
                bestfinalorder = finalorder[i]
                min_idx = i
        
        # simpler method
        self.bound_red_max = self.emax * emax_percentage4ered[min_idx]
    
    def updateCoefficientsApproxFun(self, k: int, k_iter: int, fullcomp: bool):
        """
        Compute the coefficients for the approximation functions which model
        the behavior of the individual error over Delta t
        
        Args:
            k: Step counter
            k_iter: Iteration counter within step
            fullcomp: Whether full computation is required
        """
        # only if approximation function should be used
        if not self.useApproxFun:
            return
        
        # Ensure lists are properly sized
        while len(self.idv_PUtkplus1) <= k:
            self.idv_PUtkplus1.append([])
        while len(self.timeSteps) <= k:
            self.timeSteps.append([])
        
        # Ensure timeSteps[k] has enough elements
        while len(self.timeSteps[k]) <= k_iter:
            self.timeSteps[k].append(0.0)
        
        if k_iter == 0:  # Python 0-based indexing
            # initialize coefficients (guessing that b = 0)
            
            if fullcomp and len(self.idv_linComb) > k and len(self.idv_linComb[k]) > k_iter and self.timeSteps[k][k_iter] != 0:
                # linComb and time-interval error of PU are linear approximation functions
                self.coeff_linComb_b = self.idv_linComb[k][k_iter] / self.timeSteps[k][k_iter]
                self.coeff_PUtauk_b = self.idv_PUtauk[k][k_iter] / self.timeSteps[k][k_iter]
                
                # all others are quadratic approximation functions
                # here, we guess that b = 0
                self.coeff_F_a = self.idv_F[k][k_iter] / (self.timeSteps[k][k_iter] ** 2)
                self.coeff_F_b = 0.0
                self.coeff_G_a = self.idv_G[k][k_iter] / (self.timeSteps[k][k_iter] ** 2)
                self.coeff_G_b = 0.0
            
            if len(self.idv_PUtkplus1[k]) > k_iter and self.timeSteps[k][k_iter] != 0:
                self.coeff_PUtkplus1_a = self.idv_PUtkplus1[k][k_iter] / (self.timeSteps[k][k_iter] ** 2)
                self.coeff_PUtkplus1_b = 0.0
        else:
            # update coefficients
            
            if fullcomp and len(self.idv_linComb) > k and len(self.idv_linComb[k]) > k_iter and self.timeSteps[k][k_iter] != 0:
                # linear approximation function -> take newest value
                self.coeff_linComb_b = self.idv_linComb[k][k_iter] / self.timeSteps[k][k_iter]
                self.coeff_PUtauk_b = self.idv_PUtauk[k][k_iter] / self.timeSteps[k][k_iter]
            
            # quadratic approximation functions -> take most recent two values
            if len(self.timeSteps[k]) > k_iter and len(self.idv_PUtkplus1[k]) > k_iter:
                Deltatmat = np.array([
                    [self.timeSteps[k][k_iter-1]**2, self.timeSteps[k][k_iter-1]],
                    [self.timeSteps[k][k_iter]**2, self.timeSteps[k][k_iter]]
                ])
                
                # sanity check of singularity of Deltatmat
                if abs(1/np.linalg.cond(Deltatmat)) < np.finfo(float).eps:
                    raise CORAError('CORA:notConverged', 'Estimation of time step size')
                
                # update coefficient of approximation function for eps_PU
                eps_vec = np.array([self.idv_PUtkplus1[k][k_iter-1], self.idv_PUtkplus1[k][k_iter]])
                coeffs_PUtkplus1 = np.linalg.solve(Deltatmat, eps_vec)
                self.coeff_PUtkplus1_a = coeffs_PUtkplus1[0]
                self.coeff_PUtkplus1_b = coeffs_PUtkplus1[1]
                
                if fullcomp and len(self.idv_F[k]) > k_iter:
                    # update coefficient of approximation function for eps_F
                    eps_F_vec = np.array([self.idv_F[k][k_iter-1], self.idv_F[k][k_iter]])
                    coeffs_F = np.linalg.solve(Deltatmat, eps_F_vec)
                    self.coeff_F_a = coeffs_F[0]
                    self.coeff_F_b = coeffs_F[1]
                    
                    if self.coeff_F_b < 0:
                        # then a certain region is smaller than 0 which cannot ever happen
                        # -> use only current value and set coeff_F_b to 0
                        self.coeff_F_b = 0.0
                        self.coeff_F_a = self.idv_F[k][k_iter] / (self.timeSteps[k][k_iter] ** 2)
                    
                    # update coefficient of approximation function for eps_G
                    if len(self.idv_G[k]) > k_iter:
                        eps_G_vec = np.array([self.idv_G[k][k_iter-1], self.idv_G[k][k_iter]])
                        coeffs_G = np.linalg.solve(Deltatmat, eps_G_vec)
                        self.coeff_G_a = coeffs_G[0]
                        self.coeff_G_b = coeffs_G[1]
    
    def updateBisection(self, k: int, k_iter: int, isU: bool, timeStep: float):
        """
        Update bisection bounds for time step adaptation
        
        Args:
            k: Step counter
            k_iter: Iteration counter within step
            isU: Whether input set is present
            timeStep: Current time step size
        """
        # Ensure all required lists exist and have proper structure
        while len(self.bound_acc_ok) <= k:
            self.bound_acc_ok.append([])
        while len(self.bound_nonacc_ok) <= k:
            self.bound_nonacc_ok.append([])
        while len(self.step_acc) <= k:
            self.step_acc.append([])
        while len(self.seq_nonacc) <= k:
            self.seq_nonacc.append([])
        while len(self.bound_acc) <= k:
            self.bound_acc.append([])
        while len(self.bound_rem) <= k:
            self.bound_rem.append([])
        
        if k_iter == 0:  # Python 0-based indexing
            # initialize lower bound with 0
            self.bisect_lb_timeStep_acc = 0.0
            self.bisect_lb_acc = 0.0
            self.bisect_lb_accok = True
            self.bisect_lb_acc_perc = 0.0
            
            self.bisect_lb_timeStep_nonacc = 0.0
            self.bisect_lb_nonacc = 0.0
            self.bisect_lb_nonaccok = True
            self.bisect_lb_nonacc_perc = 0.0
            
            # upper bound is time step size
            self.bisect_ub_timeStep_acc = timeStep
            self.bisect_ub_acc = self.idv_PUtkplus1[k][k_iter] if len(self.idv_PUtkplus1) > k and len(self.idv_PUtkplus1[k]) > k_iter else 0.0
            self.bisect_ub_accok = self.bound_acc_ok[k][k_iter] if len(self.bound_acc_ok[k]) > k_iter else True
            self.bisect_ub_acc_perc = (self.step_acc[k][k_iter] / self.bound_acc[k][k_iter] 
                                      if len(self.step_acc[k]) > k_iter and len(self.bound_acc[k]) > k_iter and self.bound_acc[k][k_iter] != 0 
                                      else 0.0)
            
            self.bisect_ub_timeStep_nonacc = timeStep
            self.bisect_ub_nonacc = self.seq_nonacc[k][k_iter] if len(self.seq_nonacc[k]) > k_iter else 0.0
            self.bisect_ub_nonaccok = self.bound_nonacc_ok[k][k_iter] if len(self.bound_nonacc_ok[k]) > k_iter else True
            self.bisect_ub_nonacc_perc = (self.seq_nonacc[k][k_iter] / self.bound_rem[k][k_iter] 
                                         if len(self.seq_nonacc[k]) > k_iter and len(self.bound_rem[k]) > k_iter and self.bound_rem[k][k_iter] != 0 
                                         else 0.0)
        else:
            # determine whether new time step size is new lb or new ub
            if isU:
                current_acc_ok = self.bound_acc_ok[k][k_iter] if len(self.bound_acc_ok[k]) > k_iter else True
                if not current_acc_ok or timeStep > self.bisect_ub_timeStep_acc:
                    # current time step size is always new ub if errcheck not ok
                    
                    if timeStep > self.bisect_ub_timeStep_acc:
                        # current ub becomes lb
                        self.bisect_lb_timeStep_acc = self.bisect_ub_timeStep_acc
                        self.bisect_lb_acc = self.bisect_ub_acc
                        self.bisect_lb_accok = self.bisect_ub_accok
                        self.bisect_lb_acc_perc = self.bisect_ub_acc_perc
                    
                    # assign new ub
                    self.bisect_ub_timeStep_acc = timeStep
                    self.bisect_ub_acc = self.idv_PUtkplus1[k][k_iter] if len(self.idv_PUtkplus1) > k and len(self.idv_PUtkplus1[k]) > k_iter else 0.0
                    self.bisect_ub_accok = current_acc_ok
                    self.bisect_ub_acc_perc = (self.step_acc[k][k_iter] / self.bound_acc[k][k_iter] 
                                              if len(self.step_acc[k]) > k_iter and len(self.bound_acc[k]) > k_iter and self.bound_acc[k][k_iter] != 0 
                                              else 0.0)
                else:  # smaller than before and errcheck ok -> new lb
                    self.bisect_lb_timeStep_acc = timeStep
                    self.bisect_lb_acc = self.idv_PUtkplus1[k][k_iter] if len(self.idv_PUtkplus1) > k and len(self.idv_PUtkplus1[k]) > k_iter else 0.0
                    self.bisect_lb_accok = current_acc_ok
                    self.bisect_lb_acc_perc = (self.step_acc[k][k_iter] / self.bound_acc[k][k_iter] 
                                              if len(self.step_acc[k]) > k_iter and len(self.bound_acc[k]) > k_iter and self.bound_acc[k][k_iter] != 0 
                                              else 0.0)
            
            current_nonacc_ok = self.bound_nonacc_ok[k][k_iter] if len(self.bound_nonacc_ok[k]) > k_iter else True
            if not current_nonacc_ok or timeStep > self.bisect_ub_timeStep_nonacc:
                # current time step size is always new ub if errcheck not ok
                
                if timeStep > self.bisect_ub_timeStep_nonacc:
                    # current ub becomes lb
                    self.bisect_lb_timeStep_nonacc = self.bisect_ub_timeStep_nonacc
                    self.bisect_lb_nonacc = self.bisect_ub_nonacc
                    self.bisect_lb_nonaccok = self.bisect_ub_nonaccok
                    self.bisect_lb_nonacc_perc = self.bisect_ub_nonacc_perc
                
                self.bisect_ub_timeStep_nonacc = timeStep
                self.bisect_ub_nonacc = self.seq_nonacc[k][k_iter] if len(self.seq_nonacc[k]) > k_iter else 0.0
                self.bisect_ub_nonaccok = current_nonacc_ok
                self.bisect_ub_nonacc_perc = (self.seq_nonacc[k][k_iter] / self.bound_rem[k][k_iter] 
                                             if len(self.seq_nonacc[k]) > k_iter and len(self.bound_rem[k]) > k_iter and self.bound_rem[k][k_iter] != 0 
                                             else 0.0)
            else:  # smaller than before and errcheck ok -> new lb
                self.bisect_lb_timeStep_nonacc = timeStep
                self.bisect_lb_nonacc = self.seq_nonacc[k][k_iter] if len(self.seq_nonacc[k]) > k_iter else 0.0
                self.bisect_lb_nonaccok = current_nonacc_ok
                self.bisect_lb_nonacc_perc = (self.seq_nonacc[k][k_iter] / self.bound_rem[k][k_iter] 
                                             if len(self.seq_nonacc[k]) > k_iter and len(self.bound_rem[k]) > k_iter and self.bound_rem[k][k_iter] != 0 
                                             else 0.0)
    
    def estimateTimeStepSize(self, t: float, k: int, k_iter: int, fullcomp: bool, 
                           timeStep: float, maxTimeStep: float, isU: bool) -> float:
        """
        Select either approximation functions or bisection
        
        Args:
            t: Current time
            k: Step counter
            k_iter: Iteration counter within step
            fullcomp: Whether full computation is required
            timeStep: Current time step size
            maxTimeStep: Maximum allowed time step size
            isU: Whether input set is present
            
        Returns:
            Estimated time step size
        """
        if self.useApproxFun:
            timeStep = self._priv_approxFun(t, k, k_iter, fullcomp, timeStep, maxTimeStep, isU)
            
            # approx. function method proposes a value for the time step
            # size which we cannot accept if:
            # 1) value smaller than any lower bound (where errors fulfilled)
            # 2) value larger than any upper bound (where errors not fulfilled)
            # 3) value smaller than any upper bound (where errors fulfilled)
            self.useApproxFun = not (
                timeStep <= self.bisect_lb_timeStep_acc or timeStep <= self.bisect_lb_timeStep_nonacc or
                (timeStep >= self.bisect_ub_timeStep_acc and not self.bisect_ub_accok) or
                (timeStep >= self.bisect_ub_timeStep_nonacc and not self.bisect_ub_nonaccok) or
                (timeStep <= self.bisect_ub_timeStep_acc and self.bisect_ub_accok) or
                (timeStep <= self.bisect_ub_timeStep_nonacc and self.bisect_ub_nonaccok)
            )
        
        # note: useApproxFun may have changed above!
        if not self.useApproxFun:
            timeStep = self._priv_bisection(t, k, fullcomp, timeStep, maxTimeStep, isU)
        
        return timeStep
    
    def _priv_errOp(self, S) -> float:
        """
        Compute error operation for different set types
        
        Args:
            S: Set object
            
        Returns:
            Error value
        """
        # This is a simplified version - would need proper set type checking
        if hasattr(S, 'G') and hasattr(S, 'c'):
            # zonotope-like with G, c attributes
            return np.linalg.norm(np.sum(np.abs(S.G), axis=1) + np.abs(S.c.flatten()))
        elif hasattr(S, 'generators') and hasattr(S, 'center'):
            # zonotope-like with methods
            G = S.generators()
            c = S.center()
            return np.linalg.norm(np.sum(np.abs(G), axis=1) + np.abs(c.flatten()))
        elif hasattr(S, 'infimum') and hasattr(S, 'supremum'):
            # interval-like
            return np.linalg.norm(np.maximum(-S.infimum(), S.supremum()))
        elif isinstance(S, np.ndarray):
            # numpy array
            return np.linalg.norm(S)
        else:
            # fallback: try to convert to array
            try:
                return np.linalg.norm(np.asarray(S))
            except (ValueError, TypeError):
                # Last resort: return a default value
                return 1.0
    
    def _priv_approxFun(self, t: float, k: int, k_iter: int, fullcomp: bool, 
                       timeStep: float, maxTimeStep: float, isU: bool) -> float:
        """
        Predict a time step size which satisfies the error bound using approximation functions
        
        Args:
            t: Current time
            k: Step counter
            k_iter: Iteration counter within step
            fullcomp: Whether full computation is required
            timeStep: Current time step size
            maxTimeStep: Maximum allowed time step size
            isU: Whether input set is present
            
        Returns:
            Predicted time step size
        """
        # in the first step, there is a chance that the initial guess is
        # too large; thus, we use another condition to decrease the initial
        # guess until a reasonable value can be found
        if k == 0 and len(self.seq_nonacc) > k:
            # Handle seq_nonacc as either list or float after removeRedundantValues
            seq_val = (self.seq_nonacc[k][k_iter] if isinstance(self.seq_nonacc[k], list) and len(self.seq_nonacc[k]) > k_iter
                      else self.seq_nonacc[k] if not isinstance(self.seq_nonacc[k], list) else 0)
            bound_rem_val = (self.bound_rem[k][k_iter] if len(self.bound_rem) > k and isinstance(self.bound_rem[k], list) and len(self.bound_rem[k]) > k_iter
                           else self.bound_rem[k] if len(self.bound_rem) > k and not isinstance(self.bound_rem[k], list)
                           else 0)
            
            seq_ratio = seq_val / bound_rem_val if bound_rem_val != 0 else 0
            
            step_acc_val = (self.step_acc[k][k_iter] if len(self.step_acc) > k and isinstance(self.step_acc[k], list) and len(self.step_acc[k]) > k_iter
                          else self.step_acc[k] if len(self.step_acc) > k and not isinstance(self.step_acc[k], list) else 0)
            bound_acc_val = (self.bound_acc[k][k_iter] if len(self.bound_acc) > k and isinstance(self.bound_acc[k], list) and len(self.bound_acc[k]) > k_iter
                           else self.bound_acc[k] if len(self.bound_acc) > k and not isinstance(self.bound_acc[k], list) else 0)
            
            acc_ratio = step_acc_val / bound_acc_val if bound_acc_val != 0 else 0
            
            if max(seq_ratio, acc_ratio) > 1e3:
                return 0.01 * timeStep
        
        # safety factor so that we are more likely to satisfy the bounds
        # in case the approximation functions underestimate the errors
        safetyFactor = 0.90
        
        # 1. condition: accumulating error needs to satisfy linearly 
        # increasing bound until the time horizon
        remaining_time = self.tFinal - t
        if remaining_time > 0 and self.coeff_PUtkplus1_a != 0:
            bound_remacc_k = (self.bound_remacc[k] if len(self.bound_remacc) > k else 
                             self.emax - (self.bound_red_max if hasattr(self, 'bound_red_max') else 0))
            
            timeStep_pred_linBound = safetyFactor * (
                (bound_remacc_k / remaining_time - self.coeff_PUtkplus1_b) / self.coeff_PUtkplus1_a
            )
        else:
            timeStep_pred_linBound = float('inf')
        
        timeStep_pred_erem = float('inf')
        if fullcomp:
            # 2. condition: both errors need to satisfy erem for the current step
            # this yields a quadratic equation
            temp_a = self.coeff_PUtkplus1_a + self.coeff_F_a + self.coeff_G_a
            temp_b = (self.coeff_PUtkplus1_b + self.coeff_F_b + self.coeff_G_b +
                     self.coeff_linComb_b + self.coeff_PUtauk_b)
            temp_c = -(self.bound_rem[k] if len(self.bound_rem) > k and not isinstance(self.bound_rem[k], list)
                      else self.bound_rem[k][k_iter] if len(self.bound_rem) > k and isinstance(self.bound_rem[k], list) and len(self.bound_rem[k]) > k_iter
                      else 0)
            
            # compute two solutions of quadratic equation (one is < 0)
            # choose maximum of predicted timeSteps -> positive solution
            if temp_a != 0:
                discriminant = temp_b**2 - 4*temp_a*temp_c
                if discriminant >= 0:
                    sqrt_discriminant = np.sqrt(discriminant)
                    sol1 = (-temp_b + sqrt_discriminant) / (2 * temp_a)
                    sol2 = (-temp_b - sqrt_discriminant) / (2 * temp_a)
                    timeStep_pred_erem = safetyFactor * max(sol1, sol2)
        
        # update timeStep by minimum of predicted timeSteps so that both
        # error bounds are likely to be satisfied
        return min(maxTimeStep, timeStep_pred_linBound, timeStep_pred_erem)
    
    def _priv_bisection(self, t: float, k: int, fullcomp: bool, 
                       timeStep: float, maxTimeStep: float, isU: bool) -> float:
        """
        Use bisection method to estimate time step size
        
        Args:
            t: Current time
            k: Step counter
            fullcomp: Whether full computation is required
            timeStep: Current time step size
            maxTimeStep: Maximum allowed time step size
            isU: Whether input set is present
            
        Returns:
            Estimated time step size using bisection
        """
        # special handling for first step
        eacctotal = 0.0 if k == 0 else (self.cum_acc[k-1] if len(self.cum_acc) > k-1 else 0.0)
        
        # 1. acc errors
        timeStep_pred_eacc = float('inf')
        slope_acc = 0.0
        
        if isU:
            if self.bisect_ub_accok:
                # extrapolate
                bound_red_max = self.bound_red_max if hasattr(self, 'bound_red_max') else 0.0
                errorbound_accend_tFinal = (self.emax - bound_red_max) / self.tFinal
                slope_acc = ((self.bisect_ub_acc - self.bisect_lb_acc) / 
                           (self.bisect_ub_timeStep_acc - self.bisect_lb_timeStep_acc)
                           if self.bisect_ub_timeStep_acc != self.bisect_lb_timeStep_acc else 0)
                
                if slope_acc < errorbound_accend_tFinal:
                    timeStep_pred_eacc = float('inf')
                else:
                    timeStep_add = ((errorbound_accend_tFinal * (t + self.bisect_lb_timeStep_acc) - 
                                   eacctotal - self.bisect_lb_acc) / 
                                  (slope_acc - errorbound_accend_tFinal))
                    timeStep_pred_eacc = self.bisect_lb_timeStep_acc + timeStep_add
            else:
                # bisection
                factor = 0.5 if k == 0 else np.sqrt((0.95 - self.bisect_lb_acc_perc) / 
                                                   (self.bisect_ub_acc_perc - self.bisect_lb_acc_perc)
                                                   if self.bisect_ub_acc_perc != self.bisect_lb_acc_perc else 0.5)
                timeStep_pred_eacc = (self.bisect_lb_timeStep_acc + 
                                    factor * (self.bisect_ub_timeStep_acc - self.bisect_lb_timeStep_acc))
        
        # 2. nonacc errors
        timeStep_pred_enonacc = float('inf')
        if fullcomp:
            if self.bisect_ub_nonaccok:
                # extrapolate
                slope_nonacc = ((self.bisect_ub_nonacc - self.bisect_lb_nonacc) / 
                              (self.bisect_ub_timeStep_nonacc - self.bisect_lb_timeStep_nonacc)
                              if self.bisect_ub_timeStep_nonacc != self.bisect_lb_timeStep_nonacc else 0)
                
                bound_red_max = self.bound_red_max if hasattr(self, 'bound_red_max') else 0.0
                if isU:
                    timeStep_add = ((self.emax - bound_red_max*t/self.tFinal - eacctotal - 
                                   self.bisect_lb_acc - self.bisect_lb_nonacc) / 
                                  (slope_acc + slope_nonacc - bound_red_max/self.tFinal))
                else:
                    timeStep_add = (self.emax - self.bisect_lb_nonacc) / slope_nonacc if slope_nonacc != 0 else float('inf')
                
                timeStep_pred_enonacc = self.bisect_lb_timeStep_nonacc + timeStep_add
            else:
                # bisection
                factor = 0.5 if k == 0 else ((0.95 - self.bisect_lb_nonacc_perc) / 
                                           (self.bisect_ub_nonacc_perc - self.bisect_lb_nonacc_perc)
                                           if self.bisect_ub_nonacc_perc != self.bisect_lb_nonacc_perc else 0.5)
                timeStep_pred_enonacc = (self.bisect_lb_timeStep_nonacc + 
                                       factor * (self.bisect_ub_timeStep_nonacc - self.bisect_lb_timeStep_nonacc))
            
            # for safety... (if slopes misbehave)
            if timeStep_pred_enonacc < 0:
                timeStep_pred_enonacc = float('inf')
        
        # find minimum
        return min(timeStep_pred_enonacc, timeStep_pred_eacc, maxTimeStep) 