import numpy as np
from cora_python.g.classes.simResult.simResult import SimResult

class CoraTestCase:
    """
    testCase - class that stores a test case for conformance testing and
    reachset-conformant identification (see Def. 6 in [1]; first
    publication on this topic was [2]).

    Syntax:
        obj = testCase(y,u,x,dt)
        obj = testCase(y,u,x0,dt)
        obj = testCase(y,u,x,dt,name)
        obj = testCase(y,u,x0,dt,name)

    Inputs:
        y - (a x q x s) vector of the measured outputs samples
        u - (a x p x s) vector of input samples
        x - (a x n) vector of state samples
        x0 - (n x 1 x s) vector of initial states
        dt - sampling time
        name - name of the test case

    Outputs:
        obj - generated testCase object

    References:
        [1] Liu et al., "Guarantees for Real Robotic Systems: Unifying Formal
            Controller Synthesis and Reachset-Conformant Identification", 2022
        [2] M. Althoff and J. M. Dolan. Reachability computation of low-order
            models for the safety verification of high-order road vehicle
            models. In Proc. of the American Control Conference,
            page 3559–3566, 2012.
    """

    def __init__(self, *args):
        if len(args) == 4:
            y = args[0]
            u = args[1]
            x = args[2]
            dt = args[3]
            name = None
            model = None
        elif len(args) == 5:
            y = args[0]
            u = args[1]
            x = args[2]
            dt = args[3]
            name = args[4]
            model = None
        elif len(args) == 6:
            y = args[0]
            u = args[1]
            x = args[2]
            dt = args[3]
            name = args[4]
            model = args[5]
        else:
            raise ValueError("Invalid number of arguments")
        
        if isinstance(y, list):
            # transform measurement cell array to 3D array
            y_arr = np.array(y)
            # Assuming list of 2D arrays, stack them along the third axis
            self.y = np.stack(y_arr, axis=-1)
        else:
            self.y = y
        
        self.u = u
        self.sampleTime = dt
        self.name = name
        self.model = model

        # handle the states
        if len(x.shape) == 3 and x.shape[0] > 1: # trajectories
            self.x = x
            # Slicing with x[0:1, :, :] keeps the dimension, resulting in (1, dim, traj)
            initial_slice = x[0:1, :, :]
            # Now we can transpose from (1, dim, traj) to (dim, 1, traj)
            self.initialState = np.transpose(initial_slice, (1, 0, 2))
        else: # initial state is uncertain or single trajectory
            self.x = None
            self.initialState = x
        
        self.y_a = None # deviation of measured outputs from the nominal

    def set_u(self, u):
        self.u = u
        return self

    def compute_ya(self, sys):
        """
        compute measurement deviation y_a by subtracting the nominal
        solution from the measurement trajectory
        """
        y_nom = np.zeros_like(self.y)
        params = {'tFinal': sys.dt * self.y.shape[0] - sys.dt}

        for s in range(self.y.shape[2]):
            if self.initialState.shape[2] == 1:
                params['x0'] = self.initialState
            else:
                params['x0'] = self.initialState[:,:,s]

            if self.u.shape[2] == 1:
                # If only one input trajectory, use it for all simulations
                params['u'] = self.u[:,:,0].T
            else:
                # Different inputs for each trajectory
                params['u'] = self.u[:,:,s].T
            
            # Assuming sys.simulate exists and returns [t, x, z, y]
            _, _, _, y_nom_s = sys.simulate(params)
            y_nom[:,:,s] = y_nom_s

            if self.initialState.shape[2] == 1 and self.u.shape[2] == 1:
                y_nom = np.tile(y_nom_s[:,:,np.newaxis], (1, 1, self.y.shape[2]))
                break
        
        self.y_a = self.y - y_nom
        return self

    def reduceLength(self, n_k):
        self.u = self.u[:n_k,:,:]
        self.y = self.y[:n_k,:,:]
        return self

    def combineTestCases(self, other):
        """
        combine two test cases in one test case to simplify computations
        for linear systems
        """
        self.initialState = np.concatenate((self.initialState, other.initialState), axis=2)

        u_diff = self.u.shape[0] - other.u.shape[0]
        dim_u = self.u.shape[1]

        u1 = np.pad(self.u, ((0, max(-u_diff, 0)), (0, 0), (0, 0)), 'constant', constant_values=np.nan)
        u2 = np.pad(other.u, ((0, max(u_diff, 0)), (0, 0), (0, 0)), 'constant', constant_values=np.nan)
        self.u = np.concatenate((u1, u2), axis=2)

        y_diff = self.y.shape[0] - other.y.shape[0]
        dim_y = self.y.shape[1]

        y1 = np.pad(self.y, ((0, max(-y_diff, 0)), (0, 0), (0, 0)), 'constant', constant_values=np.nan)
        y2 = np.pad(other.y, ((0, max(y_diff, 0)), (0, 0), (0, 0)), 'constant', constant_values=np.nan)
        self.y = np.concatenate((y1, y2), axis=2)
        
        return self

    def setInitialStateToMeas(self, p, tol=1e-12):
        """
        set initial state to the first p measurements (necessary for
        input.output models)
        """
        if np.mean(np.abs(np.diff(self.y[:p,:,:], axis=2))) < tol:
            self.initialState = self.y[:p,:,0].T.flatten()
            return [self]
        else:
            testSuite = []
            n_s = self.y.shape[2]
            for s in range(n_s):
                new_tc = CoraTestCase(self.y[:,:,s], self.u, self.x, self.sampleTime, self.name, self.model)
                new_tc.initialState = self.y[:p,:,s].T.flatten()
                testSuite.append(new_tc)
            return testSuite 