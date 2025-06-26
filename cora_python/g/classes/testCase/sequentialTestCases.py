import numpy as np
import copy

def sequentialTestCases(self, length):
    """
    sequentialTestCases - Creates sequential test cases of a given length,
    starting at every instant of the original test case (see Def. 7 in [1])

    Args:
        length (int): number of samples for each test case

    Returns:
        list: A list of testCase objects
    """

    # sanity checks
    if self is None:
        return [self]
    
    if self.x is None:
        raise ValueError("To create sequential test cases, the state must be available at every sampling instant.")

    nrOfTestCases = self.y.shape[0] - length
    if nrOfTestCases < 1:
        return [self]

    res = []

    for i in range(nrOfTestCases):
        # create a deep copy to avoid modifying the original object
        dataStruct = copy.deepcopy(self)
        
        # update measured outputs
        dataStruct.y = self.y[i:i+length, :]
        
        # update inputs
        if self.u is not None and self.u.size > 0:
            dataStruct.u = self.u[i:i+length, :]
            
        # update states
        dataStruct.x = self.x[i:i+length, :]
        
        # update initial state
        dataStruct.initialState = self.x[i, :].T
        
        res.append(dataStruct)
        
    return res 