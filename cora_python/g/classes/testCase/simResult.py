import numpy as np
# Assuming simResult class will be available from this path
from cora_python.g.classes.simResult.simResult import SimResult

def to_simResult(self):
    """
    to_simResult - converts a test case to a simResult object.

    Returns:
        simResult: A simResult object.
    """
    
    # create time vector
    nrOfTimeSteps = self.y.shape[0]
    dt = self.sampleTime
    tVec = np.arange(0, dt * nrOfTimeSteps, dt)

    # create lists with one entry only
    if self.x is None:
        x = []
    else:
        x = [self.x]
    
    t = [tVec]
    y = [self.y]

    # create simResult object
    simRes = SimResult(x, t, [], y)
    
    return simRes 